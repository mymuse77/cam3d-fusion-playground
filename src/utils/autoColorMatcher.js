import * as THREE from 'three';
import { MAX_CAMS } from './multiProjector.js';
import { SCREEN_QUAD_VERT } from '../shaders/freeViewpointShaders.js';
import { COLOR_SAMPLE_FRAG } from '../shaders/colorMatchShaders.js';

const SAMPLE_SIZE = 64;
const MIN_SHARED_CELLS = 30;
const GAIN_MIN = 0.6;
const GAIN_MAX = 1.6;
const DARK_LIMIT = 0.02;
const BRIGHT_LIMIT = 0.97;
const SOLVER_ITERATIONS = 300;

/**
 * 纯算法：由两两相机的对数颜色比求每路相机的线性 RGB 增益
 * 逐通道求解加权最小二乘 l_a + ratio_ab - l_b = 0，并在每个连通分量内令 sum(l) = 0（保持整体亮度）；
 * 没有任何配对的相机增益为 1
 * @param {string[]} camIds
 * @param {Array<{a:string, b:string, logRatio:number[], weight:number}>} pairStats logRatio = log(color_a / color_b)
 * @returns {Map<string, number[]>}
 */
export function solveColorGains(camIds, pairStats) {
  const gains = new Map(camIds.map((id) => [id, [1, 1, 1]]));
  const neighbors = new Map(camIds.map((id) => [id, []]));
  for (const pair of pairStats) {
    if (!neighbors.has(pair.a) || !neighbors.has(pair.b) || pair.weight <= 0) continue;
    neighbors.get(pair.a).push({ other: pair.b, sign: -1, pair });
    neighbors.get(pair.b).push({ other: pair.a, sign: 1, pair });
  }

  const visited = new Set();
  for (const root of camIds) {
    if (visited.has(root) || neighbors.get(root).length === 0) continue;
    const component = [];
    const queue = [root];
    visited.add(root);
    while (queue.length) {
      const id = queue.pop();
      component.push(id);
      for (const { other } of neighbors.get(id)) {
        if (!visited.has(other)) {
          visited.add(other);
          queue.push(other);
        }
      }
    }

    for (let ch = 0; ch < 3; ch++) {
      const l = new Map(component.map((id) => [id, 0]));
      for (let iter = 0; iter < SOLVER_ITERATIONS; iter++) {
        for (const id of component) {
          let num = 0;
          let den = 0;
          for (const { other, sign, pair } of neighbors.get(id)) {
            // 对 a：l_a = l_b - ratio；对 b：l_b = l_a + ratio
            num += pair.weight * (l.get(other) + sign * pair.logRatio[ch]);
            den += pair.weight;
          }
          l.set(id, num / den);
        }
      }
      let mean = 0;
      for (const id of component) mean += l.get(id);
      mean /= component.length;
      for (const id of component) {
        gains.get(id)[ch] = THREE.MathUtils.clamp(Math.exp(l.get(id) - mean), GAIN_MIN, GAIN_MAX);
      }
    }
  }
  return gains;
}

function median(values) {
  const sorted = Float64Array.from(values).sort();
  return sorted[sorted.length >> 1];
}

/**
 * 跨相机自动色彩匹配：
 * 把各路相机的静态背景底图投到共同的地面网格上，在多路都能看到的地面点上比较颜色，
 * 解出每路相机的线性 RGB 增益并写回共享 Uniform，消除相机之间曝光 / 白平衡差异造成的接缝。
 * 仅在槽位相机集合、底图就绪状态或色彩调节参数变化时重算；两两统计持久保存，轮换出槽位的相机仍参与求解。
 */
export class AutoColorMatcher {
  /**
   * @param {import('./groundGrid.js').GroundGrid} groundGrid
   * @param {object} sharedUniforms MultiProjectorManager.sharedUniforms
   * @param {{setCalibration:Function}} projectorMgr 用于把增益重新写入激活槽位
   */
  constructor(groundGrid, sharedUniforms, projectorMgr) {
    this._grid = groundGrid;
    this._projectorMgr = projectorMgr;
    this._enabled = true;
    this._signature = null;
    this._pairs = new Map(); // `${a}|${b}`（a < b） -> { a, b, logRatio, weight }
    this._items = new Map(); // camId -> 逻辑相机条目（用于写回 colorGain）
    this._slotCams = [];
    this.lastGains = new Map();

    this._target = new THREE.WebGLRenderTarget(SAMPLE_SIZE, SAMPLE_SIZE, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: false,
      generateMipmaps: false
    });
    this._pixels = new Uint8Array(SAMPLE_SIZE * SAMPLE_SIZE * 4);
    this._material = new THREE.ShaderMaterial({
      vertexShader: SCREEN_QUAD_VERT,
      fragmentShader: COLOR_SAMPLE_FRAG,
      uniforms: {
        uSlot: { value: 0 },
        uBg: { value: null },
        uDepthMap: sharedUniforms.uDepthMap,
        uProjViewMatrix: sharedUniforms.uProjViewMatrix,
        uProjProjectionMatrix: sharedUniforms.uProjProjectionMatrix,
        uIntrinsics: sharedUniforms.uIntrinsics,
        uDistortion: sharedUniforms.uDistortion,
        uColorAdjust: sharedUniforms.uColorAdjust,
        uHeightTex: { value: null },
        uGridOrigin: { value: groundGrid.origin },
        uGridSize: { value: groundGrid.size }
      },
      depthTest: false,
      depthWrite: false
    });
    this._quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this._material);
    this._quad.frustumCulled = false;
    this._quadScene = new THREE.Scene();
    this._quadScene.add(this._quad);
    this._quadCam = new THREE.Camera();
  }

  setEnabled(enabled) {
    if (this._enabled === enabled) return;
    this._enabled = enabled;
    this._signature = null;
    if (!enabled) this._applyGains(new Map(), this._slotCams);
  }

  clear() {
    this._pairs.clear();
    this._items.clear();
    this.lastGains = new Map();
    this._signature = null;
  }

  /**
   * 每帧调用：仅在输入签名变化时执行 GPU 采样与求解
   * @param {THREE.WebGLRenderer} renderer
   * @param {Array} slotCams 调度器当前激活槽位的相机条目（下标即槽位号）
   * @param {object} streamMgr 提供 getBackgroundTexture(camId)
   */
  update(renderer, slotCams, streamMgr) {
    this._slotCams = slotCams;
    if (!this._enabled) return;

    const slots = slotCams.map((item) => ({ item, bg: streamMgr.getBackgroundTexture(item.id) }));
    const signature = slots
      .map(({ item, bg }) => {
        const ca = item.colorAdjust;
        return `${item.id}:${bg ? 1 : 0}:${ca.brightness ?? 0},${ca.contrast ?? 1},${ca.saturation ?? 1}`;
      })
      .join('|');
    if (signature === this._signature) return;
    if (!this._grid.ensure(renderer)) return;
    this._signature = signature;

    for (const { item } of slots) this._items.set(item.id, item);

    const samples = [];
    slots.forEach(({ item, bg }, slot) => {
      if (bg) samples.push({ camId: item.id, rgba: this._sampleSlot(renderer, slot, bg) });
    });
    for (let i = 0; i < samples.length; i++) {
      for (let j = i + 1; j < samples.length; j++) this._accumulatePair(samples[i], samples[j]);
    }

    const camIds = [...this._items.keys()];
    this.lastGains = solveColorGains(camIds, [...this._pairs.values()]);
    this._applyGains(this.lastGains, slotCams);
  }

  _sampleSlot(renderer, slot, bgTexture) {
    const u = this._material.uniforms;
    u.uSlot.value = slot;
    u.uBg.value = bgTexture;
    u.uHeightTex.value = this._grid.heightTexture;

    const prevTarget = renderer.getRenderTarget();
    const prevAutoClear = renderer.autoClear;
    renderer.autoClear = true;
    renderer.setRenderTarget(this._target);
    renderer.render(this._quadScene, this._quadCam);
    renderer.readRenderTargetPixels(this._target, 0, 0, SAMPLE_SIZE, SAMPLE_SIZE, this._pixels);
    renderer.setRenderTarget(prevTarget);
    renderer.autoClear = prevAutoClear;
    return this._pixels.slice();
  }

  _accumulatePair(sa, sb) {
    const logs = [[], [], []];
    const lin = (v) => Math.pow(v / 255, 2.2);
    for (let i = 0; i < SAMPLE_SIZE * SAMPLE_SIZE; i++) {
      const o = i * 4;
      if (sa.rgba[o + 3] === 0 || sb.rgba[o + 3] === 0) continue;
      const ca = [lin(sa.rgba[o]), lin(sa.rgba[o + 1]), lin(sa.rgba[o + 2])];
      const cb = [lin(sb.rgba[o]), lin(sb.rgba[o + 1]), lin(sb.rgba[o + 2])];
      if ([...ca, ...cb].some((v) => v < DARK_LIMIT || v > BRIGHT_LIMIT)) continue;
      for (let ch = 0; ch < 3; ch++) logs[ch].push(Math.log(ca[ch] / cb[ch]));
    }
    if (logs[0].length < MIN_SHARED_CELLS) return;

    const swap = sa.camId > sb.camId;
    const sign = swap ? -1 : 1;
    const a = swap ? sb.camId : sa.camId;
    const b = swap ? sa.camId : sb.camId;
    this._pairs.set(`${a}|${b}`, {
      a,
      b,
      logRatio: logs.map((l) => sign * median(l)),
      weight: logs[0].length
    });
  }

  _applyGains(gains, slotCams) {
    for (const item of this._items.values()) item.colorGain = gains.get(item.id) ?? [1, 1, 1];
    slotCams.forEach((item, slot) => {
      if (slot >= MAX_CAMS) return;
      this._projectorMgr.setCalibration(slot, item.intrinsics, item.distortion, item.colorAdjust, item.colorGain);
    });
  }

  dispose() {
    this._target.dispose();
    this._material.dispose();
    this._quad.geometry.dispose();
  }
}
