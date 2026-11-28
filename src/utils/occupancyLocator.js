import * as THREE from 'three';
import { MAX_CAMS } from './multiProjector.js';
import { MultiCamTracker3D } from './multiCamTracker3D.js';
import { SCREEN_QUAD_VERT } from '../shaders/freeViewpointShaders.js';
import { OCCUPANCY_FRAG, OCCUPANCY_SAMPLE_COUNT, OCCUPANCY_NO_GROUND } from '../shaders/occupancyShaders.js';

const PASS_INTERVAL_MS = 100;
// 两路相机交汇出的占据斑沿视线拉长，同一个人常出现 2~3 个相距 0.5~0.6 米的峰：定位用的 NMS 半径比算法默认值更大
const PEAK_NMS_RADIUS = 0.75;
// 占地大的目标（AGV、托盘车）会在占据斑里出现多个相距较远的峰：按较强峰的占地面积放大抑制半径
const BIG_PEAK_BASE_RADIUS = 0.35;
const BIG_PEAK_RADIUS_PER_SQRT_AREA = 0.45;
// 已稳定的轨迹附近（该距离内）只允许最近的一个占据峰认领它，其余的是同一人的副峰 / 幻影，不另起新轨迹
const MATURE_HITS = 5;
const DUPLICATE_RADIUS = 1.0;
const TRACK_VISIBLE_MS = 300;
const SAMPLE_BASE_HEIGHT = 0.15;
const SAMPLE_STEP = 0.25;
const SMOOTH_LERP = 0.25;
// 占地面积过小，或既矮又不够大的峰是抠像噪点（占地大的矮目标如 AGV 予以保留）
const MIN_PEAK_AREA = 0.12;
const MIN_PEAK_HEIGHT_COUNT = 3;
const LOW_PEAK_MIN_AREA = 0.4;

/**
 * 纯算法：从占据度读回数据中提取地面峰值
 * @param {Uint8Array} data RGBA（R = 占据度，G = 看见的相机数 / MAX_CAMS，B = 连续被占据层数 / 8），第 0 行即读回的首行
 * @param {number} cols
 * @param {number} rows
 * @param {{cellSize:number, threshold?:number, nmsRadius?:number, refineRadius?:number, maxPeaks?:number}} opts
 * @returns {Array<{col:number, row:number, score:number, nCams:number, heightCount:number, area:number}>}
 *   col / row 为小数网格索引（整数 = 单元中心）
 */
export function extractOccupancyPeaks(data, cols, rows, opts) {
  const { cellSize, threshold = 0.5, nmsRadius = 0.5, refineRadius = 0.25, maxPeaks = 24 } = opts;

  const smooth = new Float32Array(cols * rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      let sum = 0;
      let n = 0;
      for (let dr = -1; dr <= 1; dr++) {
        const rr = r + dr;
        if (rr < 0 || rr >= rows) continue;
        for (let dc = -1; dc <= 1; dc++) {
          const cc = c + dc;
          if (cc < 0 || cc >= cols) continue;
          sum += data[(rr * cols + cc) * 4];
          n++;
        }
      }
      smooth[r * cols + c] = sum / n / 255;
    }
  }

  const candidates = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = smooth[r * cols + c];
      if (v < threshold) continue;
      let isMax = true;
      for (let dr = -1; dr <= 1 && isMax; dr++) {
        const rr = r + dr;
        if (rr < 0 || rr >= rows) continue;
        for (let dc = -1; dc <= 1; dc++) {
          const cc = c + dc;
          if (cc < 0 || cc >= cols || (dr === 0 && dc === 0)) continue;
          if (smooth[rr * cols + cc] > v) {
            isMax = false;
            break;
          }
        }
      }
      if (isMax) candidates.push({ col: c, row: r, value: v });
    }
  }
  candidates.sort((a, b) => b.value - a.value);

  const nmsCells = nmsRadius / cellSize;
  const kept = [];
  for (const cand of candidates) {
    if (kept.length >= maxPeaks) break;
    if (kept.every((k) => Math.hypot(k.col - cand.col, k.row - cand.row) >= nmsCells)) kept.push(cand);
  }

  const refineCells = refineRadius / cellSize;
  const areaCells = 1.0 / cellSize;
  return kept.map((k) => {
    let wSum = 0;
    let cSum = 0;
    let rSum = 0;
    let maxB = 0;
    const rMin = Math.max(0, Math.floor(k.row - refineCells));
    const rMax = Math.min(rows - 1, Math.ceil(k.row + refineCells));
    const cMin = Math.max(0, Math.floor(k.col - refineCells));
    const cMax = Math.min(cols - 1, Math.ceil(k.col + refineCells));
    for (let r = rMin; r <= rMax; r++) {
      for (let c = cMin; c <= cMax; c++) {
        if (Math.hypot(c - k.col, r - k.row) > refineCells) continue;
        maxB = Math.max(maxB, data[(r * cols + c) * 4 + 2]);
        const v = smooth[r * cols + c];
        if (v < threshold) continue;
        wSum += v;
        cSum += v * c;
        rSum += v * r;
      }
    }
    const col = wSum > 0 ? cSum / wSum : k.col;
    const row = wSum > 0 ? rSum / wSum : k.row;

    let areaCount = 0;
    const aRMin = Math.max(0, Math.floor(row - areaCells));
    const aRMax = Math.min(rows - 1, Math.ceil(row + areaCells));
    const aCMin = Math.max(0, Math.floor(col - areaCells));
    const aCMax = Math.min(cols - 1, Math.ceil(col + areaCells));
    for (let r = aRMin; r <= aRMax; r++) {
      for (let c = aCMin; c <= aCMax; c++) {
        if (Math.hypot(c - col, r - row) <= areaCells && smooth[r * cols + c] >= threshold) areaCount++;
      }
    }

    return {
      col,
      row,
      score: k.value,
      nCams: Math.round((data[(k.row * cols + k.col) * 4 + 1] / 255) * MAX_CAMS),
      heightCount: Math.round((maxB / 255) * OCCUPANCY_SAMPLE_COUNT),
      area: areaCount * cellSize * cellSize
    };
  });
}

/**
 * 在线多视角占据度定位：每 100 ms 在地面网格上渲染一遍"各相机前景支持度的一票否决占据图"，
 * 提取峰值作为人的落脚点观测，再经卡尔曼 + 匈牙利跟踪赋予稳定 ID。
 * 不依赖单路相机的检测框，因此重叠区的同一个人只会产生一个峰，且单路视线上的重影被其他相机否决。
 */
export class OccupancyLocator {
  /**
   * @param {import('./groundGrid.js').GroundGrid} groundGrid
   * @param {object} matteUniforms 来自 FreeViewpointFusion.matteUniforms 的共享 uniform 引用
   */
  constructor(groundGrid, matteUniforms) {
    this._grid = groundGrid;
    this._liveOn = matteUniforms.uLiveOn;
    this._material = new THREE.ShaderMaterial({
      vertexShader: SCREEN_QUAD_VERT,
      fragmentShader: OCCUPANCY_FRAG,
      uniforms: {
        ...matteUniforms,
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

    this._target = null;
    this._pixels = null;
    this._gridStamp = null;
    this._tracker = new MultiCamTracker3D({ personThreshold: 0.9, maxAgeMs: 600 });
    this._lastPassMs = -Infinity;
    this._obsCounter = 0;
    this._infoOfTrack = new Map(); // 全局 id -> 最近一次匹配到的观测信息 { height, width, claimsPixels }
    this._smoothed = new Map(); // 全局 id -> { height, width }
    this._tracks = [];

    /** 最近一次通道结果，仅供调试输出 */
    this.lastPeaks = [];
    this.lastFootObservations = [];
  }

  clear() {
    this._tracker = new MultiCamTracker3D({ personThreshold: 0.9, maxAgeMs: 600 });
    this._lastPassMs = -Infinity;
    this._infoOfTrack.clear();
    this._smoothed.clear();
    this._tracks = [];
    this.lastPeaks = [];
    this.lastFootObservations = [];
  }

  _syncTarget() {
    const g = this._grid;
    if (this._target && this._target.width === g.cols && this._target.height === g.rows && this._gridStamp === g.heightTexture) return;
    this._target?.dispose();
    this._target = new THREE.WebGLRenderTarget(g.cols, g.rows, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: false,
      generateMipmaps: false
    });
    this._pixels = new Uint8Array(g.cols * g.rows * 4);
    this._gridStamp = g.heightTexture;
    this._material.uniforms.uHeightTex.value = g.heightTexture;
  }

  _runPass(renderer) {
    this._syncTarget();
    const g = this._grid;
    const prevTarget = renderer.getRenderTarget();
    const prevAutoClear = renderer.autoClear;
    renderer.autoClear = true;
    renderer.setRenderTarget(this._target);
    renderer.render(this._quadScene, this._quadCam);
    renderer.readRenderTargetPixels(this._target, 0, 0, g.cols, g.rows, this._pixels);
    renderer.setRenderTarget(prevTarget);
    renderer.autoClear = prevAutoClear;
    return extractOccupancyPeaks(this._pixels, g.cols, g.rows, { cellSize: g.cellSize, nmsRadius: PEAK_NMS_RADIUS });
  }

  /**
   * 该世界位置所在格子是否有地面且恰好只被一路相机看见（单相机区域，占据度不可信，改用单目落脚点）
   */
  _isSingleCamCell(x, z) {
    const g = this._grid;
    const col = Math.floor((x - g.origin.x) / g.cellSize);
    const row = Math.floor((z - g.origin.y) / g.cellSize);
    if (col < 0 || col >= g.cols || row < 0 || row >= g.rows) return false;
    if (g.heights[row * g.cols + col] < OCCUPANCY_NO_GROUND * 0.5) return false;
    return Math.round((this._pixels[(row * g.cols + col) * 4 + 1] / 255) * MAX_CAMS) === 1;
  }

  _suppressBigObjectPeaks(peaks) {
    const kept = [];
    for (const peak of peaks) {
      const shadowed = kept.some((k) => {
        const radius = Math.max(PEAK_NMS_RADIUS, BIG_PEAK_BASE_RADIUS + BIG_PEAK_RADIUS_PER_SQRT_AREA * Math.sqrt(k.area));
        return Math.hypot(k.col - peak.col, k.row - peak.row) * this._grid.cellSize < radius;
      });
      if (!shadowed) kept.push(peak);
    }
    return kept;
  }

  _dropDuplicatePeaks(observations, infoByKey) {
    for (const track of this._tracker.tracks.values()) {
      if (track.hits < MATURE_HITS) continue;
      let nearest = null;
      let nearestDist = Infinity;
      const near = [];
      for (const obs of observations) {
        const d = Math.hypot(obs.x - track.x, obs.z - track.z);
        if (d >= DUPLICATE_RADIUS) continue;
        near.push(obs);
        if (d < nearestDist) {
          nearestDist = d;
          nearest = obs;
        }
      }
      for (const obs of near) {
        if (obs === nearest) continue;
        observations.splice(observations.indexOf(obs), 1);
        infoByKey.delete(`${obs.camId}:${obs.localTrackId}`);
      }
    }
  }

  /**
   * @param {THREE.WebGLRenderer} renderer
   * @param {number} nowMs
   * @param {() => Array} getFootObservations 单目落脚点观测（仅在本次真正执行占据度通道时调用）
   * @returns {Array} 三维目标轨迹 [{id, cls, x, y, z, height, width, hits, exactTrack, claimsPixels}]
   */
  update(renderer, nowMs, getFootObservations) {
    if (nowMs - this._lastPassMs < PASS_INTERVAL_MS) return this._tracks;
    this._lastPassMs = nowMs;

    const g = this._grid;
    const anyLive = this._liveOn.value.some((v) => v > 0.5);
    const passRan = g.ensure(renderer) && anyLive;
    const peaks = passRan ? this._suppressBigObjectPeaks(this._runPass(renderer)) : [];
    this.lastPeaks = peaks;

    const observations = [];
    const infoByKey = new Map();
    for (const peak of peaks) {
      if (peak.nCams < 2) continue;
      if (peak.area < MIN_PEAK_AREA || (peak.heightCount < MIN_PEAK_HEIGHT_COUNT && peak.area < LOW_PEAK_MIN_AREA)) continue;
      const { x, z } = g.cellToWorld(peak.col, peak.row);
      const y = g.sampleHeight(peak.col, peak.row);
      const localTrackId = this._obsCounter++;
      observations.push({ camId: 'occ', localTrackId, cls: 'target', x, y, z, distToCam: 1 });
      infoByKey.set(`occ:${localTrackId}`, {
        height: THREE.MathUtils.clamp(
          SAMPLE_BASE_HEIGHT + SAMPLE_STEP * (peak.heightCount - 1) + SAMPLE_STEP * 0.5,
          0.3,
          2.1
        ),
        width:
          peak.heightCount * SAMPLE_STEP >= 1.2
            ? 0.6
            : THREE.MathUtils.clamp(2 * Math.sqrt(peak.area / Math.PI), 0.5, 2.0),
        claimsPixels: true
      });
    }

    this._dropDuplicatePeaks(observations, infoByKey);

    this.lastFootObservations = [];
    if (passRan) {
      for (const obs of getFootObservations()) {
        if (!this._isSingleCamCell(obs.x, obs.z)) continue;
        const localTrackId = this._obsCounter++;
        observations.push({
          camId: obs.camId,
          localTrackId,
          cls: 'target',
          x: obs.x,
          y: obs.y,
          z: obs.z,
          distToCam: obs.distToCam
        });
        infoByKey.set(`${obs.camId}:${localTrackId}`, { height: obs.height, width: obs.width, claimsPixels: false });
        this.lastFootObservations.push(obs);
      }
    }

    const snapshots = this._tracker.update(observations, nowMs);
    const newInfo = new Map();
    for (const [key, info] of infoByKey) {
      const gid = this._tracker.localToGlobal.get(key);
      if (gid === undefined) continue;
      // 同一目标既有占据度峰又有落脚点观测时以占据度为准
      if (!newInfo.has(gid) || info.claimsPixels) newInfo.set(gid, info);
    }
    for (const [gid, info] of newInfo) this._infoOfTrack.set(gid, info);

    const alive = new Set(snapshots.map((s) => s.globalId));
    for (const id of this._infoOfTrack.keys()) if (!alive.has(id)) this._infoOfTrack.delete(id);
    for (const id of this._smoothed.keys()) if (!alive.has(id)) this._smoothed.delete(id);

    this._tracks = [];
    for (const snap of snapshots) {
      if (nowMs - snap.lastSeen > TRACK_VISIBLE_MS) continue;
      const info = this._infoOfTrack.get(snap.globalId);
      if (!info) continue;

      const prev = this._smoothed.get(snap.globalId);
      const smoothed = prev
        ? { height: prev.height + (info.height - prev.height) * SMOOTH_LERP, width: prev.width + (info.width - prev.width) * SMOOTH_LERP }
        : { height: info.height, width: info.width };
      this._smoothed.set(snap.globalId, smoothed);

      this._tracks.push({
        id: snap.globalId,
        cls: 'target',
        x: snap.x,
        y: snap.y,
        z: snap.z,
        height: smoothed.height,
        width: smoothed.width,
        hits: snap.hits,
        exactTrack: false,
        claimsPixels: info.claimsPixels
      });
    }
    return this._tracks;
  }

  dispose() {
    this._target?.dispose();
    this._material.dispose();
    this._quad.geometry.dispose();
  }
}
