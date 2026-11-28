import * as THREE from 'three';
import { MAX_CAMS } from './multiProjector.js';
import { projectCameraPointToPixel } from './bboxProjector.js';
import {
  SCREEN_QUAD_VERT,
  MATTE_RAW_FRAG,
  MATTE_REFINE_FRAG,
  CARD_VERT,
  CARD_FRAG,
  MAX_TARGETS,
  CONTACT_SHADOW_VERT,
  CONTACT_SHADOW_FRAG
} from '../shaders/freeViewpointShaders.js';

const ATLAS_COLS = Math.ceil(Math.sqrt(MAX_CAMS));
const ATLAS_ROWS = Math.ceil(MAX_CAMS / ATLAS_COLS);
const TILE_W = 960;
const TILE_H = 540;
// 主相机切换：候选分数需领先当前主相机超过该边际，且距上次切换满足保持时长；切换后以 CAM_FADE_MS 交叉淡化
const CAM_SWITCH_MARGIN = 0.08;
const CAM_SWITCH_HOLD_MS = 400;
const CAM_FADE_MS = 300;
// 非主相机的补全权重：只在主相机（及淡出中的前一路相机）弃权的像素上起作用
const FILL_WEIGHT = 1e-3;
// 目标躯干沿高度的采样比例，用于判定某路相机能否看到完整的人
const VISIBILITY_HEIGHTS = [0.1, 0.3, 0.5, 0.7, 0.9];
const FRONTAL_MIN = 0.45;
// 面片朝向的指数平滑时间常数（毫秒）
const YAW_SMOOTH_MS = 100;

// 一致性表决的空间容差（米）：外部高精度轨迹只需容忍多路视频不同步与标定误差；
// 纯视觉轨迹还要容忍数十厘米的落脚点误差
const SUPPORT_RADIUS_EXACT = 0.12;
const SUPPORT_RADIUS_VISUAL = 0.3;
// 低于此高度的目标（AGV、托盘车等）改用盖在顶面高度的水平面片：俯拍相机拍到的主要是其顶面
const FLAT_TARGET_HEIGHT = 0.7;

function smoothstep01(t) {
  const x = THREE.MathUtils.clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
}

/**
 * 纯算法：按目标逐个决策主纹理相机（带分数边际与保持时长的滞回）
 * @param {Array<{camId:string, score:number}>} candidates 非空
 * @param {{camId:string, prevCamId:string|null, switchMs:number}|undefined} state 上一帧状态
 * @param {number} nowMs
 * @returns {{camId:string, prevCamId:string|null, switchMs:number}}
 */
export function selectPrimaryCamera(candidates, state, nowMs) {
  let best = candidates[0];
  for (const c of candidates) if (c.score > best.score) best = c;
  const current = state ? candidates.find((c) => c.camId === state.camId) : null;
  if (!current) return { camId: best.camId, prevCamId: null, switchMs: nowMs };
  if (best.camId !== current.camId && best.score > current.score + CAM_SWITCH_MARGIN && nowMs - state.switchMs >= CAM_SWITCH_HOLD_MS) {
    return { camId: best.camId, prevCamId: current.camId, switchMs: nowMs };
  }
  return state;
}

/**
 * 目标承载面片的尺寸（与相机无关）
 * 优先采用轨迹给出的真实占地宽度；否则由检测框估计：
 * 直立目标的检测框宽度常被俯拍透视、影子或粘连目标撑大，按人体身高比例封顶；低矮目标（如 AGV）直接采用检测宽度
 */
function resolveCardExtent(track) {
  const bodyWidth =
    track.footprint ?? (track.height >= 1.2 ? Math.min(track.width, track.height * 0.38) : track.width);
  const bodyRadius = THREE.MathUtils.clamp(bodyWidth * 0.5 + 0.05, 0.3, 1.25);
  // 面片比躯干宽出一圈以容纳伸展的四肢，高出头顶少许以容忍身高误差
  const halfWidth = bodyRadius + 0.2;
  // 低于此高度的目标改用盖在顶面高度的水平面片：俯拍相机拍到的主要是其顶面
  if (track.height < FLAT_TARGET_HEIGHT) {
    return { halfWidth, bodyRadius, height: track.height, flat: true };
  }
  const height = THREE.MathUtils.clamp(track.height + 0.15, FLAT_TARGET_HEIGHT, 2.6);
  return { halfWidth, bodyRadius, height, flat: false };
}

/**
 * 纯算法：解算目标承载面片的尺寸与朝向
 * 直立目标的面片过中轴竖直放置，法向取"主纹理相机方向"与"观察方向"的角平分线：
 * 相机把目标画面投到面片上会被拉宽 1/cos(θ/2)，观察者斜看面片又被压窄 cos(θ/2)，两者恰好抵消，
 * 因此目标从任意方位看宽度都不失真；观察者与某路相机重合时则与该路原始画面逐像素一致。
 * 主相机方向由各相机按给定权重合成，切换相机时权重随时间过渡，法向随之连续转动。
 * @param {{x:number, y:number, z:number, width:number, height:number, footprint?:number}} track
 * @param {Array<{pos:{x:number, y:number, z:number}, weight:number}>} camDirs 能看到该目标的相机位置与合成权重
 * @param {{x:number, y:number, z:number}} viewerPos 当前观察相机位置
 * @returns {{halfWidth:number, bodyRadius:number, height:number, flat:boolean, normal:[number, number]}|null}
 */
export function resolveCardPose(track, camDirs, viewerPos) {
  if (!camDirs.length) return null;

  const extent = resolveCardExtent(track);
  if (extent.flat) return { ...extent, normal: [0, 1] };
  const horizontalDir = (p) => {
    const dx = p.x - track.x;
    const dz = p.z - track.z;
    const lenH = Math.hypot(dx, dz) || 1e-6;
    return [dx / lenH, dz / lenH];
  };

  const viewDh = horizontalDir(viewerPos);
  // 逐相机先取各自的角平分线法向（相机方向翻到观察者一侧，面片正反面等价，法向落在观察方向 45° 以内），
  // 再按权重对相对观察方向的偏角做加权平均：切换时不受"翻面"符号突变影响，法向随权重连续转动
  let weightSum = 0;
  let angleSum = 0;
  for (const { pos, weight } of camDirs) {
    const dh = horizontalDir(pos);
    const sign = dh[0] * viewDh[0] + dh[1] * viewDh[1] < 0 ? -1 : 1;
    const nx = sign * dh[0] + viewDh[0];
    const nz = sign * dh[1] + viewDh[1];
    const nLen = Math.hypot(nx, nz);
    const n = nLen > 1e-6 ? [nx / nLen, nz / nLen] : viewDh;
    angleSum += weight * Math.atan2(viewDh[0] * n[1] - viewDh[1] * n[0], viewDh[0] * n[0] + viewDh[1] * n[1]);
    weightSum += weight;
  }
  const mean = weightSum > 1e-6 ? angleSum / weightSum : 0;
  const normal = [
    viewDh[0] * Math.cos(mean) - viewDh[1] * Math.sin(mean),
    viewDh[0] * Math.sin(mean) + viewDh[1] * Math.cos(mean)
  ];

  return { ...extent, normal };
}

/**
 * 多视角自由视点前景融合管理器：
 * 1. 每帧把各激活相机的实时视频抠成前景抠像图集（背景差分 / AI 掩膜三分图 + 形态学精修）
 * 2. 在每个目标的三维落脚点竖起承载面片，按真实标定把各路实时视频回投其上，
 *    以视点相关权重选择最接近当前观察方向的相机；结合全部目标的三维位置判定像素归属，
 *    并经多数表决剔除目标间串入的重影
 * 3. 目标脚下叠加接触阴影，补回被静态背景底图抹掉的落地感
 */
export class FreeViewpointFusion {
  constructor(sharedUniforms) {
    this.group = new THREE.Group();
    this.group.name = 'FreeViewpointFusion';
    this.group.userData.excludeFromDepth = true;
    this.group.visible = false;
    this.visible = false;
    this.keyThreshold = 0.095;

    const atlasW = ATLAS_COLS * TILE_W;
    const atlasH = ATLAS_ROWS * TILE_H;
    const rtOptions = {
      depthBuffer: false,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      generateMipmaps: false
    };
    this._rtRaw = new THREE.WebGLRenderTarget(atlasW, atlasH, rtOptions);
    this._rtAtlas = new THREE.WebGLRenderTarget(atlasW, atlasH, rtOptions);
    this._rtRaw.scissorTest = true;
    this._rtAtlas.scissorTest = true;

    this._emptyTex = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
    this._emptyTex.needsUpdate = true;

    this._rawMat = new THREE.ShaderMaterial({
      vertexShader: SCREEN_QUAD_VERT,
      fragmentShader: MATTE_RAW_FRAG,
      uniforms: {
        uVideo: { value: this._emptyTex },
        uBg: { value: this._emptyTex },
        uMask: { value: this._emptyTex },
        uMaskTile: { value: new THREE.Vector4(0, 0, 1, 1) },
        uMaskTexel: { value: new THREE.Vector2(1, 1) },
        uTrimapPx: { value: 6.0 },
        uHasMask: { value: 0.0 },
        uGain: { value: 1.0 },
        uKey: { value: this.keyThreshold }
      },
      depthTest: false,
      depthWrite: false
    });
    this._refineMat = new THREE.ShaderMaterial({
      vertexShader: SCREEN_QUAD_VERT,
      fragmentShader: MATTE_REFINE_FRAG,
      uniforms: {
        uRaw: { value: this._rtRaw.texture },
        uTileRect: { value: new THREE.Vector4(0, 0, 1 / ATLAS_COLS, 1 / ATLAS_ROWS) },
        uTexel: { value: new THREE.Vector2(1 / atlasW, 1 / atlasH) },
        uDilatePx: { value: 5.0 }
      },
      depthTest: false,
      depthWrite: false
    });
    this._quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this._rawMat);
    this._quad.frustumCulled = false;
    this._quadScene = new THREE.Scene();
    this._quadScene.add(this._quad);
    this._quadCam = new THREE.Camera();

    // 与场景投影共享相机槽位 Uniform（同一对象引用），本模块独有的全局量另行维护
    this._cardUniforms = {
      uActiveCams: sharedUniforms.uActiveCams,
      uDepthMap: sharedUniforms.uDepthMap,
      uProjViewMatrix: sharedUniforms.uProjViewMatrix,
      uProjProjectionMatrix: sharedUniforms.uProjProjectionMatrix,
      uProjCamPos: sharedUniforms.uProjCamPos,
      uIntrinsics: sharedUniforms.uIntrinsics,
      uDistortion: sharedUniforms.uDistortion,
      uColorAdjust: sharedUniforms.uColorAdjust,
      uColorGain: sharedUniforms.uColorGain,
      uLiveTex: { value: Array.from({ length: MAX_CAMS }, () => this._emptyTex) },
      uLiveOn: { value: Array.from({ length: MAX_CAMS }, () => 0.0) },
      uMatteAtlas: { value: this._rtAtlas.texture },
      uAtlasGrid: { value: new THREE.Vector2(ATLAS_COLS, ATLAS_ROWS) },
      uAtlasPad: { value: new THREE.Vector2(0.5 / TILE_W, 0.5 / TILE_H) },
      uTargets: { value: Array.from({ length: MAX_TARGETS }, () => new THREE.Vector4()) },
      uTargetHeights: { value: new Array(MAX_TARGETS).fill(0) },
      uTargetCount: { value: 0 }
    };

    this._cardGeo = new THREE.PlaneGeometry(1, 1);
    this._cardGeo.translate(0, 0.5, 0); // 底边中心为原点（脚底落地）
    this._flatGeo = new THREE.PlaneGeometry(1, 1);
    this._flatGeo.rotateX(-Math.PI / 2); // 水平面片，中心为原点
    this._shadowGeo = new THREE.PlaneGeometry(2, 2);
    this._shadowGeo.rotateX(-Math.PI / 2);

    this._cardPool = [];
    this._shadowPool = [];
    this._slotPairs = new Array(MAX_CAMS).fill(null);
    this._camStates = new Map(); // 目标 id -> { camId, prevCamId, switchMs }
    this._yawStates = new Map(); // 目标 id -> { yaw, ms }：平滑后的面片朝向
    this._tmpView = new THREE.Vector3();
    this._tmpPoint = new THREE.Vector3();
  }

  setVisible(visible) {
    this.visible = visible;
    this.group.visible = visible;
  }

  clear() {
    this._cardPool.forEach((m) => (m.visible = false));
    this._shadowPool.forEach((m) => (m.visible = false));
    this._camStates.clear();
    this._yawStates.clear();
  }

  get matteUniforms() {
    const u = this._cardUniforms;
    return {
      uActiveCams: u.uActiveCams,
      uDepthMap: u.uDepthMap,
      uProjViewMatrix: u.uProjViewMatrix,
      uProjProjectionMatrix: u.uProjProjectionMatrix,
      uIntrinsics: u.uIntrinsics,
      uDistortion: u.uDistortion,
      uLiveOn: u.uLiveOn,
      uMatteAtlas: u.uMatteAtlas,
      uAtlasGrid: u.uAtlasGrid,
      uAtlasPad: u.uAtlasPad
    };
  }

  _acquireCard(index) {
    while (this._cardPool.length <= index) {
      const mat = new THREE.ShaderMaterial({
        vertexShader: CARD_VERT,
        fragmentShader: CARD_FRAG,
        uniforms: {
          ...this._cardUniforms,
          uCamWeight: { value: new Array(MAX_CAMS).fill(0) },
          uSelf: { value: 0 },
          uCardWidth: { value: 1.0 },
          uCardHeight: { value: 2.0 },
          uCardNormal: { value: new THREE.Vector2(0, 1) },
          uFlat: { value: 0.0 },
          uSupportRadius: { value: SUPPORT_RADIUS_VISUAL }
        },
        side: THREE.DoubleSide,
        transparent: true,
        depthTest: true,
        depthWrite: true
      });
      const mesh = new THREE.Mesh(this._cardGeo, mat);
      mesh.renderOrder = 2;
      mesh.visible = false;
      this.group.add(mesh);
      this._cardPool.push(mesh);
    }
    return this._cardPool[index];
  }

  _acquireShadow(index) {
    while (this._shadowPool.length <= index) {
      const mat = new THREE.ShaderMaterial({
        vertexShader: CONTACT_SHADOW_VERT,
        fragmentShader: CONTACT_SHADOW_FRAG,
        uniforms: {
          ...this._cardUniforms,
          uStrength: { value: 0.42 },
          uFoot: { value: new THREE.Vector3() },
          uBodyHeight: { value: 1.7 }
        },
        transparent: true,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: -2,
        polygonOffsetUnits: -2
      });
      const mesh = new THREE.Mesh(this._shadowGeo, mat);
      mesh.renderOrder = 1;
      mesh.visible = false;
      this.group.add(mesh);
      this._shadowPool.push(mesh);
    }
    return this._shadowPool[index];
  }

  /**
   * 主渲染循环调用（须在调度器刷新槽位之后）：刷新各槽位实时视频并渲染抠像图集，与目标数量无关
   * @param {THREE.WebGLRenderer} renderer
   * @param {Array} slotCams 调度器当前激活槽位的相机条目（下标即槽位号）
   * @param {object} streamMgr 提供 getLiveForegroundPair(camId)
   * @param {object} matteSource 提供 getMaskTile(camId) 与 getExposureGain(camId)
   */
  prepare(renderer, slotCams, streamMgr, matteSource) {
    if (!this.visible) return;

    let liveCount = 0;
    const liveTex = this._cardUniforms.uLiveTex.value;
    const liveOn = this._cardUniforms.uLiveOn.value;
    for (let i = 0; i < MAX_CAMS; i++) {
      const item = slotCams[i];
      const pair = item ? streamMgr.getLiveForegroundPair(item.id) : null;
      const ok = !!(pair && pair.bgTexture && pair.videoEl.readyState >= 2);
      this._slotPairs[i] = ok ? pair : null;
      liveTex[i] = ok ? pair.videoTexture : this._emptyTex;
      liveOn[i] = ok ? 1.0 : 0.0;
      if (ok) liveCount++;
    }
    if (liveCount > 0) this._renderMatteAtlas(renderer, slotCams, matteSource);
  }

  /**
   * 该路相机沿 camPos -> p 的视线是否先穿过别的可认领像素目标的躯干柱体（与着色器 claimedByOther 的遮挡判定一致）
   */
  _rayBlocked(camPos, p, occluders) {
    const dx = p.x - camPos.x;
    const dz = p.z - camPos.z;
    const len2 = Math.max(dx * dx + dz * dz, 1e-6);
    const len = Math.sqrt(len2);
    for (const o of occluders) {
      const s = ((o.x - camPos.x) * dx + (o.z - camPos.z) * dz) / len2;
      if (s <= 0.02) continue;
      const gap = Math.hypot(camPos.x + dx * s - o.x, camPos.z + dz * s - o.z);
      if (gap > o.radius + 0.2) continue;
      const yAt = camPos.y + (p.y - camPos.y) * s;
      if (yAt < o.baseY || yAt > o.baseY + o.height) continue;
      if ((1 - s) * len > o.radius && gap <= o.radius) return true;
    }
    return false;
  }

  /**
   * 目标中心指向相机与指向观察者的三维方向余弦
   */
  _viewCosine(tr, camPos, viewerPos) {
    const cy = tr.y + tr.height * 0.5;
    const cx = camPos.x - tr.x;
    const cyy = camPos.y - cy;
    const cz = camPos.z - tr.z;
    const vx = viewerPos.x - tr.x;
    const vy = viewerPos.y - cy;
    const vz = viewerPos.z - tr.z;
    return (cx * vx + cyy * vy + cz * vz) / ((Math.hypot(cx, cyy, cz) || 1e-6) * (Math.hypot(vx, vy, vz) || 1e-6));
  }

  /**
   * 某路相机作为该目标主相机时的候选分数：越贴近观察方向越好，被别的目标遮挡或正对性不足则扣分
   */
  _scoreCamera(tr, extent, camPos, viewerPos, occluders) {
    const cos3D = this._viewCosine(tr, camPos, viewerPos);

    let facing;
    if (extent.flat) {
      const cy = tr.y + tr.height * 0.5;
      facing = Math.abs(camPos.y - cy) / (Math.hypot(camPos.x - tr.x, camPos.y - cy, camPos.z - tr.z) || 1e-6);
    } else {
      const cx = camPos.x - tr.x;
      const cz = camPos.z - tr.z;
      const vx = viewerPos.x - tr.x;
      const vz = viewerPos.z - tr.z;
      const camH = Math.hypot(cx, cz) || 1e-6;
      const viewH = Math.hypot(vx, vz) || 1e-6;
      const cosHorizontal = THREE.MathUtils.clamp(Math.abs((cx * vx + cz * vz) / (camH * viewH)), 0, 1);
      const cam3D = Math.hypot(cx, camPos.y - (tr.y + tr.height * 0.5), cz) || 1e-6;
      facing = (camH / cam3D) * Math.cos(Math.acos(cosHorizontal) * 0.5);
    }

    let visible = 0;
    for (const f of VISIBILITY_HEIGHTS) {
      this._tmpPoint.set(tr.x, tr.y + tr.height * f, tr.z);
      if (!this._rayBlocked(camPos, this._tmpPoint, occluders)) visible++;
    }
    const vis = visible / VISIBILITY_HEIGHTS.length;

    return cos3D - 0.5 * (1 - vis) - (facing < FRONTAL_MIN ? 1 : 0);
  }

  /**
   * 面片朝向是无向直线（n 与 -n 等价）：按最短角差对平滑后的朝向做指数逼近，新目标首帧直接取目标朝向
   */
  _smoothNormal(id, normal, nowMs) {
    const target = Math.atan2(normal[0], normal[1]);
    const state = this._yawStates.get(id);
    let yaw = target;
    if (state) {
      const diff = THREE.MathUtils.euclideanModulo(target - state.yaw + Math.PI / 2, Math.PI) - Math.PI / 2;
      yaw = state.yaw + diff * (1 - Math.exp(-Math.max(nowMs - state.ms, 0) / YAW_SMOOTH_MS));
    }
    this._yawStates.set(id, { yaw, ms: nowMs });
    return [Math.sin(yaw), Math.cos(yaw)];
  }

  /**
   * 主渲染循环调用（须在 prepare 之后）：按目标放置面片与接触阴影
   * @param {Array} tracks 三维目标轨迹 [{id, x, y, z, width, height, hits, footprint?, exactTrack?, claimsPixels?}]
   * @param {Array} slotCams 调度器当前激活槽位的相机条目（下标即槽位号）
   * @param {THREE.Camera} mainCamera 当前观察相机
   * @param {number} nowMs 当前时间（毫秒）
   * @returns {Array} 本帧实际放置了面片的轨迹
   */
  update(tracks, slotCams, mainCamera, nowMs) {
    if (!this.visible) return [];

    const viewerPos = mainCamera.position;
    const entries = [];
    if (this._slotPairs.some(Boolean)) {
      for (const tr of tracks) {
        if (entries.length >= MAX_TARGETS) break;
        if ((tr.hits ?? 0) < 2) continue;

        const seenSlots = [];
        for (let i = 0; i < MAX_CAMS; i++) {
          if (!this._slotPairs[i]) continue;
          const item = slotCams[i];
          this._tmpView.set(tr.x, tr.y + tr.height * 0.5, tr.z).applyMatrix4(item.camera.matrixWorldInverse);
          if (projectCameraPointToPixel(this._tmpView, item.intrinsics, item.distortion)) seenSlots.push(i);
        }
        if (!seenSlots.length) continue;

        const extent = resolveCardExtent(tr);
        // 像素归属判定要求目标位置可靠：纯视觉检测的落脚点误差可达数十厘米，用它去认领别人的像素会把正常目标切出缺口
        entries.push({
          tr,
          seenSlots,
          extent,
          owner: {
            x: tr.x,
            z: tr.z,
            baseY: tr.y - 0.03,
            radius: extent.bodyRadius,
            height: extent.height,
            owns: !!(tr.exactTrack || tr.claimsPixels)
          }
        });
      }
    }

    const seenIds = new Set();
    entries.forEach(({ tr, seenSlots, extent, owner }, shown) => {
      const occluders = entries.filter((e, j) => j !== shown && e.owner.owns).map((e) => e.owner);

      const candidates = seenSlots.map((i) => ({
        camId: slotCams[i].id,
        slot: i,
        score: this._scoreCamera(tr, extent, slotCams[i].camera.position, viewerPos, occluders)
      }));
      const state = selectPrimaryCamera(candidates, this._camStates.get(tr.id), nowMs);
      this._camStates.set(tr.id, state);
      seenIds.add(tr.id);

      const primary = candidates.find((c) => c.camId === state.camId);
      const previous = state.prevCamId === null ? null : candidates.find((c) => c.camId === state.prevCamId);
      const blend = previous ? smoothstep01((nowMs - state.switchMs) / CAM_FADE_MS) : 1;

      const camDirs = [{ pos: slotCams[primary.slot].camera.position, weight: blend }];
      if (previous) camDirs.push({ pos: slotCams[previous.slot].camera.position, weight: 1 - blend });
      const pose = resolveCardPose(tr, camDirs, viewerPos);
      if (!pose.flat) pose.normal = this._smoothNormal(tr.id, pose.normal, nowMs);

      const card = this._acquireCard(shown);
      const weights = card.material.uniforms.uCamWeight.value;
      for (let i = 0; i < MAX_CAMS; i++) {
        if (!this._slotPairs[i]) weights[i] = 0;
        else if (i === primary.slot) weights[i] = blend;
        else if (previous && i === previous.slot) weights[i] = 1 - blend;
        else {
          const cos3D = this._viewCosine(tr, slotCams[i].camera.position, viewerPos);
          weights[i] = FILL_WEIGHT * Math.exp(8 * (cos3D - 1));
        }
      }

      // 面片略微探入地面以下，容忍落脚点高度误差（地面以下部分由场景深度自然遮挡）
      const baseY = tr.y - 0.03;
      card.visible = true;
      if (pose.flat) {
        card.geometry = this._flatGeo;
        card.position.set(tr.x, tr.y + pose.height, tr.z);
        card.rotation.y = 0;
        card.scale.set(pose.halfWidth * 2, 1, pose.halfWidth * 2);
      } else {
        card.geometry = this._cardGeo;
        card.position.set(tr.x, baseY, tr.z);
        card.rotation.y = Math.atan2(pose.normal[0], pose.normal[1]);
        card.scale.set(pose.halfWidth * 2, pose.height, 1);
      }
      card.material.uniforms.uFlat.value = pose.flat ? 1.0 : 0.0;
      card.material.uniforms.uSupportRadius.value = tr.exactTrack ? SUPPORT_RADIUS_EXACT : SUPPORT_RADIUS_VISUAL;
      card.material.uniforms.uSelf.value = shown;
      card.material.uniforms.uCardWidth.value = pose.halfWidth * 2;
      card.material.uniforms.uCardHeight.value = pose.height;
      card.material.uniforms.uCardNormal.value.set(pose.normal[0], pose.normal[1]);
      this._cardUniforms.uTargets.value[shown].set(tr.x, baseY, tr.z, pose.bodyRadius);
      this._cardUniforms.uTargetHeights.value[shown] = owner.owns ? pose.height : 0;

      const shadow = this._acquireShadow(shown);
      const shadowR = THREE.MathUtils.clamp(pose.halfWidth * 0.8, 0.34, 1.3);
      shadow.visible = true;
      shadow.position.set(tr.x, tr.y + 0.015, tr.z);
      shadow.scale.set(shadowR, 1, shadowR);
      shadow.material.uniforms.uFoot.value.set(tr.x, tr.y, tr.z);
      shadow.material.uniforms.uBodyHeight.value = tr.height;
    });

    for (const id of this._camStates.keys()) if (!seenIds.has(id)) this._camStates.delete(id);
    for (const id of this._yawStates.keys()) if (!seenIds.has(id)) this._yawStates.delete(id);

    const shown = entries.length;
    this._cardUniforms.uTargetCount.value = shown;
    for (let i = shown; i < this._cardPool.length; i++) this._cardPool[i].visible = false;
    for (let i = shown; i < this._shadowPool.length; i++) this._shadowPool[i].visible = false;
    return entries.map((e) => e.tr);
  }

  /**
   * 两阶段渲染抠像图集：A 逐像素置信度 -> B 形态学精修；每路相机占图集中一块瓦片
   */
  _renderMatteAtlas(renderer, slotCams, matteSource) {
    const prevTarget = renderer.getRenderTarget();
    const prevAutoClear = renderer.autoClear;
    renderer.autoClear = true;

    const raw = this._rawMat.uniforms;
    raw.uKey.value = this.keyThreshold;
    this._quad.material = this._rawMat;
    for (let i = 0; i < MAX_CAMS; i++) {
      const pair = this._slotPairs[i];
      if (!pair) continue;
      const camId = slotCams[i].id;
      const mask = matteSource.getMaskTile(camId);
      raw.uVideo.value = pair.videoTexture;
      raw.uBg.value = pair.bgTexture;
      raw.uGain.value = matteSource.getExposureGain(camId);
      raw.uMask.value = mask ? mask.texture : this._emptyTex;
      raw.uHasMask.value = mask ? 1.0 : 0.0;
      if (mask) {
        raw.uMaskTile.value.fromArray(mask.tile);
        const maskEl = mask.texture.image;
        raw.uMaskTexel.value.set(1 / (maskEl.videoWidth || 1), 1 / (maskEl.videoHeight || 1));
      }
      this._renderTile(renderer, this._rtRaw, i);
    }

    const refine = this._refineMat.uniforms;
    this._quad.material = this._refineMat;
    for (let i = 0; i < MAX_CAMS; i++) {
      if (!this._slotPairs[i]) continue;
      refine.uTileRect.value.set(
        (i % ATLAS_COLS) / ATLAS_COLS,
        Math.floor(i / ATLAS_COLS) / ATLAS_ROWS,
        1 / ATLAS_COLS,
        1 / ATLAS_ROWS
      );
      this._renderTile(renderer, this._rtAtlas, i);
    }

    renderer.setRenderTarget(prevTarget);
    renderer.autoClear = prevAutoClear;
  }

  _renderTile(renderer, target, slotIdx) {
    const x = (slotIdx % ATLAS_COLS) * TILE_W;
    const y = Math.floor(slotIdx / ATLAS_COLS) * TILE_H;
    target.viewport.set(x, y, TILE_W, TILE_H);
    target.scissor.set(x, y, TILE_W, TILE_H);
    renderer.setRenderTarget(target);
    renderer.render(this._quadScene, this._quadCam);
  }

  dispose() {
    this._rtRaw.dispose();
    this._rtAtlas.dispose();
    this._rawMat.dispose();
    this._refineMat.dispose();
    this._quad.geometry.dispose();
    this._cardGeo.dispose();
    this._flatGeo.dispose();
    this._shadowGeo.dispose();
    this._emptyTex.dispose();
    this._cardPool.forEach((m) => m.material.dispose());
    this._shadowPool.forEach((m) => m.material.dispose());
    this._cardPool = [];
    this._shadowPool = [];
  }
}
