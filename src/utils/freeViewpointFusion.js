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
// 视点相关权重锐度（与着色器共用）：取值很高，近似"最贴近观察方向的相机独占"，
// 只在两路相机夹角正中的窄带内交叉淡化，避免两个不同侧面的画面长时间叠成重影
const VIEW_SHARPNESS = 80.0;

// 一致性表决的空间容差（米）：外部高精度轨迹只需容忍多路视频不同步与标定误差；
// 纯视觉轨迹还要容忍数十厘米的落脚点误差
const SUPPORT_RADIUS_EXACT = 0.12;
const SUPPORT_RADIUS_VISUAL = 0.3;
// 低于此高度的目标（AGV、托盘车等）改用盖在顶面高度的水平面片：俯拍相机拍到的主要是其顶面
const FLAT_TARGET_HEIGHT = 0.7;

/**
 * 纯算法：解算目标承载面片的尺寸与朝向
 * 直立目标的面片过中轴竖直放置，法向取"主纹理相机方向"与"观察方向"的角平分线：
 * 相机把目标画面投到面片上会被拉宽 1/cos(θ/2)，观察者斜看面片又被压窄 cos(θ/2)，两者恰好抵消，
 * 因此目标从任意方位看宽度都不失真；观察者与某路相机重合时则与该路原始画面逐像素一致。
 * @param {{x:number, y:number, z:number, width:number, height:number, footprint?:number}} track
 * @param {Array<{x:number, y:number, z:number}>} camPositions 能看到该目标的相机位置
 * @param {{x:number, y:number, z:number}} viewerPos 当前观察相机位置
 * @returns {{halfWidth:number, bodyRadius:number, height:number, flat:boolean, normal:[number, number]}|null}
 */
export function resolveCardPose(track, camPositions, viewerPos) {
  if (!camPositions.length) return null;

  // 优先采用轨迹给出的真实占地宽度；否则由检测框估计：
  // 直立目标的检测框宽度常被俯拍透视、影子或粘连目标撑大，按人体身高比例封顶；低矮目标（如 AGV）直接采用检测宽度
  const bodyWidth =
    track.footprint ?? (track.height >= 1.2 ? Math.min(track.width, track.height * 0.38) : track.width);
  const bodyRadius = THREE.MathUtils.clamp(bodyWidth * 0.5 + 0.05, 0.3, 1.25);
  // 面片比躯干宽出一圈以容纳伸展的四肢，高出头顶少许以容忍身高误差
  const halfWidth = bodyRadius + 0.2;
  if (track.height < FLAT_TARGET_HEIGHT) {
    return { halfWidth, bodyRadius, height: track.height, flat: true, normal: [0, 1] };
  }
  const height = THREE.MathUtils.clamp(track.height + 0.15, FLAT_TARGET_HEIGHT, 2.6);
  const cy = track.y + track.height * 0.5;

  const directions = (p) => {
    const dx = p.x - track.x;
    const dy = p.y - cy;
    const dz = p.z - track.z;
    const len = Math.hypot(dx, dy, dz) || 1e-6;
    const lenH = Math.hypot(dx, dz) || 1e-6;
    return { d3: [dx / len, dy / len, dz / len], dh: [dx / lenH, dz / lenH] };
  };

  const view = directions(viewerPos);
  const cams = camPositions.map((pos) => {
    const c = directions(pos);
    return { dh: c.dh, cos: c.d3[0] * view.d3[0] + c.d3[1] * view.d3[1] + c.d3[2] * view.d3[2] };
  });
  // 权重相对最贴近观察方向的相机归一，避免高锐度指数在相机都偏离观察方向时整体下溢
  const bestCos = Math.max(...cams.map((c) => c.cos));
  let sumX = 0;
  let sumZ = 0;
  for (const c of cams) {
    const w = Math.exp(VIEW_SHARPNESS * (c.cos - bestCos));
    // 面片正反面等价：把相机方向翻到观察者一侧再合成，避免对侧相机相互抵消
    const sign = c.dh[0] * view.dh[0] + c.dh[1] * view.dh[1] < 0 ? -1 : 1;
    sumX += w * sign * c.dh[0];
    sumZ += w * sign * c.dh[1];
  }
  const sumLen = Math.hypot(sumX, sumZ);
  const camDir = sumLen > 1e-6 ? [sumX / sumLen, sumZ / sumLen] : view.dh;

  const nx = camDir[0] + view.dh[0];
  const nz = camDir[1] + view.dh[1];
  const nLen = Math.hypot(nx, nz);
  const normal = nLen > 1e-6 ? [nx / nLen, nz / nLen] : view.dh;

  return { halfWidth, bodyRadius, height, flat: false, normal };
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
      uLiveTex: { value: Array.from({ length: MAX_CAMS }, () => this._emptyTex) },
      uLiveOn: { value: Array.from({ length: MAX_CAMS }, () => 0.0) },
      uMatteAtlas: { value: this._rtAtlas.texture },
      uAtlasGrid: { value: new THREE.Vector2(ATLAS_COLS, ATLAS_ROWS) },
      uAtlasPad: { value: new THREE.Vector2(0.5 / TILE_W, 0.5 / TILE_H) },
      uViewSharpness: { value: VIEW_SHARPNESS },
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
    this._tmpView = new THREE.Vector3();
  }

  setVisible(visible) {
    this.visible = visible;
    this.group.visible = visible;
  }

  clear() {
    this._cardPool.forEach((m) => (m.visible = false));
    this._shadowPool.forEach((m) => (m.visible = false));
  }

  _acquireCard(index) {
    while (this._cardPool.length <= index) {
      const mat = new THREE.ShaderMaterial({
        vertexShader: CARD_VERT,
        fragmentShader: CARD_FRAG,
        uniforms: {
          ...this._cardUniforms,
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
   * 主渲染循环调用（须在调度器刷新槽位之后）
   * @param {THREE.WebGLRenderer} renderer
   * @param {Array} tracks 三维目标轨迹 [{x, y, z, width, height, hits, footprint?, exactTrack?}]
   * @param {Array} slotCams 调度器当前激活槽位的相机条目（下标即槽位号）
   * @param {object} streamMgr 提供 getLiveForegroundPair(camId)
   * @param {object} matteSource 提供 getMaskTile(camId) 与 getExposureGain(camId)
   * @param {THREE.Camera} mainCamera 当前观察相机
   */
  update(renderer, tracks, slotCams, streamMgr, matteSource, mainCamera) {
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

    let shown = 0;
    if (liveCount > 0) {
      for (const tr of tracks) {
        if (shown >= MAX_TARGETS) break;
        if ((tr.hits ?? 0) < 2) continue;

        const seenBy = [];
        for (let i = 0; i < MAX_CAMS; i++) {
          if (!this._slotPairs[i]) continue;
          const item = slotCams[i];
          this._tmpView.set(tr.x, tr.y + tr.height * 0.5, tr.z).applyMatrix4(item.camera.matrixWorldInverse);
          if (projectCameraPointToPixel(this._tmpView, item.intrinsics, item.distortion)) {
            seenBy.push(item.camera.position);
          }
        }
        const pose = resolveCardPose(tr, seenBy, mainCamera.position);
        if (!pose) continue;

        // 面片略微探入地面以下，容忍落脚点高度误差（地面以下部分由场景深度自然遮挡）
        const baseY = tr.y - 0.03;
        const card = this._acquireCard(shown);
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
        // 像素归属判定要求目标位置可靠：纯视觉检测的落脚点误差可达数十厘米且可能重复建轨，
        // 用它去认领别人的像素会把正常目标切出缺口，故仅外部高精度轨迹参与
        this._cardUniforms.uTargetHeights.value[shown] = tr.exactTrack ? pose.height : 0;

        const shadow = this._acquireShadow(shown);
        const shadowR = THREE.MathUtils.clamp(pose.halfWidth * 0.8, 0.34, 1.3);
        shadow.visible = true;
        shadow.position.set(tr.x, tr.y + 0.015, tr.z);
        shadow.scale.set(shadowR, 1, shadowR);
        shadow.material.uniforms.uFoot.value.set(tr.x, tr.y, tr.z);
        shadow.material.uniforms.uBodyHeight.value = tr.height;
        shown++;
      }
    }

    this._cardUniforms.uTargetCount.value = shown;
    for (let i = shown; i < this._cardPool.length; i++) this._cardPool[i].visible = false;
    for (let i = shown; i < this._shadowPool.length; i++) this._shadowPool[i].visible = false;

    if (shown > 0) this._renderMatteAtlas(renderer, slotCams, matteSource);
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
