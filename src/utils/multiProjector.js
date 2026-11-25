import * as THREE from 'three';

export const MAX_CAMS = 4;
const DEPTH_MAP_SIZE = 1024;

/**
 * 物理渲染槽位管理器：负责维护 4 组 DepthRenderTarget 与共享 Shader Uniforms
 */
export class MultiProjectorManager {
  constructor(scene) {
    this.scene = scene;
    this.cameras = new Array(MAX_CAMS).fill(null);
    this.depthTargets = [];
    this.depthOverrideMat = new THREE.MeshBasicMaterial();

    const emptyTex = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
    emptyTex.needsUpdate = true;

    for (let i = 0; i < MAX_CAMS; i++) {
      const rt = new THREE.WebGLRenderTarget(DEPTH_MAP_SIZE, DEPTH_MAP_SIZE);
      rt.depthTexture = new THREE.DepthTexture();
      rt.depthTexture.type = THREE.UnsignedShortType;
      this.depthTargets.push(rt);
    }

    this.sharedUniforms = {
      uActiveCams: { value: 0 },
      uVideoTex: { value: Array.from({ length: MAX_CAMS }, () => emptyTex) },
      uDepthMap: { value: this.depthTargets.map((rt) => rt.depthTexture) },
      uProjViewMatrix: { value: Array.from({ length: MAX_CAMS }, () => new THREE.Matrix4()) },
      uProjProjectionMatrix: { value: Array.from({ length: MAX_CAMS }, () => new THREE.Matrix4()) },
      uProjCamPos: { value: Array.from({ length: MAX_CAMS }, () => new THREE.Vector3()) },
      uIntrinsics: { value: Array.from({ length: MAX_CAMS }, () => new THREE.Vector4(1, 1, 0.5, 0.5)) },
      uDistortion: { value: Array.from({ length: MAX_CAMS }, () => new THREE.Vector4(0, 0, 0, 0)) },
      uColorAdjust: { value: Array.from({ length: MAX_CAMS }, () => new THREE.Vector4(0, 1, 1, 0)) },
      uOpacity: { value: 0.9 },
      uFeather: { value: 0.05 },
      uDepthBias: { value: 0.0025 },
      uGrazingThreshold: { value: 0.12 },
      uDepthTexelSize: { value: 1.0 / DEPTH_MAP_SIZE },
      uEnableOcclusion: { value: true },
      uEnablePCF: { value: true }
    };
  }

  /**
   * 写入指定槽位的 OpenCV 内参、畸变系数与色彩调节参数
   */
  setCalibration(slotIdx, intr, dist = {}, colorAdj = {}) {
    if (slotIdx < 0 || slotIdx >= MAX_CAMS) return;
    this.sharedUniforms.uIntrinsics.value[slotIdx].set(
      intr.fx / intr.width,
      intr.fy / intr.height,
      intr.cx / intr.width,
      intr.cy / intr.height
    );
    this.sharedUniforms.uDistortion.value[slotIdx].set(
      dist.k1 || 0,
      dist.k2 || 0,
      dist.p1 || 0,
      dist.p2 || 0
    );
    this.sharedUniforms.uColorAdjust.value[slotIdx].set(
      colorAdj.brightness ?? 0.0,
      colorAdj.contrast ?? 1.0,
      colorAdj.saturation ?? 1.0,
      0.0
    );
  }

  /**
   * Pass 1：仅重绘指定单个槽位的深度图
   */
  renderSingleDepthMap(renderer, slotIdx, hiddenObjects = []) {
    const cam = this.cameras[slotIdx];
    if (!cam) return;

    const prevVis = hiddenObjects.map((o) => o.visible);
    hiddenObjects.forEach((o) => (o.visible = false));

    this.scene.overrideMaterial = this.depthOverrideMat;
    cam.updateMatrixWorld();
    this.sharedUniforms.uProjViewMatrix.value[slotIdx].copy(cam.matrixWorldInverse);
    this.sharedUniforms.uProjProjectionMatrix.value[slotIdx].copy(cam.projectionMatrix);
    cam.getWorldPosition(this.sharedUniforms.uProjCamPos.value[slotIdx]);

    renderer.setRenderTarget(this.depthTargets[slotIdx]);
    renderer.render(this.scene, cam);

    this.scene.overrideMaterial = null;
    renderer.setRenderTarget(null);
    hiddenObjects.forEach((o, i) => (o.visible = prevVis[i]));
  }

  /**
   * Pass 1：批量重绘所有激活槽位的深度图
   */
  renderAllDepthMaps(renderer, hiddenObjects = []) {
    const count = this.sharedUniforms.uActiveCams.value;
    for (let i = 0; i < count; i++) {
      this.renderSingleDepthMap(renderer, i, hiddenObjects);
    }
  }
}
