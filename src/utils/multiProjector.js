import * as THREE from 'three';

export const MAX_CAMS = 6;
const DEPTH_MAP_SIZE = 1024;

/**
 * 根据相机内参及主点偏移动态计算深度渲染视场角（保证偏心主点与广角边缘深度覆盖完整）
 */
export function syncDepthCameraFromIntrinsics(cam, intr) {
  const aspect = intr.width / intr.height;
  const cxOffset = Math.abs((intr.cx ?? intr.width * 0.5) / intr.width - 0.5) * 2.0;
  const cyOffset = Math.abs((intr.cy ?? intr.height * 0.5) / intr.height - 0.5) * 2.0;
  const offsetScale = 1.0 + Math.max(cxOffset, cyOffset);
  const baseScale = intr.depthFovScale || 1.0;
  const effectiveScale = Math.max(baseScale, offsetScale);

  cam.fov = Math.min((intr.fovY || 45) * effectiveScale, 150);
  cam.aspect = aspect;
  if (intr.near) cam.near = intr.near;
  if (intr.far) cam.far = intr.far;
  cam.updateProjectionMatrix();
}

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
      rt.depthTexture.type = THREE.FloatType;
      this.depthTargets.push(rt);
    }

    this.sharedUniforms = {
      uActiveCams: { value: 0 },
      uVideoTex: { value: Array.from({ length: MAX_CAMS }, () => emptyTex) },
      uIsVideoTex: { value: Array.from({ length: MAX_CAMS }, () => 0.0) },
      uDepthMap: { value: this.depthTargets.map((rt) => rt.depthTexture) },
      uProjViewMatrix: { value: Array.from({ length: MAX_CAMS }, () => new THREE.Matrix4()) },
      uProjProjectionMatrix: { value: Array.from({ length: MAX_CAMS }, () => new THREE.Matrix4()) },
      uProjCamPos: { value: Array.from({ length: MAX_CAMS }, () => new THREE.Vector3()) },
      uIntrinsics: { value: Array.from({ length: MAX_CAMS }, () => new THREE.Vector4(1, 1, 0.5, 0.5)) },
      uDistortion: { value: Array.from({ length: MAX_CAMS }, () => new THREE.Vector4(0, 0, 0, 0)) },
      uColorAdjust: { value: Array.from({ length: MAX_CAMS }, () => new THREE.Vector4(0, 1, 1, 0)) },
      uColorGain: { value: Array.from({ length: MAX_CAMS }, () => new THREE.Vector3(1, 1, 1)) },
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
   * 写入指定槽位的 OpenCV 内参、畸变系数、色彩调节参数与跨相机自动色彩增益
   */
  setCalibration(slotIdx, intr, dist = {}, colorAdj = {}, colorGain = [1, 1, 1]) {
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
    this.sharedUniforms.uColorGain.value[slotIdx].fromArray(colorGain);
  }

  /**
   * Pass 1：仅重绘指定单个槽位的深度图
   */
  renderSingleDepthMap(renderer, slotIdx, hiddenObjects = []) {
    const cam = this.cameras[slotIdx];
    if (!cam) return;

    const prevVis = hiddenObjects.map((o) => o.visible);
    hiddenObjects.forEach((o) => (o.visible = false));

    // 自动临时隐藏场景中的所有线条、点云、精灵及显式排除深度渲染的对象（防止污染深度缓冲）
    const autoHidden = [];
    this.scene.traverse((obj) => {
      if ((obj.isLine || obj.isPoints || obj.isSprite || obj.userData?.excludeFromDepth) && obj.visible) {
        autoHidden.push(obj);
        obj.visible = false;
      }
    });

    this.scene.overrideMaterial = this.depthOverrideMat;
    cam.updateMatrixWorld();
    this.sharedUniforms.uProjViewMatrix.value[slotIdx].copy(cam.matrixWorldInverse);
    this.sharedUniforms.uProjProjectionMatrix.value[slotIdx].copy(cam.projectionMatrix);
    cam.getWorldPosition(this.sharedUniforms.uProjCamPos.value[slotIdx]);

    renderer.setRenderTarget(this.depthTargets[slotIdx]);
    renderer.render(this.scene, cam);

    this.scene.overrideMaterial = null;
    renderer.setRenderTarget(null);
    autoHidden.forEach((o) => (o.visible = true));
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
