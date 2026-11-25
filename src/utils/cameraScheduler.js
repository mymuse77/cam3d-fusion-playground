import * as THREE from 'three';
import { MAX_CAMS } from './multiProjector.js';
import { disposeObject3D } from './disposeUtils.js';

/**
 * 大园区多路监控动态调度器：视锥体剔除 + 距离防抖评分 -> Top 4 物理槽位分配
 */
export class CameraScheduler {
  constructor(projectorMgr, mainCamera, helperGroup) {
    this.projectorMgr = projectorMgr;
    this.mainCamera = mainCamera;
    this.helperGroup = helperGroup;

    this.logicalCameras = [];
    this.slotMap = [];
    this.activeIds = new Set();

    this._frustum = new THREE.Frustum();
    this._projScreenMatrix = new THREE.Matrix4();
    this._lastUpdate = 0;
  }

  /**
   * 清空所有已注册的逻辑相机与视锥体辅助线（用于 JSON 热重载）
   */
  clearAll() {
    for (const item of this.logicalCameras) {
      if (item.helper) {
        this.helperGroup.remove(item.helper);
        disposeObject3D(item.helper);
      }
    }
    this.logicalCameras = [];
    this.slotMap = [];
    this.activeIds.clear();
    this.projectorMgr.sharedUniforms.uActiveCams.value = 0;
  }

  registerCamera(entry) {
    const helper = new THREE.CameraHelper(entry.camera);
    this.helperGroup.add(helper);

    const item = {
      ...entry,
      colorAdjust: entry.colorAdjust || { brightness: 0.0, contrast: 1.0, saturation: 1.0 },
      helper,
      boundingSphere: this._computeFrustumSphere(entry.camera)
    };
    this.logicalCameras.push(item);
    return item;
  }

  /**
   * 当相机位置、朝向或 FOV 被微调后，刷新其空间包围球与辅助线
   */
  refreshCameraBounds(item) {
    item.camera.updateMatrixWorld(true);
    item.camera.updateProjectionMatrix();
    item.boundingSphere = this._computeFrustumSphere(item.camera);
    if (item.helper) item.helper.update();
  }

  _computeFrustumSphere(cam) {
    const far = cam.far;
    const center = new THREE.Vector3(0, 0, -far * 0.5).applyMatrix4(cam.matrixWorld);
    const halfH = Math.tan(THREE.MathUtils.degToRad(cam.fov * 0.5)) * far;
    const halfW = halfH * cam.aspect;
    const farCorner = new THREE.Vector3(halfW, halfH, -far).applyMatrix4(cam.matrixWorld);
    return new THREE.Sphere(center, center.distanceTo(farCorner));
  }

  getSlotIndexById(camId) {
    return this.slotMap.findIndex((item) => item.id === camId);
  }

  getCameraById(camId) {
    return this.logicalCameras.find((item) => item.id === camId) || null;
  }

  update(nowMs, force = false) {
    if (!force && nowMs - this._lastUpdate < 150) return false;
    this._lastUpdate = nowMs;

    this._projScreenMatrix.multiplyMatrices(
      this.mainCamera.projectionMatrix,
      this.mainCamera.matrixWorldInverse
    );
    this._frustum.setFromProjectionMatrix(this._projScreenMatrix);

    const candidates = [];
    for (const item of this.logicalCameras) {
      if (!this._frustum.intersectsSphere(item.boundingSphere)) continue;
      const dist = this.mainCamera.position.distanceTo(item.boundingSphere.center);
      const score = this.activeIds.has(item.id) ? dist * 0.85 : dist;
      candidates.push({ item, score });
    }

    candidates.sort((a, b) => a.score - b.score);
    const selected = candidates.slice(0, MAX_CAMS).map((c) => c.item);
    const nextIds = new Set(selected.map((s) => s.id));

    if (!force && this._isSameSet(this.activeIds, nextIds)) return false;

    this.activeIds = nextIds;
    this.slotMap = selected;
    const uniforms = this.projectorMgr.sharedUniforms;
    uniforms.uActiveCams.value = selected.length;

    selected.forEach((item, slotIdx) => {
      this.projectorMgr.cameras[slotIdx] = item.camera;
      uniforms.uVideoTex.value[slotIdx] = item.texture;
      uniforms.uProjViewMatrix.value[slotIdx].copy(item.camera.matrixWorldInverse);
      uniforms.uProjProjectionMatrix.value[slotIdx].copy(item.camera.projectionMatrix);
      item.camera.getWorldPosition(uniforms.uProjCamPos.value[slotIdx]);
      this.projectorMgr.setCalibration(slotIdx, item.intrinsics, item.distortion, item.colorAdjust);
    });

    return true;
  }

  _isSameSet(a, b) {
    if (a.size !== b.size) return false;
    for (const v of a) if (!b.has(v)) return false;
    return true;
  }
}
