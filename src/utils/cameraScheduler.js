import * as THREE from 'three';
import { MAX_CAMS } from './multiProjector.js';
import { disposeObject3D } from './disposeUtils.js';

const NDC_CORNERS = [
  [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
  [-1, -1,  1], [1, -1,  1], [-1, 1,  1], [1, 1,  1]
];

/**
 * 大园区多路监控动态调度器：
 * 紧致视锥体包围球剔除 + 视线焦点（Focus-Aware）感知评分 + 防抖滞回 -> Top 4 物理槽位分配
 */
export class CameraScheduler {
  constructor(projectorMgr, mainCamera, helperGroup, getFocusTarget = null) {
    this.projectorMgr = projectorMgr;
    this.mainCamera = mainCamera;
    this.helperGroup = helperGroup;
    this.getFocusTarget = getFocusTarget;

    this.logicalCameras = [];
    this.slotMap = [];
    this.activeIds = new Set();

    this._frustum = new THREE.Frustum();
    this._projScreenMatrix = new THREE.Matrix4();
    this._camFrustum = new THREE.Frustum();
    this._camProjScreenMatrix = new THREE.Matrix4();
    this._invProj = new THREE.Matrix4();
    this._tmpCorner = new THREE.Vector3();
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

    const bounds = this._computeFrustumBounds(entry.camera);
    const item = {
      ...entry,
      colorAdjust: entry.colorAdjust || { brightness: 0.0, contrast: 1.0, saturation: 1.0 },
      helper,
      boundingSphere: bounds.sphere,
      effectiveCenter: bounds.effectiveCenter
    };
    this.logicalCameras.push(item);
    return item;
  }

  /**
   * 当相机位置、朝向或 FOV 被微调后，刷新其紧致空间包围球与辅助线
   */
  refreshCameraBounds(item) {
    item.camera.updateMatrixWorld(true);
    item.camera.updateProjectionMatrix();
    const bounds = this._computeFrustumBounds(item.camera);
    item.boundingSphere = bounds.sphere;
    item.effectiveCenter = bounds.effectiveCenter;
    if (item.helper) item.helper.update();
  }

  /**
   * 通过 NDC 8 角点反算精确紧致视锥包围球，并计算有效监控覆盖中心
   */
  _computeFrustumBounds(cam) {
    cam.updateMatrixWorld(true);
    cam.updateProjectionMatrix();
    this._invProj.copy(cam.projectionMatrix).invert();

    const corners = [];
    const center = new THREE.Vector3();
    for (const [nx, ny, nz] of NDC_CORNERS) {
      const pt = new THREE.Vector3(nx, ny, nz).applyMatrix4(this._invProj).applyMatrix4(cam.matrixWorld);
      corners.push(pt);
      center.add(pt);
    }
    center.multiplyScalar(1 / corners.length);

    let maxRadius = 0;
    for (const pt of corners) {
      const d = center.distanceTo(pt);
      if (d > maxRadius) maxRadius = d;
    }

    const effectiveDist = Math.min(cam.far || 45, 28) * 0.45;
    const effectiveCenter = new THREE.Vector3(0, 0, -effectiveDist).applyMatrix4(cam.matrixWorld);

    return {
      sphere: new THREE.Sphere(center, maxRadius),
      effectiveCenter
    };
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

    this.mainCamera.updateMatrixWorld(true);
    this._projScreenMatrix.multiplyMatrices(
      this.mainCamera.projectionMatrix,
      this.mainCamera.matrixWorldInverse
    );
    this._frustum.setFromProjectionMatrix(this._projScreenMatrix);

    const focusTarget = this.getFocusTarget ? this.getFocusTarget() : null;
    const candidates = [];

    for (const item of this.logicalCameras) {
      if (!this._frustum.intersectsSphere(item.boundingSphere)) continue;

      const refPt = item.effectiveCenter || item.boundingSphere.center;
      const eyeDist = this.mainCamera.position.distanceTo(refPt);
      const focusDist = focusTarget ? focusTarget.distanceTo(refPt) : eyeDist;

      let score = focusDist * 0.65 + eyeDist * 0.35;

      // 若用户当前视线焦点（controls.target）直接位于该监控视锥内，给予强优先加权
      if (focusTarget) {
        this._camProjScreenMatrix.multiplyMatrices(
          item.camera.projectionMatrix,
          item.camera.matrixWorldInverse
        );
        this._camFrustum.setFromProjectionMatrix(this._camProjScreenMatrix);
        if (this._camFrustum.containsPoint(focusTarget)) {
          score *= 0.55;
        }
      }

      // 防抖滞回：已激活槽位保持一定黏性
      if (this.activeIds.has(item.id)) {
        score *= 0.85;
      }

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
