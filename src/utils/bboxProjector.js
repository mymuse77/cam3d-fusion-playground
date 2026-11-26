import * as THREE from 'three';

/**
 * 与 Shader 一致的 OpenCV Brown-Conrady 正向畸变投影：相机坐标系点 -> 畸变像素坐标 (u, v)
 */
export function projectCameraPointToPixel(viewPos, intr, dist = {}) {
  if (viewPos.z >= -0.05) return null;
  const x = viewPos.x / (-viewPos.z);
  const y = viewPos.y / viewPos.z;
  const r2 = x * x + y * y;
  if (r2 > 2.5) return null;

  const r4 = r2 * r2;
  const k1 = dist.k1 || 0;
  const k2 = dist.k2 || 0;
  const p1 = dist.p1 || 0;
  const p2 = dist.p2 || 0;

  const radial = 1 + k1 * r2 + k2 * r4;
  const xDist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
  const yDist = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;

  const u = xDist * intr.fx + intr.cx;
  const v = yDist * intr.fy + intr.cy;
  if (u < 0 || u > intr.width || v < 0 || v > intr.height) return null;
  return { u, v };
}

/**
 * 收集场景中可用于行人/车辆落地反投影的可行走网格
 * 若场景已显式标注 userData.isWalkable，则仅收集标记网格；否则收集全部非辅助 Mesh（外部 GLB 自动适配）
 */
export function collectWalkableMeshes(root) {
  if (!root) return [];
  const tagged = [];
  const allMeshes = [];
  root.traverse((obj) => {
    if (!obj.isMesh || obj.userData?.excludeFromDepth) return;
    allMeshes.push(obj);
    if (obj.userData?.isWalkable) tagged.push(obj);
  });
  return tagged.length > 0 ? tagged : allMeshes;
}

/**
 * YOLO 2D 检测框反投影器：底边中心 -> 迭代去畸变 -> 世界射线 -> 3D 地形网格求交（未命中时回退地面平面）
 */
export class BBoxProjector {
  constructor(camera, intrinsics, distortion = {}, walkableMeshes = []) {
    this.camera = camera;
    this.walkableMeshes = walkableMeshes;
    this.setCalibration(intrinsics, distortion);
    this._ray = new THREE.Ray();
    this._raycaster = new THREE.Raycaster();
    this._dir = new THREE.Vector3();
    this._groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    this._normalMat = new THREE.Matrix3();
    this._worldNormal = new THREE.Vector3();
  }

  /**
   * 同步更新内参与畸变系数（用于 GUI 微调与 PTZ 变焦联动）
   */
  setCalibration(intrinsics, distortion = {}) {
    this.intr = { ...intrinsics };
    this.dist = {
      k1: distortion.k1 || 0,
      k2: distortion.k2 || 0,
      p1: distortion.p1 || 0,
      p2: distortion.p2 || 0
    };
  }

  setWalkableMeshes(meshes) {
    this.walkableMeshes = meshes || [];
  }

  _undistortPixel(u, v) {
    const xDist = (u - this.intr.cx) / this.intr.fx;
    const yDist = (v - this.intr.cy) / this.intr.fy;
    const { k1, k2, p1, p2 } = this.dist;
    if (!k1 && !k2 && !p1 && !p2) return { xn: xDist, yn: yDist };

    let x = xDist, y = yDist;
    for (let i = 0; i < 8; i++) {
      const r2 = x * x + y * y;
      const radial = 1 + k1 * r2 + k2 * r2 * r2;
      const dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
      const dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
      x = (xDist - dx) / radial;
      y = (yDist - dy) / radial;
    }
    return { xn: x, yn: y };
  }

  _buildRayFromBBox(bbox) {
    const [x1, , x2, y2] = bbox;
    const { xn, yn } = this._undistortPixel((x1 + x2) * 0.5, y2);

    this.camera.updateMatrixWorld();
    this._dir.set(xn, -yn, -1).normalize().applyQuaternion(this.camera.quaternion).normalize();
    this._ray.origin.copy(this.camera.position);
    this._ray.direction.copy(this._dir);
  }

  /**
   * 纯水平平面反投影（兼容无地形网格回退）
   */
  projectToPlane(bbox, groundY = 0) {
    this._buildRayFromBBox(bbox);
    this._groundPlane.constant = -groundY;

    const hit = new THREE.Vector3();
    const ok = this._ray.intersectPlane(this._groundPlane, hit);
    if (!ok || hit.distanceTo(this.camera.position) > this.camera.far * 1.6) return null;
    return hit;
  }

  /**
   * 3D 地形与高台网格射线反投影（筛选朝上可站立表面法线 ny > 0.45，未命中时回退至水平地面）
   */
  projectToScene(bbox, fallbackY = 0) {
    this._buildRayFromBBox(bbox);

    if (this.walkableMeshes && this.walkableMeshes.length > 0) {
      this._raycaster.set(this._ray.origin, this._ray.direction);
      this._raycaster.near = this.camera.near || 0.1;
      this._raycaster.far = (this.camera.far || 60) * 1.6;

      const hits = this._raycaster.intersectObjects(this.walkableMeshes, false);
      for (const hit of hits) {
        if (!hit.face) continue;
        this._normalMat.getNormalMatrix(hit.object.matrixWorld);
        this._worldNormal.copy(hit.face.normal).applyMatrix3(this._normalMat).normalize();
        if (this._worldNormal.y > 0.45) {
          return hit.point.clone();
        }
      }
    }

    return this.projectToPlane(bbox, fallbackY);
  }
}
