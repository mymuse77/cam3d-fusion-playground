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
 * YOLO 2D 检测框反投影器：底边中心 -> 迭代去畸变 -> 世界射线 -> 地面求交
 */
export class BBoxProjector {
  constructor(camera, intrinsics, distortion = {}) {
    this.camera = camera;
    this.setCalibration(intrinsics, distortion);
    this._ray = new THREE.Ray();
    this._dir = new THREE.Vector3();
    this._groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
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

  projectToPlane(bbox, groundY = 0) {
    const [x1, , x2, y2] = bbox;
    const { xn, yn } = this._undistortPixel((x1 + x2) * 0.5, y2);

    this._dir.set(xn, -yn, -1).normalize().applyQuaternion(this.camera.quaternion).normalize();
    this._ray.origin.copy(this.camera.position);
    this._ray.direction.copy(this._dir);
    this._groundPlane.constant = -groundY;

    const hit = new THREE.Vector3();
    const ok = this._ray.intersectPlane(this._groundPlane, hit);
    if (!ok || hit.distanceTo(this.camera.position) > this.camera.far * 1.5) return null;
    return hit;
  }
}
