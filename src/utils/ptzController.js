import * as THREE from 'three';

/**
 * 球机（PTZ）全通道控制器：
 * 支持 Pan / Tilt / Zoom 映射、视频流延迟对齐队列、阻尼平滑插值，
 * 以及变焦时自动联动更新投影相机 FOV、OpenCV 内参 fx/fy 与广角畸变衰减。
 */
export class PTZController {
  constructor(camera, intrinsics, baseDistortion, options = {}) {
    this.camera = camera;
    this.camera.rotation.order = 'YXZ';
    this.intrinsics = intrinsics;
    this.baseDistortion = { ...baseDistortion };
    this.depthFovScale = intrinsics.depthFovScale || 1.1;

    this.syncBaseFromCamera();

    this.panRangeRad = THREE.MathUtils.degToRad(options.panRangeDeg ?? 120);
    this.tiltMinRad = THREE.MathUtils.degToRad(options.tiltMinDeg ?? -60);
    this.tiltMaxRad = THREE.MathUtils.degToRad(options.tiltMaxDeg ?? 10);
    this.minFocalMm = options.minFocalMm ?? 4.8;
    this.maxFocalMm = options.maxFocalMm ?? 24.0;
    this.sensorHeightMm = options.sensorHeightMm ?? 3.6;
    this.videoLatencyMs = options.videoLatencyMs ?? 300;

    this.buffer = [];
    this.target = { yaw: this.baseYaw, pitch: this.basePitch, zoom: 0 };
    this.current = { yaw: this.baseYaw, pitch: this.basePitch, zoom: 0 };
  }

  /**
   * 若用户在 GUI 手动微调了球机基准姿态，重新同步基准角
   */
  syncBaseFromCamera() {
    const baseEuler = new THREE.Euler().setFromQuaternion(this.camera.quaternion, 'YXZ');
    this.basePitch = baseEuler.x;
    this.baseYaw = baseEuler.y;
    this.baseRoll = baseEuler.z;
  }

  /**
   * 推入 ONVIF 云台数据（pan/tilt ∈ [-1, 1], zoom ∈ [0, 1]）
   */
  pushOnvifData({ pan = 0, tilt = 0, zoom = 0 }) {
    const deltaYaw = -pan * (this.panRangeRad * 0.5);
    const deltaPitch = tilt * THREE.MathUtils.degToRad(15);

    this.buffer.push({
      applyTime: performance.now() + this.videoLatencyMs,
      yaw: this.baseYaw + deltaYaw,
      pitch: THREE.MathUtils.clamp(this.basePitch + deltaPitch, this.tiltMinRad, this.tiltMaxRad),
      zoom: THREE.MathUtils.clamp(zoom, 0, 1)
    });
  }

  /**
   * 每帧更新云台姿态与光学变焦
   * @returns {Object|null} 若发生移动/变焦，返回最新内参与动态衰减后的畸变系数
   */
  update(dt) {
    const now = performance.now();
    while (this.buffer.length > 0 && this.buffer[0].applyTime <= now) {
      const next = this.buffer.shift();
      this.target.yaw = next.yaw;
      this.target.pitch = next.pitch;
      this.target.zoom = next.zoom;
    }

    const smooth = 1 - Math.exp(-10 * dt);
    const prevYaw = this.current.yaw;
    const prevPitch = this.current.pitch;
    const prevZoom = this.current.zoom;

    this.current.yaw = THREE.MathUtils.lerp(this.current.yaw, this.target.yaw, smooth);
    this.current.pitch = THREE.MathUtils.lerp(this.current.pitch, this.target.pitch, smooth);
    this.current.zoom = THREE.MathUtils.lerp(this.current.zoom, this.target.zoom, smooth);

    const delta =
      Math.abs(this.current.yaw - prevYaw) +
      Math.abs(this.current.pitch - prevPitch) +
      Math.abs(this.current.zoom - prevZoom);

    if (delta < 0.0001) return null;

    // 1. 更新云台旋转
    this.camera.rotation.set(this.current.pitch, this.current.yaw, this.baseRoll, 'YXZ');

    // 2. 光学变焦 -> 动态计算 FOV 与像素焦距 fx/fy
    const focalMm = THREE.MathUtils.lerp(this.minFocalMm, this.maxFocalMm, this.current.zoom);
    const halfFovRad = Math.atan(this.sensorHeightMm / (2 * focalMm));
    const fovY = THREE.MathUtils.radToDeg(2 * halfFovRad);

    this.camera.fov = fovY * this.depthFovScale;
    this.camera.updateProjectionMatrix();
    this.camera.updateMatrixWorld(true);

    const fy = (this.intrinsics.height * 0.5) / Math.tan(halfFovRad);
    const dynamicIntrinsics = { ...this.intrinsics, fx: fy, fy, fovY };

    // 3. 长焦端广角畸变自然衰减（zoom=0 保留 100% 畸变，zoom=1 衰减至 15%）
    const distScale = 1.0 - this.current.zoom * 0.85;
    const dynamicDistortion = {
      k1: (this.baseDistortion.k1 || 0) * distScale,
      k2: (this.baseDistortion.k2 || 0) * distScale,
      p1: (this.baseDistortion.p1 || 0) * distScale,
      p2: (this.baseDistortion.p2 || 0) * distScale
    };

    return { intrinsics: dynamicIntrinsics, distortion: dynamicDistortion };
  }
}
