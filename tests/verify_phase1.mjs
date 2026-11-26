import assert from 'node:assert/strict';
import * as THREE from 'three';
import { BBoxProjector, projectCameraPointToPixel } from '../src/utils/bboxProjector.js';
import { PTZController } from '../src/utils/ptzController.js';
import { Geofence3D, TargetVisualManager } from '../src/utils/geofenceAndVisuals.js';

// Polyfill browser globals needed by CanvasTexture in Node test
if (typeof globalThis.document === 'undefined') {
  const mockCtx = {
    clearRect() {},
    beginPath() {},
    roundRect() {},
    fill() {},
    stroke() {},
    fillText() {},
    set fillStyle(_) {},
    set strokeStyle(_) {},
    set lineWidth(_) {},
    set font(_) {},
    set textAlign(_) {},
    set textBaseline(_) {}
  };
  globalThis.document = {
    createElement() {
      return { width: 256, height: 64, getContext: () => mockCtx };
    }
  };
}

// 1. 验证正逆畸变往返误差 < 1e-3 px
{
  const intr = { width: 1920, height: 1080, fx: 1300, fy: 1300, cx: 980, cy: 525, fovY: 45, near: 0.5, far: 50 };
  const dist = { k1: -0.35, k2: 0.08, p1: 0.01, p2: -0.01 };
  const cam = new THREE.PerspectiveCamera(45, 16 / 9, 0.5, 50);
  cam.position.set(-10, 8.5, 11);
  cam.lookAt(-1, 2.5, -6);
  cam.updateMatrixWorld(true);

  const groundPt = new THREE.Vector3(1.2, 0, -1.5);
  const viewPos = groundPt.clone().applyMatrix4(cam.matrixWorldInverse);
  const px = projectCameraPointToPixel(viewPos, intr, dist);
  assert.ok(px, '正向畸变应命中图像范围');

  const proj = new BBoxProjector(cam, intr, dist);
  const undist = proj._undistortPixel(px.u, px.v);
  const expectedXn = viewPos.x / -viewPos.z;
  const expectedYn = viewPos.y / viewPos.z;
  const reErrPx = Math.hypot((undist.xn - expectedXn) * intr.fx, (undist.yn - expectedYn) * intr.fy);
  assert.ok(reErrPx < 1e-3, `正逆畸变往返误差过大: ${reErrPx} px`);

  // 2. 修改 k1 后 setCalibration 反投影误差 < 0.01m
  const newDist = { k1: -0.42, k2: 0.12, p1: 0, p2: 0 };
  const pxNew = projectCameraPointToPixel(viewPos, intr, newDist);
  proj.setCalibration(intr, newDist);
  const hit = proj.projectToPlane([pxNew.u - 20, pxNew.v - 80, pxNew.u + 20, pxNew.v], 0);
  assert.ok(hit, 'setCalibration 后应求交成功');
  const distErrM = hit.distanceTo(groundPt);
  assert.ok(distErrM < 0.01, `反投影空间误差过大: ${distErrM} m`);
  console.log(`[PASS] 正逆畸变往返误差 ${reErrPx.toExponential(2)} px, 地面反投影误差 ${distErrM.toExponential(2)} m`);
}

// 3. PTZController 巡航变焦后 getBaseMatrixWorld() 保持安装基准不变
{
  const cam = new THREE.PerspectiveCamera(38, 16 / 9, 0.5, 55);
  cam.position.set(0, 12, 15);
  cam.lookAt(0, 2, -4);
  cam.updateMatrixWorld(true);
  const expectedBase = cam.matrixWorld.clone();

  const intr = { width: 1920, height: 1080, fx: 1500, fy: 1500, cx: 960, cy: 540, fovY: 38, depthFovScale: 1.15 };
  const dist = { k1: -0.15, k2: 0.03, p1: 0, p2: 0 };
  const ptz = new PTZController(cam, intr, dist, { videoLatencyMs: 0 });
  ptz.pushOnvifData({ pan: 0.8, tilt: -0.5, zoom: 0.9 });
  ptz.update(0.2);

  const exportedBase = ptz.getBaseMatrixWorld();
  for (let i = 0; i < 16; i++) {
    assert.ok(Math.abs(exportedBase.elements[i] - expectedBase.elements[i]) < 1e-6, 'PTZ 基准矩阵不应被巡航姿态污染');
  }
  console.log('[PASS] PTZController 安装基准矩阵导出隔离验证通过');
}

// 4. Geofence3D 视觉隐藏 (visible = false) 且 enabled = true 时仍正常告警
{
  const fence = new Geofence3D([[-2, -2], [2, -2], [2, 2], [-2, 2]], 2.5);
  fence.group.visible = false;
  fence.enabled = true;
  const parent = new THREE.Group();
  const mgr = new TargetVisualManager(parent, 10);
  const alarms = mgr.updateTargets([{ globalId: 7, cls: 'person', x: 0.5, y: 0, z: 0.5 }], fence);
  assert.deepEqual(alarms, [7]);
  fence.enabled = false;
  const alarmsOff = mgr.updateTargets([{ globalId: 7, cls: 'person', x: 0.5, y: 0, z: 0.5 }], fence);
  assert.deepEqual(alarmsOff, []);
  console.log('[PASS] Geofence3D 业务开关与显隐解耦验证通过');
}
