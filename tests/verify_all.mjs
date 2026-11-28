import assert from 'node:assert/strict';
import * as THREE from 'three';

import { createPracticeScene } from '../src/scene/createPracticeScene.js';
import { BBoxProjector, collectWalkableMeshes, projectCameraPointToPixel } from '../src/utils/bboxProjector.js';
import { MultiCamTracker3D } from '../src/utils/multiCamTracker3D.js';
import { CameraScheduler } from '../src/utils/cameraScheduler.js';
import { MultiProjectorManager } from '../src/utils/multiProjector.js';
import { PTZController } from '../src/utils/ptzController.js';
import { Geofence3D, TargetVisualManager } from '../src/utils/geofenceAndVisuals.js';
import { VideoStreamManager } from '../src/utils/videoStreamManager.js';
import { extractForegroundBlobsFromRgba } from '../src/utils/videoBillboard.js';
import { resolveCardPose, selectPrimaryCamera } from '../src/utils/freeViewpointFusion.js';
import { extractOccupancyPeaks } from '../src/utils/occupancyLocator.js';
import { solveColorGains } from '../src/utils/autoColorMatcher.js';
import { solveCameraPnP } from '../src/utils/webCalibrator.js';

// Polyfill 最小 DOM 环境供 Three.js CanvasTexture / VideoStreamManager 在 Node 下运行
if (typeof globalThis.document === 'undefined') {
  const mockCtx = {
    clearRect() {}, beginPath() {}, roundRect() {}, fill() {}, stroke() {}, fillText() {},
    set fillStyle(_) {}, set strokeStyle(_) {}, set lineWidth(_) {}, set font(_) {},
    set textAlign(_) {}, set textBaseline(_) {}
  };
  globalThis.document = {
    hidden: false,
    addEventListener() {},
    removeEventListener() {},
    createElement(tag) {
      if (tag === 'video') {
        return {
          paused: true,
          muted: false,
          loop: false,
          playsInline: false,
          autoplay: false,
          crossOrigin: '',
          srcObject: null,
          play() { this.paused = false; return Promise.resolve(); },
          pause() { this.paused = true; },
          removeAttribute() {},
          load() {},
          canPlayType() { return ''; }
        };
      }
      return { width: 256, height: 64, getContext: () => mockCtx };
    }
  };
}

// ==========================================
// 1. 第一阶段回归：正逆畸变往返 & 地面反投影
// ==========================================
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
  assert.ok(px);

  const proj = new BBoxProjector(cam, intr, dist);
  const undist = proj._undistortPixel(px.u, px.v);
  const reErrPx = Math.hypot(
    (undist.xn - viewPos.x / -viewPos.z) * intr.fx,
    (undist.yn - viewPos.y / viewPos.z) * intr.fy
  );
  assert.ok(reErrPx < 1e-3, `正逆畸变往返误差过大: ${reErrPx}`);

  const newDist = { k1: -0.42, k2: 0.12, p1: 0, p2: 0 };
  const pxNew = projectCameraPointToPixel(viewPos, intr, newDist);
  proj.setCalibration(intr, newDist);
  const hit = proj.projectToPlane([pxNew.u - 20, pxNew.v - 80, pxNew.u + 20, pxNew.v], 0);
  assert.ok(hit && hit.distanceTo(groundPt) < 0.01);
  console.log(`[PASS] 1. 正逆畸变往返误差 ${reErrPx.toExponential(2)} px`);
}

// ==========================================
// 2. 第一阶段回归：PTZ 基准隔离 & 电子围栏解耦
// ==========================================
{
  const cam = new THREE.PerspectiveCamera(38, 16 / 9, 0.5, 55);
  cam.position.set(0, 12, 15);
  cam.lookAt(0, 2, -4);
  cam.updateMatrixWorld(true);
  const expectedBase = cam.matrixWorld.clone();

  const ptz = new PTZController(
    cam,
    { width: 1920, height: 1080, fx: 1500, fy: 1500, cx: 960, cy: 540, fovY: 38, depthFovScale: 1.15 },
    { k1: -0.15, k2: 0.03, p1: 0, p2: 0 },
    { videoLatencyMs: 0 }
  );
  ptz.pushOnvifData({ pan: 0.8, tilt: -0.5, zoom: 0.9 });
  ptz.update(0.2);
  const exportedBase = ptz.getBaseMatrixWorld();
  for (let i = 0; i < 16; i++) {
    assert.ok(Math.abs(exportedBase.elements[i] - expectedBase.elements[i]) < 1e-6);
  }

  const fence = new Geofence3D([[-2, -2], [2, -2], [2, 2], [-2, 2]], 2.5);
  fence.group.visible = false;
  fence.enabled = true;
  const mgr = new TargetVisualManager(new THREE.Group(), 10);
  assert.deepEqual(mgr.updateTargets([{ globalId: 7, cls: 'person', x: 0.5, y: 0, z: 0.5 }], fence), [7]);
  fence.enabled = false;
  assert.deepEqual(mgr.updateTargets([{ globalId: 7, cls: 'person', x: 0.5, y: 0, z: 0.5 }], fence), []);
  console.log('[PASS] 2. PTZ 安装基准导出隔离 & 电子围栏显隐解耦');
}

// ==========================================
// 3. 第二阶段：3D 地形/高台/斜坡网格射线反投影
// ==========================================
{
  const sceneRoot = createPracticeScene();
  sceneRoot.updateMatrixWorld(true);
  const walkable = collectWalkableMeshes(sceneRoot);
  assert.ok(walkable.length >= 3, '应收集到地面、平台、斜坡三个可行走网格');

  const intr = { width: 1920, height: 1080, fx: 1300, fy: 1300, cx: 960, cy: 540, fovY: 45, near: 0.5, far: 55 };
  const dist = { k1: -0.2, k2: 0.04, p1: 0, p2: 0 };
  const cam = new THREE.PerspectiveCamera(45, 16 / 9, 0.5, 55);
  cam.position.set(10, 8.5, 11);
  cam.lookAt(5, 1.0, -2);
  cam.updateMatrixWorld(true);

  const proj = new BBoxProjector(cam, intr, dist, walkable);

  // 平台顶面中心 (7.0, 0.8, -2.5)
  const platPt = new THREE.Vector3(7.0, 0.8, -2.5);
  const pxPlat = projectCameraPointToPixel(platPt.clone().applyMatrix4(cam.matrixWorldInverse), intr, dist);
  const hitPlat = proj.projectToScene([pxPlat.u - 20, pxPlat.v - 80, pxPlat.u + 20, pxPlat.v], 0);
  assert.ok(hitPlat && Math.abs(hitPlat.y - 0.8) < 0.01 && hitPlat.distanceTo(platPt) < 0.02, `平台命中偏差: ${hitPlat?.y}`);

  // 斜坡中点 (7.0, 0.4, 2.0)
  const rampPt = new THREE.Vector3(7.0, 0.4, 2.0);
  const pxRamp = projectCameraPointToPixel(rampPt.clone().applyMatrix4(cam.matrixWorldInverse), intr, dist);
  const hitRamp = proj.projectToScene([pxRamp.u - 20, pxRamp.v - 80, pxRamp.u + 20, pxRamp.v], 0);
  assert.ok(hitRamp && Math.abs(hitRamp.y - 0.4) < 0.01 && hitRamp.distanceTo(rampPt) < 0.02, `斜坡命中偏差: ${hitRamp?.y}`);
  console.log(`[PASS] 3. 3D 地形/高台网格反投影 (平台 y=${hitPlat.y.toFixed(4)}m, 斜坡 y=${hitRamp.y.toFixed(4)}m)`);
}

// ==========================================
// 4. 第二阶段：卡尔曼滤波 + 匈牙利匹配双目标交叉 0 ID Switch
// ==========================================
{
  const tracker = new MultiCamTracker3D({ personThreshold: 1.5 });
  let seed = 42;
  const noise = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return (seed / 4294967296 - 0.5) * 0.25;
  };

  let idA = null;
  let idB = null;
  let switches = 0;

  for (let step = 0; step < 30; step++) {
    const t = step * 0.1; // 100ms 步长
    const nowMs = 1000 + step * 100;
    // 目标 A 从 (-4, -3) 走到 (+4, +3); 目标 B 从 (-4, +3) 走到 (+4, -3)，在 (0, 0) 交叉
    const ax = -4 + t * 2.6;
    const az = -3 + t * 2.0;
    const bx = -4 + t * 2.6;
    const bz = 3 - t * 2.0;

    const obs = [];
    // 第 15 帧模拟目标 B 短暂单帧漏检（依靠卡尔曼预测接续）
    obs.push({ camId: 'cam_01', localTrackId: 101, cls: 'person', x: ax + noise(), y: 0, z: az + noise(), distToCam: 12 });
    obs.push({ camId: 'cam_02', localTrackId: 201, cls: 'person', x: ax + noise(), y: 0, z: az + noise(), distToCam: 14 });

    if (step !== 15) {
      obs.push({ camId: 'cam_01', localTrackId: 102, cls: 'person', x: bx + noise(), y: 0.4, z: bz + noise(), distToCam: 12 });
      obs.push({ camId: 'cam_02', localTrackId: 202, cls: 'person', x: bx + noise(), y: 0.4, z: bz + noise(), distToCam: 14 });
    }

    const tracks = tracker.update(obs, nowMs);
    const tA = tracks.reduce((best, tr) => (!best || Math.hypot(tr.x - ax, tr.z - az) < Math.hypot(best.x - ax, best.z - az) ? tr : best), null);
    const tB = tracks.reduce((best, tr) => (!best || Math.hypot(tr.x - bx, tr.z - bz) < Math.hypot(best.x - bx, best.z - bz) ? tr : best), null);

    if (step === 0) {
      idA = tA.globalId;
      idB = tB.globalId;
      assert.notEqual(idA, idB);
    } else {
      if (tA.globalId !== idA) switches++;
      if (step !== 15 && tB.globalId !== idB) switches++;
    }
  }
  assert.equal(switches, 0, `双目标交叉发生 ID Switch: ${switches} 次`);
  console.log('[PASS] 4. 卡尔曼滤波 + 匈牙利匹配双目标交叉 0 次 ID Switch');
}

// ==========================================
// 5. 第二阶段：视线焦点感知调度 & 流媒体自动休眠
// ==========================================
{
  const scene = new THREE.Scene();
  const mgr = new MultiProjectorManager(scene);
  const mainCam = new THREE.PerspectiveCamera(50, 16 / 9, 0.5, 200);
  mainCam.position.set(0, 28, 45);
  const focusTarget = new THREE.Vector3(20, 0, -10);
  mainCam.lookAt(focusTarget);
  mainCam.updateMatrixWorld(true);

  const scheduler = new CameraScheduler(mgr, mainCam, new THREE.Group(), () => focusTarget);

  const mkCam = (id, pos, target) => {
    const c = new THREE.PerspectiveCamera(45, 16 / 9, 0.5, 40);
    c.position.set(...pos);
    c.lookAt(...target);
    c.updateMatrixWorld(true);
    return {
      id, name: id, type: 'fixed', camera: c,
      intrinsics: { width: 1920, height: 1080, fx: 1300, fy: 1300, cx: 960, cy: 540, fovY: 45 },
      distortion: { k1: 0, k2: 0, p1: 0, p2: 0 },
      texture: mgr.sharedUniforms.uVideoTex.value[0]
    };
  };

  // 6 个离主相机更近但拍向左侧 (-15, 0, -10) 的相机（超过 MAX_CAMS=6 容量以验证焦点抢占）
  for (let i = 1; i <= 6; i++) {
    scheduler.registerCamera(mkCam(`cam_near_miss_${i}`, [-10 - i, 8, 25], [-16, 0, -10]));
  }
  // 1 个覆盖视线焦点 (20, 0, -10) 的远端相机
  scheduler.registerCamera(mkCam('cam_focus_hit', [20, 8, 8], [20, 0, -10]));

  scheduler.update(1000, true);
  assert.ok(scheduler.activeIds.has('cam_focus_hit'), '视线焦点所在监控必须入选 Top 槽位');

  // 流媒体休眠/恢复测试
  const streamMgr = new VideoStreamManager();
  const fakeFile = {};
  globalThis.URL.createObjectURL = () => 'blob:mock-cam';
  globalThis.URL.revokeObjectURL = () => {};
  const entry = streamMgr.bindLocalFile('cam_focus_hit', fakeFile);
  assert.equal(entry.video.paused, false);

  streamMgr.syncActiveCameras(new Set(['cam_focus_hit']), 1000, 3000);
  streamMgr.syncActiveCameras(new Set(), 5000, 3000);
  assert.equal(entry.video.paused, true, '滑出槽位超过 3s 应自动暂停视频解码');

  streamMgr.syncActiveCameras(new Set(['cam_focus_hit']), 6000, 3000);
  assert.equal(entry.video.paused, false, '重新进入激活槽位应自动恢复播放');
  streamMgr.disposeAll();
  console.log('[PASS] 5. 视线焦点感知调度 & 流媒体槽位自动休眠/恢复');
}

// ==========================================
// 6. 第二阶段：Web 端 RANSAC + DLT + LM PnP 求解器（含广角畸变与 80px 误点剔除）
// ==========================================
{
  const intr = { width: 1920, height: 1080, fx: 1300, fy: 1300, cx: 960, cy: 540 };
  const dist = { k1: -0.28, k2: 0.09, p1: 0.0, p2: 0.0 };

  const trueCam = new THREE.PerspectiveCamera(45, 16 / 9, 0.5, 60);
  trueCam.position.set(-10, 8.5, 11);
  trueCam.lookAt(-1, 2.5, -6);
  trueCam.updateMatrixWorld(true);

  const pts3D = [
    [-9, 8, -4],
    [9, 8, -4],
    [-9, 0, -4],
    [9, 0, -4],
    [-3, 5.5, 1],
    [3.5, 2.4, 2],
    [7.0, 0.8, -2.5],
    [0.0, 0.0, 0.0]
  ];

  const pts2D = pts3D.map(([x, y, z]) => {
    const viewPos = new THREE.Vector3(x, y, z).applyMatrix4(trueCam.matrixWorldInverse);
    const px = projectCameraPointToPixel(viewPos, intr, dist);
    assert.ok(px, '合成测试控制点应在视野内');
    return [px.u, px.v];
  });

  // 故意在第 8 个点注入 80px 误点击外点
  pts2D[7] = [pts2D[7][0] + 80, pts2D[7][1] - 45];

  const result = solveCameraPnP({ pts3D, pts2D, intrinsics: intr, distortion: dist, useRansac: true });
  assert.equal(result.inlierCount, 7, `RANSAC 应识别 7 个内点并剔除 1 个外点，实际: ${result.inlierCount}`);
  assert.ok(!result.inliers.includes(7), '第 8 个 80px 偏移外点应被剔除');
  assert.ok(result.reprojectionErrorPx < 0.5, `重投影误差过高: ${result.reprojectionErrorPx} px`);

  const posErr = result.position.distanceTo(trueCam.position);
  assert.ok(posErr < 0.05, `还原相机位置误差过大: ${posErr} m`);
  console.log(
    `[PASS] 6. Web RANSAC+LM PnP 标定 (内点 ${result.inlierCount}/8, 误差 ${result.reprojectionErrorPx} px, 位姿误差 ${posErr.toExponential(2)} m)`
  );
}

// ==========================================
// 7. 前景融合模式：运动前景检测、上下部件合并与地面暗影抑制
// ==========================================
{
  const w = 160, h = 90;
  const bg = new Uint8ClampedArray(w * h * 4);
  const vid = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h * 4; i += 4) {
    bg[i] = 110; bg[i + 1] = 115; bg[i + 2] = 122; bg[i + 3] = 255;
    vid[i] = 110; vid[i + 1] = 115; vid[i + 2] = 122; vid[i + 3] = 255;
  }

  const paintRect = (arr, x1, y1, x2, y2, r, g, b) => {
    for (let y = y1; y <= y2; y++) {
      for (let x = x1; x <= x2; x++) {
        const p = (y * w + x) * 4;
        arr[p] = r; arr[p + 1] = g; arr[p + 2] = b; arr[p + 3] = 255;
      }
    }
  };

  // 绘制地面投射暗影（同色度、亮度减半）-> 应被 Cucchiara 阴影抑制完全过滤
  paintRect(vid, 20, 50, 45, 70, 55, 58, 61);
  // 绘制橙色身体 (x: 70..76, y: 46..62) + 间隔 4 格的黄色安全帽 (x: 71..75, y: 39..42) -> 应合并为 1 个检测框
  paintRect(vid, 70, 46, 76, 62, 245, 110, 25);
  paintRect(vid, 71, 39, 75, 42, 250, 225, 30);

  const blobs = extractForegroundBlobsFromRgba(vid, bg, w, h, 1280, 720);
  assert.equal(blobs.length, 1, `应过滤暗影并合并身体+安全帽为 1 个前景框，实际: ${blobs.length}`);
  const [u1, v1, u2, v2] = blobs[0];
  assert.ok(u1 > 500 && u2 < 650 && v1 < 330 && v2 > 490, `检测框范围异常: ${blobs[0]}`);
  console.log(`[PASS] 7. 前景融合检测（暗影抑制 + 部位合并）-> bbox=[${blobs[0].map((n) => Math.round(n)).join(', ')}]`);
}

// ==========================================
// 8. 自由视点融合：承载面片朝向（相机方向与观察方向的角平分线）与尺寸解算
// ==========================================
{
  const person = { x: 0, y: 0, z: 0, width: 0.7, height: 1.7 };
  const camA = { x: 10, y: 6, z: 0 };
  const camB = { x: 0, y: 6, z: 10 };
  const angleOf = (n) => THREE.MathUtils.radToDeg(Math.atan2(n[1], n[0]));

  assert.equal(resolveCardPose(person, [], camA), null, '没有任何相机看到目标时不应生成面片');

  // 观察者与相机 A 同向：面片正对相机 A，此时画面与该路原始视频逐像素一致
  const atCamA = resolveCardPose(person, [{ pos: camA, weight: 1 }, { pos: camB, weight: 0 }], { x: 8, y: 5, z: 0 });
  assert.ok(Math.abs(angleOf(atCamA.normal)) < 1, `面片应正对相机 A，实际法向角 ${angleOf(atCamA.normal)}°`);

  // 观察者偏离唯一相机 60°：法向应取两者角平分线 (30°)，保证目标宽度不失真
  const off60 = resolveCardPose(person, [{ pos: camA, weight: 1 }], { x: 5, y: 3, z: 5 * Math.sqrt(3) });
  assert.ok(Math.abs(angleOf(off60.normal) - 30) < 1, `法向应为 30° 角平分线，实际 ${angleOf(off60.normal)}°`);

  // 观察者绕到唯一相机的正对面：面片正反面等价，应正对观察者而非侧立成一条线
  const behind = resolveCardPose(person, [{ pos: camA, weight: 1 }], { x: -8, y: 3, z: 0 });
  assert.ok(Math.abs(Math.abs(angleOf(behind.normal)) - 180) < 1, `面片应正对背面观察者，实际 ${angleOf(behind.normal)}°`);

  // 主纹理相机权重全部给相机 B：法向落在 B 与观察者之间
  const nearB = resolveCardPose(person, [{ pos: camA, weight: 0 }, { pos: camB, weight: 1 }], { x: 2, y: 5, z: 8 });
  assert.ok(angleOf(nearB.normal) > 75 && angleOf(nearB.normal) < 90, `法向应靠近相机 B，实际 ${angleOf(nearB.normal)}°`);

  // 两路相机各占 0.5 权重：法向落在两路各自角平分线之间
  const viewerMid = { x: 5, y: 3, z: 5 };
  const nA = angleOf(resolveCardPose(person, [{ pos: camA, weight: 1 }], viewerMid).normal);
  const nB = angleOf(resolveCardPose(person, [{ pos: camB, weight: 1 }], viewerMid).normal);
  const nMix = angleOf(resolveCardPose(person, [{ pos: camA, weight: 0.5 }, { pos: camB, weight: 0.5 }], viewerMid).normal);
  assert.ok(nMix > Math.min(nA, nB) - 1e-6 && nMix < Math.max(nA, nB) + 1e-6, `混合法向 ${nMix}° 应介于 ${nA}° 与 ${nB}° 之间`);

  // 直立目标：检测框宽度按身高比例封顶；提供真实占地宽度时直接采用
  const wide = resolveCardPose({ ...person, width: 2.4 }, [{ pos: camA, weight: 1 }], camA);
  assert.ok(wide.bodyRadius < 0.45 && !wide.flat, `直立目标躯干半径应受身高约束，实际 ${wide.bodyRadius}`);
  const exact = resolveCardPose({ ...person, width: 2.4, footprint: 0.52 }, [{ pos: camA, weight: 1 }], camA);
  assert.ok(Math.abs(exact.bodyRadius - 0.31) < 1e-6, `应采用真实占地宽度，实际 ${exact.bodyRadius}`);

  // 低矮目标（AGV）改用盖在顶面高度的水平面片
  const agv = resolveCardPose({ x: 0, y: 0, z: 0, width: 0.4, height: 0.2, footprint: 1.43 }, [{ pos: camA, weight: 1 }], camB);
  assert.ok(agv.flat && Math.abs(agv.height - 0.2) < 1e-9 && agv.bodyRadius > 0.7, 'AGV 应使用水平面片并覆盖完整占地');

  console.log(`[PASS] 8. 自由视点融合面片解算（角平分线朝向 ${angleOf(off60.normal).toFixed(1)}° + 水平面片 + 尺寸先验）`);
}

// ==========================================
// 9. 自由视点融合：按目标选择主相机（分数边际 + 保持时长滞回）
// ==========================================
{
  const cands = (a, b) => [{ camId: 'A', score: a }, { camId: 'B', score: b }];

  const init = selectPrimaryCamera(cands(0.2, 0.6), undefined, 1000);
  assert.deepEqual(init, { camId: 'B', prevCamId: null, switchMs: 1000 }, '首次应直接选最高分且不淡入');

  const state = { camId: 'A', prevCamId: null, switchMs: 1000 };
  assert.equal(selectPrimaryCamera(cands(0.5, 0.57), state, 5000), state, '分数差未超过边际不应切换');
  assert.equal(selectPrimaryCamera(cands(-1, 0.9), state, 1300), state, '保持时长未满即使分数差很大也不应切换');

  const switched = selectPrimaryCamera(cands(0.2, 0.5), state, 1400);
  assert.deepEqual(switched, { camId: 'B', prevCamId: 'A', switchMs: 1400 }, '保持时长已满且领先边际应切换并记录前一路相机');

  const lost = selectPrimaryCamera([{ camId: 'B', score: 0.1 }], state, 1050);
  assert.deepEqual(lost, { camId: 'B', prevCamId: null, switchMs: 1050 }, '当前相机消失应立即切换且不淡入');

  console.log('[PASS] 9. 自由视点主相机选择（边际 + 保持时长滞回 + 相机丢失立即切换）');
}

// ==========================================
// 10. 占据度峰值提取：平滑 + 局部极大 + NMS + 加权质心细化
// ==========================================
{
  const cols = 64;
  const rows = 64;
  const cellSize = 0.1;
  const data = new Uint8Array(cols * rows * 4);
  const sigma = 1.5; // 单元
  const addBlob = (cx, cy, peak, nCams, heightCount) => {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = peak * Math.exp(-((c - cx) ** 2 + (r - cy) ** 2) / (2 * sigma * sigma));
        const i = (r * cols + c) * 4;
        data[i] = Math.max(data[i], Math.round(v * 255));
        if (v > 0.1) {
          data[i + 1] = Math.round((nCams / 6) * 255);
          data[i + 2] = Math.round((heightCount / 8) * 255);
        }
      }
    }
  };
  const A = { c: 20, r: 30 };
  const B = { c: 27.4, r: 30.3 }; // 与 A 相距 0.74 m，落在小数单元位置
  addBlob(A.c, A.r, 1.0, 3, 7);
  addBlob(B.c, B.r, 1.0, 2, 6);
  addBlob(50, 12, 0.35, 1, 2);

  const peaks = extractOccupancyPeaks(data, cols, rows, { cellSize });
  assert.equal(peaks.length, 2, `应恰好检出 2 个峰，实际 ${peaks.length}`);
  const near = (p, t) => Math.hypot(p.col - t.c, p.row - t.r) * cellSize;
  const pa = peaks.find((p) => near(p, A) < 0.3);
  const pb = peaks.find((p) => near(p, B) < 0.3);
  assert.ok(pa && pb, '两个峰应分别落在两个目标附近');
  assert.ok(near(pa, A) < 0.05, `A 质心误差 ${near(pa, A).toFixed(3)} m`);
  assert.ok(near(pb, B) < 0.05, `B 质心误差 ${near(pb, B).toFixed(3)} m`);
  assert.equal(pa.nCams, 3);
  assert.equal(pa.heightCount, 7);
  assert.equal(pb.nCams, 2);
  assert.equal(pb.heightCount, 6);
  assert.ok(pa.area > 0.1 && pa.area < 1.5, `面积应为合理的人体占地，实际 ${pa.area}`);

  console.log(`[PASS] 10. 占据度峰值提取（2 峰，质心误差 ${near(pa, A).toFixed(3)} / ${near(pb, B).toFixed(3)} m，弱峰被忽略）`);
}

// ==========================================
// 11. 自动色彩匹配：由两两对数颜色比解出各路相机增益（仅 a-b、b-c 配对，孤立相机增益为 1）
// ==========================================
{
  // 相机观测亮度倍率（逐通道）：a 为基准，b 偏亮 25%，c 偏暗 20%
  const mult = { a: [1.0, 1.0, 1.0], b: [1.25, 1.1, 1.25], c: [0.8, 0.9, 0.8] };
  const pairOf = (x, y, weight) => ({ a: x, b: y, logRatio: mult[x].map((m, ch) => Math.log(m / mult[y][ch])), weight });
  const gains = solveColorGains(['a', 'b', 'c', 'iso'], [pairOf('a', 'b', 120), pairOf('b', 'c', 80)]);

  for (let ch = 0; ch < 3; ch++) {
    const corrected = ['a', 'b', 'c'].map((id) => gains.get(id)[ch] * mult[id][ch]);
    const spread = Math.max(...corrected) / Math.min(...corrected) - 1;
    assert.ok(spread < 0.02, `通道 ${ch} 校正后三路相机应一致，相对差 ${(spread * 100).toFixed(2)}%`);
    const geoMean = Math.cbrt(['a', 'b', 'c'].reduce((p, id) => p * gains.get(id)[ch], 1));
    assert.ok(Math.abs(geoMean - 1) < 0.01, `通道 ${ch} 增益几何均值应为 1，实际 ${geoMean}`);
  }
  assert.deepEqual(gains.get('iso'), [1, 1, 1], '没有任何配对的相机增益应为 1');

  console.log(`[PASS] 11. 自动色彩匹配求解（增益 b=${gains.get('b').map((g) => g.toFixed(3))}，孤立相机 [1,1,1]）`);
}
