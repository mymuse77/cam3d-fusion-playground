import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import GUI from 'lil-gui';

import { createPracticeScene } from './scene/createPracticeScene.js';
import { injectMultiProjection } from './shaders/injectMultiProjection.js';
import { disposeObject3D } from './utils/disposeUtils.js';
import { MultiProjectorManager, syncDepthCameraFromIntrinsics } from './utils/multiProjector.js';
import { CameraScheduler } from './utils/cameraScheduler.js';
import { PTZController } from './utils/ptzController.js';
import { PointPicker } from './utils/pointPicker.js';
import { BBoxProjector, collectWalkableMeshes } from './utils/bboxProjector.js';
import { MultiCamTracker3D } from './utils/multiCamTracker3D.js';
import { Geofence3D, TargetVisualManager } from './utils/geofenceAndVisuals.js';
import { VideoStreamManager } from './utils/videoStreamManager.js';
import { mountWebCalibratorModal } from './utils/webCalibrator.js';
import { createTestCanvasTexture, generateMockYoloDetections } from './utils/mockData.js';

// ==========================================
// 1. 初始化渲染器、场景、光照与主观察相机
// ==========================================
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14181f);
scene.add(new THREE.AmbientLight(0xffffff, 0.85));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.4);
dirLight.position.set(20, 35, 20);
scene.add(dirLight);

const mainCamera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 400);
mainCamera.position.set(16, 15, 24);
const controls = new OrbitControls(mainCamera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 2, -2);

// 辅助对象分组（Pass 1 深度图渲染时统一隐藏，防止污染深度缓冲）
const helperGroup = new THREE.Group();
scene.add(helperGroup);
const yoloVisualGroup = new THREE.Group();
yoloVisualGroup.visible = false;
scene.add(yoloVisualGroup);

// 3D 多边形电子围栏（覆盖园区中央区域）
const geofence = new Geofence3D([
  [-2.5, -3.5],
  [2.5, -3.5],
  [2.5, 1.5],
  [-2.5, 1.5]
], 2.5);
geofence.group.visible = false;
scene.add(geofence.group);

const targetVisualMgr = new TargetVisualManager(yoloVisualGroup, 45);
const alarmBanner = document.getElementById('alarm-banner');
const statusBar = document.getElementById('status-bar');
const streamCamSelect = document.getElementById('stream-cam-select');

// ==========================================
// 2. 初始化多路投影管理器、视线焦点调度器与流生命周期管理器
// ==========================================
const projectorMgr = new MultiProjectorManager(scene);
const scheduler = new CameraScheduler(projectorMgr, mainCamera, helperGroup, () => controls.target);

function updateStatusBar(extraMsg = '') {
  const base = `激活槽位: ${scheduler.slotMap.length} / 4 (池总数: ${scheduler.logicalCameras.length})`;
  statusBar.textContent = extraMsg ? `${base} | ${extraMsg}` : base;
}

const streamMgr = new VideoStreamManager({
  onTextureChange(camId, texture) {
    const item = scheduler.getCameraById(camId);
    if (!item) return;
    item.texture = texture;
    const slot = scheduler.getSlotIndexById(camId);
    if (slot >= 0) {
      projectorMgr.sharedUniforms.uVideoTex.value[slot] = texture;
    } else {
      scheduler.update(performance.now(), true);
    }
  },
  onStatus(msg) {
    updateStatusBar(msg);
  }
});

function injectSceneMaterials(root) {
  root.traverse((child) => {
    if (!child.isMesh) return;
    if (child.geometry && !child.geometry.attributes.normal) {
      child.geometry.computeVertexNormals();
    }
    const mats = Array.isArray(child.material) ? child.material : [child.material];
    mats.forEach((m) => injectMultiProjection(m, projectorMgr.sharedUniforms));
  });
}

let activeParkModel = createPracticeScene();
injectSceneMaterials(activeParkModel);
scene.add(activeParkModel);
let walkableMeshes = collectWalkableMeshes(activeParkModel);

// ==========================================
// 3. 初始化 3D 采点工具（练习 4）
// ==========================================
const pickerListEl = document.getElementById('picker-list');
const pointPicker = new PointPicker(mainCamera, renderer.domElement, activeParkModel, (pts) => {
  pickerListEl.innerHTML = pts.length
    ? pts.map((p, i) => `#${i + 1}: [${p.x}, ${p.y}, ${p.z}]`).join('<br/>')
    : '尚未采点（建议采集 6 个以上不共面的点）';
});
scene.add(pointPicker.markerGroup);

document.getElementById('btn-copy-pts').onclick = async () => {
  if (!pointPicker.points.length) return alert('请先按住 Shift + 左键点击模型采点');
  await navigator.clipboard.writeText(pointPicker.toPythonString());
  alert(`已复制 ${pointPicker.points.length} 个 3D 点到剪贴板！`);
};
document.getElementById('btn-undo-pt').onclick = () => pointPicker.undo();
document.getElementById('btn-clear-pts').onclick = () => pointPicker.clear();

const hiddenInDepthPass = [helperGroup, pointPicker.markerGroup, yoloVisualGroup, geofence.group];

// ==========================================
// 4. 配置加载与热重载机制
// ==========================================
const bboxProjectors = new Map();
const tracker3D = new MultiCamTracker3D({ personThreshold: 1.8 });
let currentConfig = null;
let ptzItem = null;

function refreshStreamCameraSelect(cameras) {
  const prev = streamCamSelect.value;
  streamCamSelect.innerHTML = '';
  cameras.forEach((c) => {
    const opt = document.createElement('option');
    opt.value = c.id;
    opt.textContent = `${c.name} (${c.id})`;
    streamCamSelect.appendChild(opt);
  });
  if (prev && cameras.some((c) => c.id === prev)) {
    streamCamSelect.value = prev;
  }
}

async function applyCamerasConfig(config) {
  currentConfig = config;
  scheduler.clearAll();
  bboxProjectors.clear();
  streamMgr.disposeAll();
  ptzItem = null;

  if (config.scene?.glbUrl) {
    try {
      await loadSceneModelFromUrl(config.scene.glbUrl, false);
    } catch (err) {
      console.error('场景 GLB 加载失败:', err);
      alert('预设场景 GLB 加载失败: ' + err.message);
    }
  }

  const colors = ['#00e5ff', '#ff9f1c', '#a855f7'];
  config.cameras.forEach((cfg, idx) => {
    if (cfg.enabled === false) return;
    const intr = { ...cfg.intrinsics };
    const dist = { k1: 0, k2: 0, p1: 0, p2: 0, ...(cfg.distortion || {}) };
    const colorAdjust = { brightness: 0, contrast: 1, saturation: 1, ...(cfg.colorAdjust || {}) };

    const cam = new THREE.PerspectiveCamera(
      intr.fovY * (intr.depthFovScale || 1.0),
      intr.width / intr.height,
      intr.near || 0.5,
      intr.far || 60
    );
    syncDepthCameraFromIntrinsics(cam, intr);

    if (cfg.extrinsics?.matrixWorldRowMajor) {
      const m = new THREE.Matrix4().set(...cfg.extrinsics.matrixWorldRowMajor);
      m.decompose(cam.position, cam.quaternion, new THREE.Vector3());
    } else if (cfg.position && cfg.target) {
      cam.position.set(...cfg.position);
      cam.lookAt(...cfg.target);
    }
    cam.updateMatrixWorld(true);

    if (cfg.extrinsics?.reprojectionErrorPx > 3.0) {
      console.warn(`[标定精度提示] ${cfg.id} 重投影误差偏高: ${cfg.extrinsics.reprojectionErrorPx} px`);
    }

    const canvasAsset = createTestCanvasTexture(cfg.id.toUpperCase(), colors[idx % colors.length]);
    const ptzCtrl = cfg.type === 'ptz' ? new PTZController(cam, intr, dist, cfg.ptz) : null;

    const item = scheduler.registerCamera({
      id: cfg.id,
      name: cfg.name || cfg.id,
      type: cfg.type || 'fixed',
      camera: cam,
      intrinsics: intr,
      distortion: dist,
      colorAdjust,
      texture: canvasAsset.texture,
      ptzController: ptzCtrl,
      rawConfig: cfg
    });

    if (cfg.stream?.url) {
      streamMgr.bindStreamUrl(cfg.id, cfg.stream.url, cfg.stream.type || 'auto', cfg.stream.autoSuspend !== false);
    } else {
      streamMgr.bindCanvas(cfg.id, canvasAsset);
    }

    if (cfg.type === 'ptz') ptzItem = item;
    bboxProjectors.set(cfg.id, new BBoxProjector(cam, intr, dist, walkableMeshes));
  });

  scheduler.update(performance.now(), true);
  streamMgr.syncActiveCameras(scheduler.activeIds, performance.now());
  projectorMgr.renderAllDepthMaps(renderer, hiddenInDepthPass);
  updateStatusBar();
  refreshTuneCameraOptions();
  refreshStreamCameraSelect(scheduler.logicalCameras);
}

fetch('/config/park_cameras.json')
  .then((r) => r.json())
  .then(applyCamerasConfig);

// JSON 热重载上传
document.getElementById('json-upload').onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    applyCamerasConfig(data);
    alert(`已热重载配置，共加载 ${scheduler.logicalCameras.length} 路监控`);
  } catch (err) {
    alert('JSON 解析失败: ' + err.message);
  }
};

// ==========================================
// 5. 多路本地视频 / 远程流接入与离线 Draco GLB 上传
// ==========================================
document.getElementById('video-upload').onchange = (e) => {
  const file = e.target.files[0];
  const targetId = streamCamSelect.value || scheduler.logicalCameras[0]?.id;
  if (!file || !targetId) return;
  streamMgr.bindLocalFile(targetId, file);
};

document.getElementById('btn-bind-stream').onclick = () => {
  const url = document.getElementById('stream-url-input').value.trim();
  const targetId = streamCamSelect.value || scheduler.logicalCameras[0]?.id;
  if (!url || !targetId) return alert('请先输入有效的流地址（MP4 / .m3u8 / WHEP）');
  streamMgr.bindStreamUrl(targetId, url, 'auto', true);
};

document.getElementById('btn-reset-canvas').onclick = () => {
  const targetId = streamCamSelect.value || scheduler.logicalCameras[0]?.id;
  if (!targetId) return;
  const idx = scheduler.logicalCameras.findIndex((c) => c.id === targetId);
  const colors = ['#00e5ff', '#ff9f1c', '#a855f7'];
  const asset = createTestCanvasTexture(targetId.toUpperCase(), colors[Math.max(idx, 0) % colors.length]);
  streamMgr.bindCanvas(targetId, asset);
  updateStatusBar(`[${targetId}] 已恢复动态测试画布`);
};

// 配置支持 Draco 压缩的 GLTFLoader（使用本地静态解码器，支持纯内网）
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath('/draco/');
const gltfLoader = new GLTFLoader();
gltfLoader.setDRACOLoader(dracoLoader);

function loadSceneModelFromUrl(url, autoCenter = false) {
  return new Promise((resolve, reject) => {
    gltfLoader.load(
      url,
      (gltf) => {
        scene.remove(activeParkModel);
        disposeObject3D(activeParkModel);

        activeParkModel = gltf.scene;

        if (autoCenter) {
          const box = new THREE.Box3().setFromObject(activeParkModel);
          const center = box.getCenter(new THREE.Vector3());
          const size = box.getSize(new THREE.Vector3());
          activeParkModel.position.set(-center.x, -box.min.y, -center.z);

          const maxDim = Math.max(size.x, size.y, size.z, 10);
          controls.target.set(0, size.y * 0.3, 0);
          mainCamera.position.set(maxDim * 0.9, maxDim * 0.7, maxDim * 1.1);
          controls.update();
        }

        injectSceneMaterials(activeParkModel);
        scene.add(activeParkModel);
        pointPicker.targetRoot = activeParkModel;
        walkableMeshes = collectWalkableMeshes(activeParkModel);
        bboxProjectors.forEach((proj) => proj.setWalkableMeshes(walkableMeshes));
        projectorMgr.renderAllDepthMaps(renderer, hiddenInDepthPass);
        resolve(activeParkModel);
      },
      undefined,
      reject
    );
  });
}

function restoreDefaultPracticeScene() {
  scene.remove(activeParkModel);
  disposeObject3D(activeParkModel);
  activeParkModel = createPracticeScene();
  injectSceneMaterials(activeParkModel);
  scene.add(activeParkModel);
  pointPicker.targetRoot = activeParkModel;
  walkableMeshes = collectWalkableMeshes(activeParkModel);
  bboxProjectors.forEach((proj) => proj.setWalkableMeshes(walkableMeshes));
  projectorMgr.renderAllDepthMaps(renderer, hiddenInDepthPass);
}

document.getElementById('btn-load-blender').onclick = async () => {
  try {
    const res = await fetch('/demo-blender/park_cameras_blender.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await applyCamerasConfig(await res.json());
    updateStatusBar('已加载 Blender 5.3 成套实景与双路 MP4');
  } catch (err) {
    alert('加载 Blender 5.3 套组失败，请先运行 blender-tools/generate_blender_suite.py: ' + err.message);
  }
};

document.getElementById('btn-load-dirty').onclick = async () => {
  try {
    const res = await fetch('/demo-blender-dirty/park_cameras_dirty.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await applyCamerasConfig(await res.json());
    updateStatusBar('已加载实景脏数据压测套组 (广角畸变+色差+模实偏差)');
  } catch (err) {
    alert('加载实景脏数据压测套组失败，请先运行 blender-tools/generate_dirty_suite.py: ' + err.message);
  }
};

document.getElementById('btn-load-default').onclick = async () => {
  restoreDefaultPracticeScene();
  const res = await fetch('/config/park_cameras.json');
  await applyCamerasConfig(await res.json());
  updateStatusBar('已恢复内置默认园区场景');
};

let lastGlbUrl = null;
document.getElementById('glb-upload').onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  if (lastGlbUrl) URL.revokeObjectURL(lastGlbUrl);
  lastGlbUrl = URL.createObjectURL(file);

  try {
    await loadSceneModelFromUrl(lastGlbUrl, document.getElementById('glb-auto-center').checked);
  } catch (err) {
    alert('GLB 加载失败: ' + err.message);
  }
};

// ==========================================
// 6. GUI 控制台：画质调优 + 可视化微调导出 + 6 大练习
// ==========================================
const state = {
  enableOcclusion: true,
  enablePCF: true,
  depthBias: 0.0025,
  grazingThreshold: 0.12,
  opacity: 0.9,
  feather: 0.05,
  showHelpers: true,
  ptzAutoPatrol: false,
  ptzLatencyMs: 300,
  stressTest30Cams: false,
  enableMockYolo: false,
  enableGeofenceAlarm: true,
  showGeofence: false
};

const gui = new GUI({ title: '数字孪生 v3.0 实操控制台' });

const f1 = gui.addFolder('练习 1：投影画质、PCF 与去拉伸');
f1.add(state, 'enableOcclusion').name('开启深度遮挡').onChange((v) => (projectorMgr.sharedUniforms.uEnableOcclusion.value = v));
f1.add(state, 'enablePCF').name('PCF 3×3 软边缘').onChange((v) => (projectorMgr.sharedUniforms.uEnablePCF.value = v));
f1.add(state, 'depthBias', 0.0001, 0.02, 0.0002).name('深度偏移 Bias').onChange((v) => (projectorMgr.sharedUniforms.uDepthBias.value = v));
f1.add(state, 'grazingThreshold', 0.0, 0.5, 0.02).name('掠射角淡出阈值').onChange((v) => (projectorMgr.sharedUniforms.uGrazingThreshold.value = v));
f1.add(state, 'opacity', 0, 1, 0.05).name('融合透明度').onChange((v) => (projectorMgr.sharedUniforms.uOpacity.value = v));
f1.add(state, 'feather', 0, 0.2, 0.005).name('边缘羽化').onChange((v) => (projectorMgr.sharedUniforms.uFeather.value = v));
f1.add(state, 'showHelpers').name('显示视锥体线框').onChange((v) => (helperGroup.visible = v));

const tuneState = {
  selectedId: 'cam_fixed_01',
  posX: 0, posY: 8, posZ: 10,
  yawDeg: 0, pitchDeg: 0, rollDeg: 0,
  fovY: 45, k1: 0, k2: 0,
  brightness: 0, contrast: 1, saturation: 1,
  exportJson: () => exportUpdatedParkJson()
};

const fTune = gui.addFolder('练习 2 & 4：相机微调与 JSON 导出');
let camSelectCtrl = fTune.add(tuneState, 'selectedId', ['cam_fixed_01']).name('选中相机').onChange(loadSelectedCamToGui);
const tuneCtrls = [
  fTune.add(tuneState, 'posX', -40, 40, 0.1).name('位置 X').onChange(applyGuiToSelectedCam),
  fTune.add(tuneState, 'posY', 0.5, 30, 0.1).name('位置 Y').onChange(applyGuiToSelectedCam),
  fTune.add(tuneState, 'posZ', -40, 40, 0.1).name('位置 Z').onChange(applyGuiToSelectedCam),
  fTune.add(tuneState, 'yawDeg', -180, 180, 0.5).name('水平角 Yaw(°)').onChange(applyGuiToSelectedCam),
  fTune.add(tuneState, 'pitchDeg', -89, 89, 0.5).name('俯仰角 Pitch(°)').onChange(applyGuiToSelectedCam),
  fTune.add(tuneState, 'rollDeg', -45, 45, 0.5).name('滚转角 Roll(°)').onChange(applyGuiToSelectedCam),
  fTune.add(tuneState, 'fovY', 10, 100, 0.5).name('视场角 FOV(°)').onChange(applyGuiToSelectedCam),
  fTune.add(tuneState, 'k1', -0.6, 0.6, 0.01).name('径向畸变 k1').onChange(applyGuiToSelectedCam),
  fTune.add(tuneState, 'k2', -0.3, 0.3, 0.01).name('径向畸变 k2').onChange(applyGuiToSelectedCam),
  fTune.add(tuneState, 'brightness', -0.4, 0.4, 0.02).name('画面亮度').onChange(applyGuiToSelectedCam),
  fTune.add(tuneState, 'contrast', 0.5, 1.8, 0.05).name('画面对比度').onChange(applyGuiToSelectedCam),
  fTune.add(tuneState, 'saturation', 0.0, 2.0, 0.05).name('画面饱和度').onChange(applyGuiToSelectedCam)
];
fTune.add(tuneState, 'exportJson').name('一键导出 park_cameras.json');

function refreshTuneCameraOptions() {
  const ids = scheduler.logicalCameras.map((c) => c.id);
  if (!ids.length) return;
  camSelectCtrl = camSelectCtrl.options(ids);
  tuneState.selectedId = ids[0];
  camSelectCtrl.updateDisplay();
  loadSelectedCamToGui();
}

function loadSelectedCamToGui() {
  const item = scheduler.getCameraById(tuneState.selectedId);
  if (!item) return;
  const e = new THREE.Euler().setFromQuaternion(item.camera.quaternion, 'YXZ');
  tuneState.posX = +item.camera.position.x.toFixed(2);
  tuneState.posY = +item.camera.position.y.toFixed(2);
  tuneState.posZ = +item.camera.position.z.toFixed(2);
  tuneState.pitchDeg = +THREE.MathUtils.radToDeg(e.x).toFixed(1);
  tuneState.yawDeg = +THREE.MathUtils.radToDeg(e.y).toFixed(1);
  tuneState.rollDeg = +THREE.MathUtils.radToDeg(e.z).toFixed(1);
  tuneState.fovY = +(item.intrinsics.fovY || 45).toFixed(1);
  tuneState.k1 = item.distortion.k1 || 0;
  tuneState.k2 = item.distortion.k2 || 0;
  tuneState.brightness = item.colorAdjust.brightness ?? 0;
  tuneState.contrast = item.colorAdjust.contrast ?? 1;
  tuneState.saturation = item.colorAdjust.saturation ?? 1;
  tuneCtrls.forEach((c) => c.updateDisplay());
}

function applyGuiToSelectedCam() {
  const item = scheduler.getCameraById(tuneState.selectedId);
  if (!item) return;

  item.camera.position.set(tuneState.posX, tuneState.posY, tuneState.posZ);
  item.camera.rotation.set(
    THREE.MathUtils.degToRad(tuneState.pitchDeg),
    THREE.MathUtils.degToRad(tuneState.yawDeg),
    THREE.MathUtils.degToRad(tuneState.rollDeg),
    'YXZ'
  );

  const fovRad = THREE.MathUtils.degToRad(tuneState.fovY);
  const fy = (item.intrinsics.height * 0.5) / Math.tan(fovRad * 0.5);
  item.intrinsics.fovY = tuneState.fovY;
  item.intrinsics.fx = fy;
  item.intrinsics.fy = fy;
  syncDepthCameraFromIntrinsics(item.camera, item.intrinsics);

  item.distortion.k1 = tuneState.k1;
  item.distortion.k2 = tuneState.k2;
  item.colorAdjust.brightness = tuneState.brightness;
  item.colorAdjust.contrast = tuneState.contrast;
  item.colorAdjust.saturation = tuneState.saturation;

  scheduler.refreshCameraBounds(item);
  if (item.ptzController) {
    item.ptzController.syncBaseFromCamera(item.intrinsics, item.distortion);
  }
  bboxProjectors.get(item.id)?.setCalibration(item.intrinsics, item.distortion);

  const slot = scheduler.getSlotIndexById(item.id);
  if (slot >= 0) {
    projectorMgr.setCalibration(slot, item.intrinsics, item.distortion, item.colorAdjust);
    projectorMgr.renderSingleDepthMap(renderer, slot, hiddenInDepthPass);
  }
}

function exportUpdatedParkJson() {
  const out = {
    schemaVersion: '3.0.0',
    scene: currentConfig?.scene || { coordinateSystem: 'threejs-y-up', unit: 'meter' },
    cameras: scheduler.logicalCameras.map((item) => {
      let matrixWorld;
      let intrOut = item.intrinsics;
      let distOut = item.distortion;
      if (item.ptzController) {
        matrixWorld = item.ptzController.getBaseMatrixWorld();
        intrOut = item.ptzController.baseIntrinsics;
        distOut = item.ptzController.baseDistortion;
      } else {
        item.camera.updateMatrixWorld(true);
        matrixWorld = item.camera.matrixWorld;
      }
      const e = matrixWorld.elements;
      const rowMajor = [
        e[0], e[4], e[8], e[12],
        e[1], e[5], e[9], e[13],
        e[2], e[6], e[10], e[14],
        e[3], e[7], e[11], e[15]
      ].map((v) => +v.toFixed(6));

      return {
        ...(item.rawConfig || {}),
        id: item.id,
        name: item.name,
        type: item.type,
        enabled: true,
        intrinsics: intrOut,
        distortion: distOut,
        colorAdjust: item.colorAdjust,
        extrinsics: { matrixWorldRowMajor: rowMajor }
      };
    })
  };

  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'park_cameras.json';
  a.click();
  URL.revokeObjectURL(url);
}

// 挂载 Web 端 2D/3D 联合标定弹窗
mountWebCalibratorModal({
  getSelectedCamera: () => scheduler.getCameraById(tuneState.selectedId) || scheduler.logicalCameras[0],
  getCaptureSource: (camId) => streamMgr.getCaptureSource(camId),
  getPoints3D: () => pointPicker.points,
  onApplyCalibration(camItem, result) {
    camItem.camera.position.copy(result.position);
    camItem.camera.quaternion.copy(result.quaternion);
    camItem.camera.updateMatrixWorld(true);

    scheduler.refreshCameraBounds(camItem);
    if (camItem.ptzController) {
      camItem.ptzController.syncBaseFromCamera(camItem.intrinsics, camItem.distortion);
    }
    bboxProjectors.get(camItem.id)?.setCalibration(camItem.intrinsics, camItem.distortion);
    loadSelectedCamToGui();

    const slot = scheduler.getSlotIndexById(camItem.id);
    if (slot >= 0) {
      projectorMgr.renderSingleDepthMap(renderer, slot, hiddenInDepthPass);
    }
    updateStatusBar(`[${camItem.id}] Web PnP 标定完成 (${result.reprojectionErrorPx} px)`);
  }
});

// 文件夹 3：球机 PTZ 巡航与变焦（练习 3）
const f3 = gui.addFolder('练习 3：球机 PTZ 巡航与光学变焦');
f3.add(state, 'ptzAutoPatrol').name('开启球机自动巡航(含变焦)');
f3.add(state, 'ptzLatencyMs', 0, 1000, 50).name('视频延迟补偿(ms)').onChange((v) => {
  if (ptzItem?.ptzController) ptzItem.ptzController.videoLatencyMs = v;
});

// 文件夹 4：大园区调度与 AI 业务闭环（练习 5 & 6）
const f5 = gui.addFolder('练习 5 & 6：30路调度与 AI 闭环');
f5.add(state, 'stressTest30Cams').name('注入 27 路虚拟监控(共30路)').onChange((v) => {
  if (!v || scheduler.logicalCameras.length >= 30 || !scheduler.logicalCameras.length) return;
  const baseTex = scheduler.logicalCameras[0].texture;
  for (let i = 4; i <= 30; i++) {
    const cam = new THREE.PerspectiveCamera(45, 16 / 9, 0.5, 30);
    const angle = (i / 27) * Math.PI * 2;
    cam.position.set(Math.cos(angle) * 32, 9, Math.sin(angle) * 32);
    cam.lookAt(0, 2, 0);
    cam.updateMatrixWorld(true);
    scheduler.registerCamera({
      id: `cam_stress_${i}`, name: `虚拟监控 ${i}`, type: 'fixed', camera: cam, texture: baseTex,
      intrinsics: { width: 1920, height: 1080, fx: 1300, fy: 1300, cx: 960, cy: 540, fovY: 45, depthFovScale: 1.0 },
      distortion: { k1: 0, k2: 0, p1: 0, p2: 0 }
    });
  }
  scheduler.update(performance.now(), true);
  projectorMgr.renderAllDepthMaps(renderer, hiddenInDepthPass);
  updateStatusBar();
});
f5.add(state, 'enableMockYolo').name('开启模拟 YOLO 跨镜追踪').onChange((v) => {
  yoloVisualGroup.visible = v;
  if (!v) {
    alarmBanner.style.display = 'none';
    geofence.setAlarmState(false);
  }
});
f5.add(state, 'enableGeofenceAlarm').name('开启电子围栏越界检测').onChange((v) => {
  geofence.enabled = v;
  if (!v) {
    alarmBanner.style.display = 'none';
    geofence.setAlarmState(false);
  }
});
f5.add(state, 'showGeofence').name('显示 3D 电子围栏光墙').onChange((v) => {
  geofence.group.visible = v;
});

// ==========================================
// 7. 主渲染循环
// ==========================================
const clock = new THREE.Clock();
let frameCount = 0;

function animate(now) {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  const elapsed = clock.getElapsedTime();
  frameCount++;

  controls.update();
  streamMgr.updateCanvases(elapsed);

  // A. 视线焦点感知动态调度（仅槽位变更时重绘深度图，并联动流媒体休眠/恢复）
  if (scheduler.update(now)) {
    projectorMgr.renderAllDepthMaps(renderer, hiddenInDepthPass);
    updateStatusBar();
  }
  if (frameCount % 30 === 0) {
    streamMgr.syncActiveCameras(scheduler.activeIds, now);
  }

  // B. 球机 PTZ 巡航（含 Pan + 光学变焦 Zoom 联动内参与畸变）
  if (ptzItem?.ptzController) {
    if (state.ptzAutoPatrol && frameCount % 6 === 0) {
      ptzItem.ptzController.pushOnvifData({
        pan: Math.sin(elapsed * 0.55) * 0.8,
        tilt: Math.sin(elapsed * 0.3) * 0.2,
        zoom: (Math.sin(elapsed * 0.4) + 1) * 0.35
      });
    }
    const ptzUpdate = ptzItem.ptzController.update(dt);
    if (ptzUpdate) {
      ptzItem.helper.update();
      ptzItem.intrinsics = ptzUpdate.intrinsics;
      ptzItem.distortion = ptzUpdate.distortion;
      bboxProjectors.get(ptzItem.id)?.setCalibration(ptzUpdate.intrinsics, ptzUpdate.distortion);
      const slot = scheduler.getSlotIndexById(ptzItem.id);
      if (slot >= 0) {
        projectorMgr.setCalibration(slot, ptzUpdate.intrinsics, ptzUpdate.distortion, ptzItem.colorAdjust);
        if (frameCount % 2 === 0) {
          projectorMgr.renderSingleDepthMap(renderer, slot, hiddenInDepthPass);
        }
      }
    }
  }

  // C. YOLO 2D 反投影（3D 地形网格求交） + 卡尔曼滤波与匈牙利跨镜追踪 + 电子围栏告警（练习 6）
  if (state.enableMockYolo && frameCount % 3 === 0) {
    const detMap = generateMockYoloDetections(elapsed, scheduler.logicalCameras);
    const observations = [];
    for (const [camId, dets] of detMap.entries()) {
      const proj = bboxProjectors.get(camId);
      if (!proj) continue;
      for (const d of dets) {
        const hit = proj.projectToScene(d.bbox, 0);
        if (hit) {
          observations.push({
            camId,
            localTrackId: d.trackId,
            cls: d.cls,
            x: hit.x,
            y: hit.y,
            z: hit.z,
            distToCam: hit.distanceTo(proj.camera.position)
          });
        }
      }
    }
    const tracks = tracker3D.update(observations, now);
    const alarmIds = targetVisualMgr.updateTargets(tracks, geofence);
    const hasAlarm = alarmIds.length > 0;
    geofence.setAlarmState(hasAlarm);
    alarmBanner.style.display = hasAlarm ? 'block' : 'none';
    if (hasAlarm) {
      alarmBanner.textContent = `电子围栏越界告警：目标 ID #${alarmIds.join(', #')} 闯入核心禁区！`;
    }
  }

  renderer.render(scene, mainCamera);
}
animate(performance.now());

window.addEventListener('resize', () => {
  mainCamera.aspect = window.innerWidth / window.innerHeight;
  mainCamera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
