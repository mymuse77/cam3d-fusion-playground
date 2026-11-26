import * as THREE from 'three';

/**
 * 构建内置 PBR 工业园区练习场景
 * 包含地面、主厂房、侧翼配电室、前排遮挡立柱、设备箱，以及东侧带斜坡的 0.8m 巡检高台
 */
export function createPracticeScene() {
  const group = new THREE.Group();
  group.name = 'PracticeParkScene';

  const makeMat = (hex) =>
    new THREE.MeshStandardMaterial({ color: hex, roughness: 0.85, metalness: 0.1 });

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), makeMat(0x3a4048));
  ground.rotation.x = -Math.PI / 2;
  ground.name = 'Ground';
  ground.userData.isWalkable = true;
  group.add(ground);

  const grid = new THREE.GridHelper(80, 40, 0x555555, 0x2c3035);
  grid.position.y = 0.02;
  grid.userData.excludeFromDepth = true;
  group.add(grid);

  const mainBuilding = new THREE.Mesh(new THREE.BoxGeometry(18, 8, 8), makeMat(0x7d8a97));
  mainBuilding.position.set(0, 4, -8);
  group.add(mainBuilding);

  const sideBuilding = new THREE.Mesh(new THREE.BoxGeometry(6, 5, 10), makeMat(0x6b7785));
  sideBuilding.position.set(-12, 2.5, -2);
  group.add(sideBuilding);

  // 前排遮挡立柱与设备箱：测试深度遮挡、PCF 软边缘与掠射角防拉丝
  const pillarLeft = new THREE.Mesh(new THREE.BoxGeometry(2.0, 5.5, 2.0), makeMat(0xb08968));
  pillarLeft.position.set(-3, 2.75, 1);
  group.add(pillarLeft);

  const equipmentRight = new THREE.Mesh(new THREE.BoxGeometry(3.2, 2.4, 2.0), makeMat(0x52796f));
  equipmentRight.position.set(3.5, 1.2, 2);
  group.add(equipmentRight);

  // 东侧 0.8m 巡检平台 (x ∈ [4.5, 9.5], z ∈ [-5.5, 0.5])：测试高差地形三维射线反投影
  const platform = new THREE.Mesh(new THREE.BoxGeometry(5.0, 0.8, 6.0), makeMat(0x586370));
  platform.position.set(7.0, 0.4, -2.5);
  platform.name = 'InspectionPlatform';
  platform.userData.isWalkable = true;
  group.add(platform);

  // 平台斜坡引道 (x ∈ [5.5, 8.5], z 从 3.5 爬升至 0.5, 高度 0 -> 0.8m)
  const rampPositions = new Float32Array([
    // 顶面斜坡 (两个逆时针三角形，法线朝上偏 +Z)
    5.5, 0.0, 3.5,   8.5, 0.0, 3.5,   8.5, 0.8, 0.5,
    5.5, 0.0, 3.5,   8.5, 0.8, 0.5,   5.5, 0.8, 0.5,
    // 左侧立面
    5.5, 0.0, 3.5,   5.5, 0.8, 0.5,   5.5, 0.0, 0.5,
    // 右侧立面
    8.5, 0.0, 3.5,   8.5, 0.0, 0.5,   8.5, 0.8, 0.5,
    // 背面立面 (z = 0.5)
    5.5, 0.0, 0.5,   5.5, 0.8, 0.5,   8.5, 0.8, 0.5,
    5.5, 0.0, 0.5,   8.5, 0.8, 0.5,   8.5, 0.0, 0.5
  ]);
  const rampGeo = new THREE.BufferGeometry();
  rampGeo.setAttribute('position', new THREE.BufferAttribute(rampPositions, 3));
  rampGeo.computeVertexNormals();
  const ramp = new THREE.Mesh(rampGeo, makeMat(0x64748b));
  ramp.name = 'InspectionRamp';
  ramp.userData.isWalkable = true;
  group.add(ramp);

  return group;
}
