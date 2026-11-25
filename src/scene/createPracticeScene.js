import * as THREE from 'three';

/**
 * 构建内置 PBR 工业园区练习场景
 * 包含地面、主厂房、侧翼配电室、前排遮挡立柱与设备箱
 */
export function createPracticeScene() {
  const group = new THREE.Group();
  group.name = 'PracticeParkScene';

  const makeMat = (hex) =>
    new THREE.MeshStandardMaterial({ color: hex, roughness: 0.85, metalness: 0.1 });

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), makeMat(0x3a4048));
  ground.rotation.x = -Math.PI / 2;
  ground.name = 'Ground';
  group.add(ground);

  const grid = new THREE.GridHelper(80, 40, 0x555555, 0x2c3035);
  grid.position.y = 0.02;
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

  return group;
}
