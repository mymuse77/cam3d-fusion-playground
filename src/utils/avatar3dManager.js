import * as THREE from 'three';

const WORKER_PALETTES = [
  { hat: 0xf97316, shirt: 0xf1f5f9, pants: 0xc2410c }, // 白衣橙帽红棕裤 (如 #350)
  { hat: 0xeab308, shirt: 0xf8fafc, pants: 0x0d9488 }, // 白衣黄帽青绿裤 (如 #625)
  { hat: 0x64748b, shirt: 0x94a3b8, pants: 0x1d4ed8 }, // 灰帽灰衣蓝背带裤 (如 #699)
  { hat: 0xf97316, shirt: 0xe2e8f0, pants: 0x334155 }  // 工装灰裤
];

function normalizeAngleDiff(a) {
  while (a > Math.PI) a -= Math.PI * 2;
  while (a < -Math.PI) a += Math.PI * 2;
  return a;
}

/**
 * 3D 立体数字替身管理器（方向 B / 混合增强模式）：
 * 根据 3D 轨迹（外部真值/AI 轨迹，或纯视觉立牌反投影轨迹）在真实 3D 坐标驱动：
 * - 普通工人（安全帽 + 工装 + 行走摆臂迈腿骨骼动画）
 * - 人形机器人（FourierGR1T2 风格黑钛金属机身 + 青色发光面罩 + 关节步态）
 * - AGV 搬运车（工业黄低矮底盘 + 激光雷达防撞条 + 警示灯）
 * 360° 任意视角零透视畸变、零抠像残缺。
 */
export class Avatar3DManager {
  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'Avatar3DGroup';
    this.group.userData.excludeFromDepth = true;
    this.visible = false;
    this.group.visible = false;

    this._externalTracks = null;
    this._entities = new Map(); // key -> { root, cls, limbs, prevX, prevZ, yaw, walkPhase, lastSeenMs }
    this._labelTexCache = new Map();

    this._ringGeo = new THREE.RingGeometry(0.32, 0.42, 32);
    this._ringGeo.rotateX(-Math.PI / 2);
  }

  setVisible(visible) {
    this.visible = visible;
    this.group.visible = visible;
  }

  setExternalTracks(tracksData) {
    this._externalTracks = tracksData;
    this._clearEntities();
  }

  clear() {
    this._externalTracks = null;
    this._clearEntities();
  }

  _clearEntities() {
    for (const ent of this._entities.values()) {
      this.group.remove(ent.root);
    }
    this._entities.clear();
  }

  _getLabelTexture(cls, idLabel) {
    const key = `${cls}_${idLabel}`;
    if (this._labelTexCache.has(key)) return this._labelTexCache.get(key);

    const canvas = document.createElement('canvas');
    canvas.width = 220;
    canvas.height = 56;
    const ctx = canvas.getContext('2d');

    const style =
      cls === 'robot'
        ? { bg: 'rgba(234, 88, 12, 0.90)', border: '#fdba74', text: `机器人 #${idLabel}` }
        : cls === 'agv'
          ? { bg: 'rgba(101, 163, 13, 0.90)', border: '#d9f99d', text: `AGV #${idLabel}` }
          : { bg: 'rgba(2, 132, 199, 0.90)', border: '#7dd3fc', text: `工人 #${idLabel}` };

    ctx.fillStyle = style.bg;
    ctx.strokeStyle = style.border;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.roundRect(6, 6, 208, 44, 12);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 26px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(style.text, 110, 29);

    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    this._labelTexCache.set(key, tex);
    return tex;
  }

  _stdMat(color, rough = 0.65, metal = 0.08, emissive = 0x000000, emissiveIntensity = 0.0) {
    const m = new THREE.MeshStandardMaterial({
      color,
      roughness: rough,
      metalness: metal,
      emissive,
      emissiveIntensity
    });
    m.userData.skipProjection = true;
    return m;
  }

  _buildWorkerAvatar(numId) {
    const pal = WORKER_PALETTES[Math.abs(numId) % WORKER_PALETTES.length];
    const body = new THREE.Group();

    const matSkin = this._stdMat(0xe5b899, 0.72, 0.02);
    const matHat = this._stdMat(pal.hat, 0.35, 0.12);
    const matShirt = this._stdMat(pal.shirt, 0.75, 0.05);
    const matPants = this._stdMat(pal.pants, 0.78, 0.05);
    const matShoe = this._stdMat(0x1e293b, 0.80, 0.10);

    // 躯干与骨盆
    const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.165, 0.34, 8, 14), matShirt);
    torso.position.y = 1.15;
    body.add(torso);

    const pelvis = new THREE.Mesh(new THREE.CylinderGeometry(0.155, 0.145, 0.16, 14), matPants);
    pelvis.position.y = 0.88;
    body.add(pelvis);

    // 头部 + 安全帽
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.115, 16, 14), matSkin);
    head.position.y = 1.54;
    body.add(head);

    const hatDome = new THREE.Mesh(
      new THREE.SphereGeometry(0.126, 16, 12, 0, Math.PI * 2, 0, Math.PI * 0.55),
      matHat
    );
    hatDome.position.y = 1.56;
    body.add(hatDome);

    const hatBrim = new THREE.Mesh(new THREE.CylinderGeometry(0.145, 0.145, 0.018, 16), matHat);
    hatBrim.position.set(0, 1.56, 0.018);
    body.add(hatBrim);

    // 四肢关节枢轴
    const makeArm = (side) => {
      const pivot = new THREE.Group();
      pivot.position.set(side * 0.215, 1.34, 0);
      const arm = new THREE.Mesh(new THREE.CapsuleGeometry(0.052, 0.38, 6, 10), matShirt);
      arm.position.y = -0.22;
      const hand = new THREE.Mesh(new THREE.SphereGeometry(0.05, 10, 10), matSkin);
      hand.position.y = -0.44;
      pivot.add(arm, hand);
      body.add(pivot);
      return pivot;
    };

    const makeLeg = (side) => {
      const pivot = new THREE.Group();
      pivot.position.set(side * 0.095, 0.84, 0);
      const leg = new THREE.Mesh(new THREE.CapsuleGeometry(0.068, 0.56, 6, 12), matPants);
      leg.position.y = -0.36;
      const shoe = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.08, 0.22), matShoe);
      shoe.position.set(0, -0.78, 0.035);
      pivot.add(leg, shoe);
      body.add(pivot);
      return pivot;
    };

    return {
      body,
      limbs: {
        armL: makeArm(-1),
        armR: makeArm(1),
        legL: makeLeg(-1),
        legR: makeLeg(1)
      },
      baseHeight: 1.72
    };
  }

  _buildRobotAvatar() {
    const body = new THREE.Group();
    const matArmor = this._stdMat(0x1f242d, 0.28, 0.78);
    const matJoint = this._stdMat(0x475569, 0.38, 0.85);
    const matVisor = this._stdMat(0x22d3ee, 0.15, 0.2, 0x06b6d4, 1.4);
    const matCore = this._stdMat(0xf97316, 0.2, 0.3, 0xea580c, 1.1);

    // 装甲胸腔 + 纤细机械腰 + 髋关节
    const chest = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.34, 0.22), matArmor);
    chest.position.y = 1.22;
    body.add(chest);

    const core = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.23, 14), matCore);
    core.rotation.x = Math.PI / 2;
    core.position.set(0, 1.24, 0.01);
    body.add(core);

    const waist = new THREE.Mesh(new THREE.CylinderGeometry(0.085, 0.105, 0.18, 12), matJoint);
    waist.position.y = 0.97;
    body.add(waist);

    const pelvis = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.13, 0.18), matArmor);
    pelvis.position.y = 0.83;
    body.add(pelvis);

    // 头部 + 青色发光传感器面罩
    const head = new THREE.Mesh(new THREE.CapsuleGeometry(0.095, 0.08, 8, 14), matArmor);
    head.position.y = 1.52;
    body.add(head);

    const visor = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.042, 0.12), matVisor);
    visor.position.set(0, 1.53, 0.055);
    body.add(visor);

    const makeArm = (side) => {
      const pivot = new THREE.Group();
      pivot.position.set(side * 0.235, 1.35, 0);
      const shoulder = new THREE.Mesh(new THREE.SphereGeometry(0.062, 12, 10), matJoint);
      const upper = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.04, 0.42, 10), matArmor);
      upper.position.y = -0.23;
      const hand = new THREE.Mesh(new THREE.SphereGeometry(0.048, 10, 10), matJoint);
      hand.position.y = -0.45;
      pivot.add(shoulder, upper, hand);
      body.add(pivot);
      return pivot;
    };

    const makeLeg = (side) => {
      const pivot = new THREE.Group();
      pivot.position.set(side * 0.105, 0.80, 0);
      const hip = new THREE.Mesh(new THREE.SphereGeometry(0.065, 12, 10), matJoint);
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.058, 0.046, 0.64, 12), matArmor);
      leg.position.y = -0.36;
      const knee = new THREE.Mesh(new THREE.SphereGeometry(0.058, 10, 10), matJoint);
      knee.position.y = -0.36;
      const foot = new THREE.Mesh(new THREE.BoxGeometry(0.11, 0.065, 0.23), matJoint);
      foot.position.set(0, -0.74, 0.04);
      pivot.add(hip, leg, knee, foot);
      body.add(pivot);
      return pivot;
    };

    return {
      body,
      limbs: {
        armL: makeArm(-1),
        armR: makeArm(1),
        legL: makeLeg(-1),
        legR: makeLeg(1)
      },
      baseHeight: 1.66
    };
  }

  _buildAgvAvatar() {
    const body = new THREE.Group();
    const matDeck = this._stdMat(0xeab308, 0.35, 0.25);
    const matBumper = this._stdMat(0x1e293b, 0.65, 0.35);
    const matTop = this._stdMat(0xcbd5e1, 0.45, 0.55);
    const matBeacon = this._stdMat(0xa3e635, 0.2, 0.1, 0x84cc16, 1.3);

    const chassis = new THREE.Mesh(new THREE.BoxGeometry(0.78, 0.20, 1.28), matDeck);
    chassis.position.y = 0.15;
    body.add(chassis);

    const topPlate = new THREE.Mesh(new THREE.BoxGeometry(0.66, 0.03, 1.08), matTop);
    topPlate.position.y = 0.265;
    body.add(topPlate);

    const bumperF = new THREE.Mesh(new THREE.BoxGeometry(0.80, 0.08, 0.08), matBumper);
    bumperF.position.set(0, 0.10, 0.64);
    const bumperB = bumperF.clone();
    bumperB.position.set(0, 0.10, -0.64);
    body.add(bumperF, bumperB);

    const beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.07, 12), matBeacon);
    beacon.position.set(0, 0.31, 0.48);
    body.add(beacon);

    return { body, limbs: null, baseHeight: 0.36 };
  }

  _getOrCreateEntity(key, cls, idLabel, x, y, z) {
    let ent = this._entities.get(key);
    if (ent) return ent;

    const root = new THREE.Group();
    root.userData.excludeFromDepth = true;

    const numId = typeof idLabel === 'number' ? idLabel : parseInt(String(idLabel).replace(/\D/g, ''), 10) || 1;
    const built =
      cls === 'robot'
        ? this._buildRobotAvatar()
        : cls === 'agv'
          ? this._buildAgvAvatar()
          : this._buildWorkerAvatar(numId);

    root.add(built.body);

    const ringColor = cls === 'robot' ? 0xf97316 : cls === 'agv' ? 0xa3e635 : 0x38bdf8;
    const ringMat = new THREE.MeshBasicMaterial({
      color: ringColor,
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    ringMat.userData.skipProjection = true;
    const ring = new THREE.Mesh(this._ringGeo, ringMat);
    ring.position.y = 0.025;
    const rScale = cls === 'agv' ? 1.55 : 1.0;
    ring.scale.set(rScale, 1, rScale);
    root.add(ring);

    const labelMat = new THREE.SpriteMaterial({
      map: this._getLabelTexture(cls, idLabel),
      transparent: true,
      depthTest: false,
      depthWrite: false
    });
    const label = new THREE.Sprite(labelMat);
    label.scale.set(1.15, 0.30, 1);
    label.position.y = built.baseHeight + 0.28;
    root.add(label);

    root.position.set(x, y, z);
    this.group.add(root);

    ent = {
      root,
      body: built.body,
      limbs: built.limbs,
      baseHeight: built.baseHeight,
      label,
      cls,
      prevX: x,
      prevZ: z,
      yaw: 0,
      walkPhase: (numId % 7) * 0.9,
      lastSeenMs: performance.now()
    };
    this._entities.set(key, ent);
    return ent;
  }

  update(dt, nowMs, streamMgr, fallbackBillboardTracks = []) {
    if (!this.visible) return;

    const items = [];
    const ext = this._externalTracks;
    if (ext?.frames?.length) {
      const masterTime = streamMgr.getMasterMp4Time?.();
      if (masterTime !== null && masterTime !== undefined) {
        const fps = ext.fps || 15;
        const fFloat = (masterTime % (ext.duration || 20)) * fps;
        const idx0 = Math.floor(fFloat) % ext.frames.length;
        const idx1 = (idx0 + 1) % ext.frames.length;
        const frac = fFloat - Math.floor(fFloat);
        const list0 = ext.frames[idx0] || [];
        const list1 = ext.frames[idx1] || [];
        const map1 = new Map(list1.map((o) => [o.id, o]));

        for (const o0 of list0) {
          const o1 = map1.get(o0.id);
          const x = o1 ? o0.p[0] + (o1.p[0] - o0.p[0]) * frac : o0.p[0];
          const y = o1 ? o0.p[1] + (o1.p[1] - o0.p[1]) * frac : o0.p[1];
          const z = o1 ? o0.p[2] + (o1.p[2] - o0.p[2]) * frac : o0.p[2];
          // NVIDIA Z-Up yaw -> Three.js Y-Up yaw (+Z_three forward)
          const gtYaw = (o0.yaw ?? 0) + Math.PI * 0.5;
          items.push({
            key: `${o0.cls}_${o0.id}`,
            id: o0.id,
            cls: o0.cls,
            x,
            y,
            z,
            height: o0.s?.[1] || 1.7,
            gtYaw
          });
        }
      }
    } else {
      // 无外部轨迹的场景（如 Blender 套组 / EPFL Terrace）：直接由视觉前景 3D 轨迹驱动 3D 工人替身
      fallbackBillboardTracks.forEach((tr, idx) => {
        if ((tr.hits ?? 0) < 2) return;
        items.push({
          key: `vis_${idx}`,
          id: idx + 1,
          cls: 'person',
          x: tr.x,
          y: tr.y,
          z: tr.z,
          height: tr.height || 1.7,
          gtYaw: null
        });
      });
    }

    const activeKeys = new Set();
    const safeDt = Math.min(Math.max(dt, 0.001), 0.1);

    for (const it of items) {
      activeKeys.add(it.key);
      const ent = this._getOrCreateEntity(it.key, it.cls, it.id, it.x, it.y, it.z);
      ent.root.visible = true;
      ent.lastSeenMs = nowMs;

      const vx = (it.x - ent.prevX) / safeDt;
      const vz = (it.z - ent.prevZ) / safeDt;
      const speed = Math.hypot(vx, vz);
      ent.prevX = it.x;
      ent.prevZ = it.z;

      ent.root.position.set(it.x, it.y, it.z);

      // 按真实身高微调等比缩放
      if (ent.cls !== 'agv' && ent.baseHeight > 0.5) {
        const sc = THREE.MathUtils.clamp(it.height / ent.baseHeight, 0.82, 1.18);
        ent.body.scale.setScalar(sc);
        ent.label.position.y = ent.baseHeight * sc + 0.28;
      }

      // 速度足够大时面向真实前进方向，静止时平滑对齐真值朝向
      let targetYaw = ent.yaw;
      if (speed > 0.15 && speed < 8.0) {
        targetYaw = Math.atan2(vx, vz);
      } else if (it.gtYaw !== null && it.gtYaw !== undefined) {
        targetYaw = it.gtYaw;
      }
      ent.yaw += normalizeAngleDiff(targetYaw - ent.yaw) * Math.min(1.0, safeDt * 10.0);
      ent.body.rotation.y = ent.yaw;

      // 行走骨骼步态动画
      if (ent.limbs) {
        const activeSpeed = speed < 8.0 ? Math.min(speed, 2.2) : 0;
        if (activeSpeed > 0.08) {
          ent.walkPhase += safeDt * (4.5 + activeSpeed * 3.8);
        }
        const amp = THREE.MathUtils.clamp(activeSpeed * 0.42, 0.0, 0.58);
        const swing = Math.sin(ent.walkPhase) * amp;
        ent.limbs.legL.rotation.x = swing;
        ent.limbs.legR.rotation.x = -swing;
        ent.limbs.armL.rotation.x = -swing * 0.75;
        ent.limbs.armR.rotation.x = swing * 0.75;
      }
    }

    for (const [k, ent] of this._entities.entries()) {
      if (!activeKeys.has(k)) {
        if (nowMs - ent.lastSeenMs > 300) {
          ent.root.visible = false;
        }
      }
    }
  }

  dispose() {
    this._clearEntities();
    this._ringGeo.dispose();
    for (const tex of this._labelTexCache.values()) tex.dispose();
    this._labelTexCache.clear();
  }
}
