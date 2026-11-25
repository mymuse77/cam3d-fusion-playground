import * as THREE from 'three';
import { disposeObject3D } from './disposeUtils.js';

/**
 * 3D 多边形电子围栏：支持射线法越界判定与告警变色
 */
export class Geofence3D {
  /**
   * @param {Array<[number, number]>} polygonXZ 围栏地面顶点 [[x, z], ...]
   * @param {number} height 围栏光墙高度
   */
  constructor(polygonXZ, height = 2.6) {
    this.polygon = polygonXZ;
    this.group = new THREE.Group();
    this.isAlarm = false;

    const positions = [];
    const linePts = [];
    for (let i = 0; i < polygonXZ.length; i++) {
      const [x1, z1] = polygonXZ[i];
      const [x2, z2] = polygonXZ[(i + 1) % polygonXZ.length];
      linePts.push(new THREE.Vector3(x1, 0.08, z1));

      // 每条边构成两个三角形光墙
      positions.push(
        x1, 0, z1,  x2, 0, z2,  x2, height, z2,
        x1, 0, z1,  x2, height, z2,  x1, height, z1
      );
    }
    linePts.push(linePts[0].clone());

    const wallGeo = new THREE.BufferGeometry();
    wallGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    this.wallMat = new THREE.MeshBasicMaterial({
      color: 0x06b6d4,
      transparent: true,
      opacity: 0.22,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    this.group.add(new THREE.Mesh(wallGeo, this.wallMat));

    const borderGeo = new THREE.BufferGeometry().setFromPoints(linePts);
    this.borderMat = new THREE.LineBasicMaterial({ color: 0x22d3ee });
    this.group.add(new THREE.Line(borderGeo, this.borderMat));
  }

  /**
   * 2D 射线法判断 (x, z) 是否落入多边形围栏内部
   */
  contains(x, z) {
    let inside = false;
    const pts = this.polygon;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, zi] = pts[i];
      const [xj, zj] = pts[j];
      const intersect = ((zi > z) !== (zj > z)) && (x < ((xj - xi) * (z - zi)) / (zj - zi + 1e-9) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }

  setAlarmState(alarm) {
    if (this.isAlarm === alarm) return;
    this.isAlarm = alarm;
    this.wallMat.color.setHex(alarm ? 0xef4444 : 0x06b6d4);
    this.wallMat.opacity = alarm ? 0.36 : 0.22;
    this.borderMat.color.setHex(alarm ? 0xfca5a5 : 0x22d3ee);
  }
}

/**
 * AI 3D 追踪目标可视化管理器：圆柱标记 + 头顶悬浮 ID 标签 + 历史运动尾迹
 */
export class TargetVisualManager {
  constructor(parentGroup, maxTrailPoints = 45) {
    this.parentGroup = parentGroup;
    this.maxTrailPoints = maxTrailPoints;
    this.items = new Map(); // globalId -> { group, mesh, sprite, labelTex, trailLine, history, isAlarm }
  }

  _drawLabelCanvas(ctx, text, isAlarm) {
    ctx.clearRect(0, 0, 256, 64);
    ctx.fillStyle = isAlarm ? 'rgba(220, 38, 38, 0.9)' : 'rgba(15, 23, 42, 0.85)';
    ctx.beginPath();
    ctx.roundRect(8, 8, 240, 48, 10);
    ctx.fill();
    ctx.strokeStyle = isAlarm ? '#fca5a5' : '#4ade80';
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 22px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 128, 32);
  }

  updateTargets(tracks, geofence) {
    const activeIds = new Set(tracks.map((t) => t.globalId));
    const alarmIds = [];

    for (const t of tracks) {
      const inFence = geofence && geofence.group.visible && geofence.contains(t.x, t.z);
      if (inFence) alarmIds.push(t.globalId);

      let entry = this.items.get(t.globalId);
      if (!entry) {
        const group = new THREE.Group();
        const mesh = new THREE.Mesh(
          new THREE.CylinderGeometry(0.38, 0.38, 1.7, 16),
          new THREE.MeshStandardMaterial({ color: 0x00e676, roughness: 0.4 })
        );
        mesh.position.y = 0.85;
        group.add(mesh);

        // 头顶悬浮 ID 标签
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');
        const labelTex = new THREE.CanvasTexture(canvas);
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: labelTex, transparent: true }));
        sprite.position.set(0, 2.35, 0);
        sprite.scale.set(2.8, 0.7, 1);
        group.add(sprite);

        // 地面历史运动尾迹
        const trailGeo = new THREE.BufferGeometry();
        const trailLine = new THREE.Line(
          trailGeo,
          new THREE.LineBasicMaterial({ color: 0x4ade80, transparent: true, opacity: 0.85 })
        );

        this.parentGroup.add(group);
        this.parentGroup.add(trailLine);

        entry = { group, mesh, sprite, ctx, labelTex, trailLine, history: [], isAlarm: null };
        this.items.set(t.globalId, entry);
      }

      // 更新位置与告警状态
      entry.group.position.set(t.x, t.y, t.z);
      if (entry.isAlarm !== inFence) {
        entry.isAlarm = inFence;
        entry.mesh.material.color.setHex(inFence ? 0xef4444 : 0x00e676);
        entry.trailLine.material.color.setHex(inFence ? 0xf87171 : 0x4ade80);
        const labelText = inFence ? `#${t.globalId} 越界告警!` : `#${t.globalId} ${t.cls.toUpperCase()}`;
        this._drawLabelCanvas(entry.ctx, labelText, inFence);
        entry.labelTex.needsUpdate = true;
      }

      // 追加历史运动轨迹点
      const last = entry.history[entry.history.length - 1];
      if (!last || Math.hypot(last.x - t.x, last.z - t.z) > 0.15) {
        entry.history.push(new THREE.Vector3(t.x, t.y + 0.12, t.z));
        if (entry.history.length > this.maxTrailPoints) entry.history.shift();
        entry.trailLine.geometry.setFromPoints(entry.history);
      }
    }

    // 清理超时消失目标并释放显存
    for (const [gid, entry] of this.items.entries()) {
      if (!activeIds.has(gid)) {
        this.parentGroup.remove(entry.group);
        this.parentGroup.remove(entry.trailLine);
        disposeObject3D(entry.group);
        disposeObject3D(entry.trailLine);
        this.items.delete(gid);
      }
    }

    return alarmIds;
  }
}
