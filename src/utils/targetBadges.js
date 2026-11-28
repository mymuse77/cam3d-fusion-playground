import * as THREE from 'three';

/**
 * 目标身份徽标：脚底彩色身份光环 + 顶部类型标签，区分普通工人 / 人形机器人 / AGV。
 * 与具体前景呈现方式（立牌 / 自由视点融合）解耦，只消费三维轨迹。
 */
export class TargetBadges {
  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'TargetBadges';

    this._ringGeo = new THREE.RingGeometry(0.30, 0.40, 28);
    this._ringGeo.rotateX(-Math.PI / 2);
    this._ringPool = [];
    this._labelPool = [];
    this._labelTexCache = new Map();
  }

  _getLabelTexture(cls, id) {
    const key = `${cls}_${id}`;
    const cached = this._labelTexCache.get(key);
    if (cached) return cached;

    const canvas = document.createElement('canvas');
    canvas.width = 220;
    canvas.height = 56;
    const ctx = canvas.getContext('2d');

    const style =
      cls === 'robot'
        ? { bg: 'rgba(234, 88, 12, 0.90)', border: '#fdba74', text: `机器人 #${id}` }
        : cls === 'agv'
          ? { bg: 'rgba(101, 163, 13, 0.90)', border: '#d9f99d', text: `AGV #${id}` }
          : cls === 'target'
            ? { bg: 'rgba(71, 85, 105, 0.90)', border: '#cbd5e1', text: `目标 #${id}` }
            : { bg: 'rgba(2, 132, 199, 0.90)', border: '#7dd3fc', text: `工人 #${id}` };

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
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    this._labelTexCache.set(key, tex);
    return tex;
  }

  _acquireRing(index) {
    while (this._ringPool.length <= index) {
      const mat = new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.85,
        side: THREE.DoubleSide,
        depthWrite: false
      });
      const ring = new THREE.Mesh(this._ringGeo, mat);
      ring.frustumCulled = false;
      ring.visible = false;
      this.group.add(ring);
      this._ringPool.push(ring);
    }
    return this._ringPool[index];
  }

  _acquireLabel(index) {
    while (this._labelPool.length <= index) {
      const mat = new THREE.SpriteMaterial({
        transparent: true,
        depthTest: false,
        depthWrite: false
      });
      const sprite = new THREE.Sprite(mat);
      sprite.frustumCulled = false;
      sprite.visible = false;
      this.group.add(sprite);
      this._labelPool.push(sprite);
    }
    return this._labelPool[index];
  }

  /**
   * @param {Array} tracks 三维轨迹 [{id, cls, x, y, z, height, hits}]
   * @param {{rings: boolean}} options rings=false 时只显示标签（如自由视点融合已有接触阴影标记落脚点），
   *   且标签参与深度测试：目标被墙体 / 地面挡住时承载面片不可见，标签随之隐藏
   */
  update(tracks, { rings }) {
    let badgeIdx = 0;
    for (const tr of tracks) {
      if (!tr.cls || tr.id === undefined || tr.hits < 2) continue;

      if (rings) {
        const ring = this._acquireRing(badgeIdx);
        ring.visible = true;
        ring.position.set(tr.x, tr.y + 0.03, tr.z);
        const ringColor = tr.cls === 'robot' ? 0xf97316 : tr.cls === 'agv' ? 0xa3e635 : 0x38bdf8;
        ring.material.color.setHex(ringColor);
        const rScale = tr.cls === 'agv' ? 1.55 : 1.0;
        ring.scale.set(rScale, 1, rScale);
      }

      const label = this._acquireLabel(badgeIdx);
      label.visible = true;
      label.material.map = this._getLabelTexture(tr.cls, tr.id);
      label.material.depthTest = !rings;
      label.material.needsUpdate = true;
      label.position.set(tr.x, tr.y + tr.height + 0.28, tr.z);
      label.scale.set(1.15, 0.30, 1);
      badgeIdx++;
    }

    const ringCount = rings ? badgeIdx : 0;
    for (let i = ringCount; i < this._ringPool.length; i++) this._ringPool[i].visible = false;
    for (let i = badgeIdx; i < this._labelPool.length; i++) this._labelPool[i].visible = false;
  }

  clear() {
    this._ringPool.forEach((r) => (r.visible = false));
    this._labelPool.forEach((l) => (l.visible = false));
  }

  dispose() {
    this._ringGeo.dispose();
    this._ringPool.forEach((r) => r.material.dispose());
    this._labelPool.forEach((l) => l.material.dispose());
    this._labelTexCache.forEach((t) => t.dispose());
    this._labelTexCache.clear();
    this._ringPool = [];
    this._labelPool = [];
  }
}
