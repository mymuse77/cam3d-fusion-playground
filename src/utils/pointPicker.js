import * as THREE from 'three';

/**
 * 3D 控制点采集工具：按住 Shift + 左键点击模型打红点，一键复制为 Python solvePnP 数组
 */
export class PointPicker {
  constructor(camera, domElement, targetRoot, onChange) {
    this.camera = camera;
    this.domElement = domElement;
    this.targetRoot = targetRoot;
    this.onChange = onChange;

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.points = [];

    this.markerGroup = new THREE.Group();
    this.markerGeo = new THREE.SphereGeometry(0.18, 16, 16);
    this.markerMat = new THREE.MeshBasicMaterial({ color: 0xff2a2a });

    this._onPointerDown = this._onPointerDown.bind(this);
    this.domElement.addEventListener('pointerdown', this._onPointerDown);
  }

  _onPointerDown(event) {
    if (!event.shiftKey || event.button !== 0) return;
    const rect = this.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.pointer, this.camera);
    const meshes = [];
    this.targetRoot.traverse((obj) => {
      if (obj.isMesh) meshes.push(obj);
    });

    const hits = this.raycaster.intersectObjects(meshes, false);
    if (hits.length === 0) return;

    const p = hits[0].point;
    const pt = { x: +p.x.toFixed(4), y: +p.y.toFixed(4), z: +p.z.toFixed(4) };
    this.points.push(pt);

    const marker = new THREE.Mesh(this.markerGeo, this.markerMat);
    marker.position.set(pt.x, pt.y, pt.z);
    this.markerGroup.add(marker);
    this.onChange?.(this.points);
  }

  undo() {
    if (!this.points.length) return;
    this.points.pop();
    this.markerGroup.remove(this.markerGroup.children[this.markerGroup.children.length - 1]);
    this.onChange?.(this.points);
  }

  clear() {
    this.points = [];
    this.markerGroup.clear();
    this.onChange?.(this.points);
  }

  toPythonString() {
    const lines = this.points.map((p, i) => `    [${p.x}, ${p.y}, ${p.z}],  # 点 ${i + 1}`);
    return `pts_3d = [
${lines.join('\n')}
]`;
  }
}
