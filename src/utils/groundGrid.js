import * as THREE from 'three';
import { GROUND_HEIGHT_VERT, GROUND_HEIGHT_FRAG, GROUND_MAX_STEP, OCCUPANCY_NO_GROUND } from '../shaders/occupancyShaders.js';

const MAX_GRID_DIM = 256;
const MIN_CELL_SIZE = 0.08;
// 计入相机覆盖范围时，视线最远只取到该距离（米）：更远处的人在画面里太小，不值得占用网格分辨率
const MAX_COVERAGE_REACH = 35;

/**
 * 俯视地面网格：以可行走网格的世界包围盒（XZ）划分等距单元，并渲染每个单元的地面高度。
 * 占据度定位与自动色彩匹配共用同一张网格；行号随 Z 增大（第 0 行 = minZ）。
 */
export class GroundGrid {
  constructor() {
    this.heightTexture = null;
    this.origin = new THREE.Vector2();
    this.size = new THREE.Vector2();
    this.cols = 0;
    this.rows = 0;
    this.cellSize = 0;
    this.heights = null; // 与高度图同序的 CPU 副本，无地面处为 OCCUPANCY_NO_GROUND

    this._meshes = [];
    this._coverageCameras = [];
    this._dirty = false;
    this._target = null;
    this._lowTarget = null;
    this._scene = new THREE.Scene();
    this._scene.matrixWorldAutoUpdate = false;
    this._camera = new THREE.Camera();
    const makeMaterial = (depthSign, maxStep) =>
      new THREE.ShaderMaterial({
        vertexShader: GROUND_HEIGHT_VERT,
        fragmentShader: GROUND_HEIGHT_FRAG,
        uniforms: {
          uGridOrigin: { value: this.origin },
          uGridSize: { value: this.size },
          uYRange: { value: this._yRange },
          uDepthSign: { value: depthSign },
          uLowTex: { value: null },
          uMaxStep: { value: maxStep }
        },
        side: THREE.BackSide,
        depthTest: true,
        depthWrite: true
      });
    this._yRange = new THREE.Vector2();
    this._lowMaterial = makeMaterial(1, 0);
    this._highMaterial = makeMaterial(-1, GROUND_MAX_STEP);
    this._clearColor = new THREE.Color();
    this._box = new THREE.Box3();
    this._meshBox = new THREE.Box3();
    this._coverageBox = new THREE.Box3();
    this._tmpPoint = new THREE.Vector3();
    this._tmpOrigin = new THREE.Vector3();
  }

  setWalkableMeshes(meshes) {
    this._meshes = meshes;
    this._dirty = true;
  }

  /**
   * 设定相机集合：网格只覆盖它们的地面视野范围，避免巨大的地面网格把有限的分辨率摊薄
   */
  setCoverageCameras(cameras) {
    this._coverageCameras = cameras;
    this._dirty = true;
  }

  /**
   * 网格索引（可为小数，整数为单元中心）对应的世界 XZ
   */
  cellToWorld(col, row) {
    return {
      x: this.origin.x + (col + 0.5) * this.cellSize,
      z: this.origin.y + (row + 0.5) * this.cellSize
    };
  }

  sampleHeight(col, row) {
    const c = THREE.MathUtils.clamp(Math.round(col), 0, this.cols - 1);
    const r = THREE.MathUtils.clamp(Math.round(row), 0, this.rows - 1);
    return this.heights[r * this.cols + c];
  }

  /**
   * 按需（重新）渲染高度图；返回当前是否存在可用的网格
   */
  ensure(renderer) {
    if (this._dirty) {
      this._dirty = false;
      this._rebuild(renderer);
    }
    return !!this.heightTexture;
  }

  _rebuild(renderer) {
    this._box.makeEmpty();
    for (const mesh of this._meshes) {
      mesh.updateWorldMatrix(true, false);
      if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
      this._meshBox.copy(mesh.geometry.boundingBox).applyMatrix4(mesh.matrixWorld);
      this._box.union(this._meshBox);
    }
    if (this._box.isEmpty()) {
      this._disposeTarget();
      return;
    }
    this._clipToCoverage();

    const extentX = this._box.max.x - this._box.min.x;
    const extentZ = this._box.max.z - this._box.min.z;
    this.cellSize = Math.max(MIN_CELL_SIZE, Math.max(extentX, extentZ) / MAX_GRID_DIM);
    this.cols = Math.max(1, Math.ceil(extentX / this.cellSize));
    this.rows = Math.max(1, Math.ceil(extentZ / this.cellSize));
    this.origin.set(this._box.min.x, this._box.min.z);
    this.size.set(this.cols * this.cellSize, this.rows * this.cellSize);
    this._yRange.set(this._box.min.y - 0.01, this._box.max.y + 0.01);

    this._disposeTarget();
    const makeTarget = () =>
      new THREE.WebGLRenderTarget(this.cols, this.rows, {
        type: THREE.FloatType,
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        depthBuffer: true,
        generateMipmaps: false
      });
    this._lowTarget = makeTarget();
    this._target = makeTarget();
    this.heightTexture = this._target.texture;

    this._scene.clear();
    const proxies = this._meshes.map((mesh) => {
      const proxy = new THREE.Mesh(mesh.geometry, this._lowMaterial);
      proxy.matrixAutoUpdate = false;
      proxy.matrixWorldAutoUpdate = false;
      proxy.matrix.copy(mesh.matrixWorld);
      proxy.matrixWorld.copy(mesh.matrixWorld);
      proxy.frustumCulled = false;
      this._scene.add(proxy);
      return proxy;
    });

    const prevTarget = renderer.getRenderTarget();
    const prevAutoClear = renderer.autoClear;
    renderer.getClearColor(this._clearColor);
    const prevClearAlpha = renderer.getClearAlpha();
    renderer.autoClear = true;
    renderer.setClearColor(new THREE.Color(OCCUPANCY_NO_GROUND, 0, 0), 1);

    renderer.setRenderTarget(this._lowTarget);
    renderer.render(this._scene, this._camera);

    this._highMaterial.uniforms.uLowTex.value = this._lowTarget.texture;
    proxies.forEach((proxy) => (proxy.material = this._highMaterial));
    renderer.setRenderTarget(this._target);
    renderer.render(this._scene, this._camera);

    const pixels = new Float32Array(this.cols * this.rows * 4);
    renderer.readRenderTargetPixels(this._target, 0, 0, this.cols, this.rows, pixels);
    this.heights = new Float32Array(this.cols * this.rows);
    for (let i = 0; i < this.heights.length; i++) this.heights[i] = pixels[i * 4];

    renderer.setRenderTarget(prevTarget);
    renderer.autoClear = prevAutoClear;
    renderer.setClearColor(this._clearColor, prevClearAlpha);
    this._scene.clear();
    this._lowTarget.dispose();
    this._lowTarget = null;
  }

  /**
   * 把包围盒（XZ）收缩到各相机视锥四条边缘视线与地面的交点（最远取 MAX_COVERAGE_REACH）并外扩 1 米
   */
  _clipToCoverage() {
    if (!this._coverageCameras.length) return;
    const groundY = this._box.min.y;
    const cover = this._coverageBox.makeEmpty();
    for (const cam of this._coverageCameras) {
      cam.updateMatrixWorld(true);
      cam.getWorldPosition(this._tmpOrigin);
      for (const [nx, ny] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
        const dir = this._tmpPoint.set(nx, ny, 1).unproject(cam).sub(this._tmpOrigin).normalize();
        const reach = dir.y < -1e-3 ? Math.min((groundY - this._tmpOrigin.y) / dir.y, MAX_COVERAGE_REACH) : MAX_COVERAGE_REACH;
        cover.expandByPoint(new THREE.Vector3(
          this._tmpOrigin.x + dir.x * reach,
          groundY,
          this._tmpOrigin.z + dir.z * reach
        ));
      }
    }
    cover.min.x -= 1;
    cover.min.z -= 1;
    cover.max.x += 1;
    cover.max.z += 1;
    const minX = Math.max(this._box.min.x, cover.min.x);
    const maxX = Math.min(this._box.max.x, cover.max.x);
    const minZ = Math.max(this._box.min.z, cover.min.z);
    const maxZ = Math.min(this._box.max.z, cover.max.z);
    if (minX >= maxX || minZ >= maxZ) return;
    this._box.min.x = minX;
    this._box.max.x = maxX;
    this._box.min.z = minZ;
    this._box.max.z = maxZ;
  }

  _disposeTarget() {
    if (this._target) this._target.dispose();
    if (this._lowTarget) this._lowTarget.dispose();
    this._lowTarget = null;
    this._target = null;
    this.heightTexture = null;
    this.heights = null;
  }

  dispose() {
    this._disposeTarget();
    this._lowMaterial.dispose();
    this._highMaterial.dispose();
  }
}
