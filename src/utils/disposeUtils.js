/**
 * 递归彻底释放 Object3D 及其子节点占用的 GPU 几何体、材质与贴图显存
 */
export function disposeObject3D(root) {
  if (!root) return;
  const texKeys = [
    'map', 'lightMap', 'bumpMap', 'normalMap', 'specularMap', 'envMap',
    'alphaMap', 'aoMap', 'displacementMap', 'emissiveMap', 'metalnessMap', 'roughnessMap'
  ];

  root.traverse((obj) => {
    if (obj.geometry) {
      obj.geometry.dispose();
    }
    if (obj.material) {
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      for (const mat of mats) {
        for (const key of texKeys) {
          if (mat[key] && typeof mat[key].dispose === 'function') {
            mat[key].dispose();
          }
        }
        mat.dispose();
      }
    }
  });
}
