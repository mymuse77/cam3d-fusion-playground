import fs from 'node:fs';
import path from 'node:path';
import { defineConfig } from 'vite';

/**
 * 将 Three.js 自带的 Draco 解码器静态资源自动拷贝至 public/draco/，确保内网/离线可用
 */
function localDracoPlugin() {
  const files = ['draco_decoder.js', 'draco_decoder.wasm', 'draco_wasm_wrapper.js'];

  const copyDracoAssets = (rootDir) => {
    const gltfDir = path.resolve(rootDir, 'node_modules/three/examples/jsm/libs/draco/gltf');
    const baseDir = path.resolve(rootDir, 'node_modules/three/examples/jsm/libs/draco');
    const srcDir = fs.existsSync(gltfDir) ? gltfDir : baseDir;
    if (!fs.existsSync(srcDir)) return;

    const outDir = path.resolve(rootDir, 'public/draco');
    fs.mkdirSync(outDir, { recursive: true });

    for (const file of files) {
      const src = path.join(srcDir, file);
      const dst = path.join(outDir, file);
      if (fs.existsSync(src)) {
        fs.copyFileSync(src, dst);
      }
    }
  };

  return {
    name: 'local-draco-assets',
    configResolved(config) {
      copyDracoAssets(config.root);
    },
    buildStart() {
      copyDracoAssets(process.cwd());
    }
  };
}

export default defineConfig({
  server: {
    host: '0.0.0.0',
    port: 5173
  },
  plugins: [localDracoPlugin()]
});
