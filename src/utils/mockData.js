import * as THREE from 'three';
import { projectCameraPointToPixel } from './bboxProjector.js';

/**
 * 生成带 OSD 水印和扫描网格的动态测试画布（无本地视频时开箱即用）
 */
export function createTestCanvasTexture(label, accentColor = '#00e5ff') {
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 360;
  const ctx = canvas.getContext('2d');
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;

  return {
    texture,
    update(time) {
      ctx.fillStyle = '#0d1b2a';
      ctx.fillRect(0, 0, 640, 360);

      ctx.strokeStyle = '#1b4965';
      ctx.lineWidth = 2;
      for (let x = 0; x <= 640; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 360); ctx.stroke(); }
      for (let y = 0; y <= 360; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(640, y); ctx.stroke(); }

      const scanX = ((time * 100) % 720) - 40;
      ctx.fillStyle = 'rgba(0, 229, 255, 0.25)';
      ctx.fillRect(scanX, 0, 24, 360);

      ctx.fillStyle = accentColor;
      ctx.font = 'bold 26px monospace';
      ctx.fillText(label, 20, 46);
      ctx.strokeStyle = accentColor;
      ctx.lineWidth = 6;
      ctx.strokeRect(3, 3, 634, 354);

      texture.needsUpdate = true;
    }
  };
}

/**
 * 模拟 YOLO 多摄像头检测器：让虚拟行人穿越 1、2 号枪机重叠区及中央电子围栏
 */
export function generateMockYoloDetections(timeSec, cameras) {
  const truePos = new THREE.Vector3(Math.sin(timeSec * 0.45) * 7.5, 0, -1.0 + Math.cos(timeSec * 0.45) * 2.2);
  const results = new Map();

  for (const item of cameras) {
    if (item.type !== 'fixed') continue;
    const viewPos = truePos.clone().applyMatrix4(item.camera.matrixWorldInverse);
    if (viewPos.z >= -0.5) continue;

    const pixel = projectCameraPointToPixel(viewPos, item.intrinsics, item.distortion);
    if (pixel) {
      const u = pixel.u + (Math.random() - 0.5) * 6;
      const v = pixel.v + (Math.random() - 0.5) * 6;
      results.set(item.id, [
        { trackId: item.id === 'cam_fixed_01' ? 101 : 202, cls: 'person', bbox: [u - 30, v - 120, u + 30, v] }
      ]);
    }
  }
  return results;
}
