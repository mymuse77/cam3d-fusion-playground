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
 * 模拟 YOLO 多摄像头检测器（双目标交叉 + 三维斜坡/平台巡检）：
 * - 目标 A：往返穿越 1、2 号枪机重叠区及中央电子围栏
 * - 目标 B：沿东侧斜坡走上 0.8m 巡检平台再走下，并与目标 A 在重叠区交叉
 */
export function generateMockYoloDetections(timeSec, cameras) {
  const posA = new THREE.Vector3(
    Math.sin(timeSec * 0.45) * 6.5,
    0,
    -1.0 + Math.cos(timeSec * 0.45) * 2.2
  );

  // 目标 B 在 [z = 3.5 -> -4.5] 往返巡检：经过斜坡 (z∈[0.5,3.5]) 与平台 (z∈[-5.5,0.5], y=0.8)
  const phaseB = (Math.sin(timeSec * 0.35) + 1) * 0.5; // 0 ~ 1
  const zB = 3.5 - phaseB * 8.0; // 3.5 -> -4.5
  let yB = 0;
  if (zB <= 0.5) {
    yB = 0.8;
  } else if (zB < 3.5) {
    yB = 0.8 * ((3.5 - zB) / 3.0);
  }
  const posB = new THREE.Vector3(7.0 + Math.sin(timeSec * 0.7) * 0.4, yB, zB);

  const actors = [
    { pos: posA, cls: 'person', idOffset: 1 },
    { pos: posB, cls: 'person', idOffset: 2 }
  ];

  const results = new Map();

  for (const item of cameras) {
    if (item.type !== 'fixed') continue;
    const camSeed = item.id === 'cam_fixed_01' ? 100 : 200;
    const dets = [];

    for (const actor of actors) {
      const viewPos = actor.pos.clone().applyMatrix4(item.camera.matrixWorldInverse);
      const pixel = projectCameraPointToPixel(viewPos, item.intrinsics, item.distortion);
      if (pixel) {
        const u = pixel.u + (Math.random() - 0.5) * 5;
        const v = pixel.v + (Math.random() - 0.5) * 5;
        dets.push({
          trackId: camSeed + actor.idOffset,
          cls: actor.cls,
          bbox: [u - 28, v - 115, u + 28, v]
        });
      }
    }

    if (dets.length > 0) {
      results.set(item.id, dets);
    }
  }

  return results;
}
