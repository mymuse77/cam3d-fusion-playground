import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { solveCameraPnP } from '../src/utils/webCalibrator.js';

const dirtyDir = path.resolve(process.cwd(), 'public/demo-blender-dirty');

// 1. 验证 chess.mp4 棋盘格标定结果精度
const cal = JSON.parse(fs.readFileSync(path.join(dirtyDir, 'intrinsics_calibrated.json'), 'utf8'));
assert.ok(cal.rms < 0.35, `棋盘格标定 RMS 过高: ${cal.rms} px`);
assert.ok(cal.validFrames >= 15, `棋盘格有效帧数不足: ${cal.validFrames}`);

const fxRelErr = Math.abs(cal.intrinsics.fx - 866.6667) / 866.6667;
const fyRelErr = Math.abs(cal.intrinsics.fy - 866.6667) / 866.6667;
const k1AbsErr = Math.abs(cal.distortion.k1 - (-0.26));
assert.ok(fxRelErr < 0.015 && fyRelErr < 0.015, `焦距标定相对误差过大: fx=${fxRelErr}, fy=${fyRelErr}`);
assert.ok(k1AbsErr < 0.03, `径向畸变 k1 标定误差过大: ${k1AbsErr}`);
console.log(
  `[PASS] 棋盘格标定: RMS=${cal.rms} px (${cal.validFrames}帧), fx=${cal.intrinsics.fx}, cx=${cal.intrinsics.cx}, k1=${cal.distortion.k1}`
);

// 2. 验证使用实测内参 + 手抖噪声(±1.5px) + 第9点严重误点(+68px/-45px) 下的 Web & Python PnP 容错能力
const parkDirty = JSON.parse(fs.readFileSync(path.join(dirtyDir, 'park_cameras_dirty.json'), 'utf8'));
for (const camCfg of parkDirty.cameras) {
  const ptsFile = JSON.parse(fs.readFileSync(path.join(dirtyDir, `calib_points_${camCfg.id}.json`), 'utf8'));
  const res = solveCameraPnP({
    pts3D: ptsFile.pts_3d,
    pts2D: ptsFile.pts_2d,
    intrinsics: ptsFile.intrinsics,
    distortion: ptsFile.distortion,
    useRansac: true,
    ransacThreshPx: 4.0
  });

  assert.equal(res.inlierCount, 8, `${camCfg.id} RANSAC 应保留 8/9 内点并剔除第 9 号误点，实际: ${res.inlierCount}`);
  assert.ok(!res.inliers.includes(8), `${camCfg.id} 应准确剔除索引 8 的严重误点`);
  assert.ok(res.reprojectionErrorPx < 2.5, `${camCfg.id} 内点重投影误差过高: ${res.reprojectionErrorPx} px`);

  const gt = ptsFile.groundTruthPosition;
  const posErr = Math.hypot(res.position.x - gt[0], res.position.y - gt[1], res.position.z - gt[2]);
  assert.ok(posErr < 0.12, `${camCfg.id} 含噪声+滚转角位置还原偏差过大: ${posErr} m`);

  // 验证 Python solvePnPRansac + LM 输出
  assert.ok(camCfg.extrinsics?.matrixWorldRowMajor?.length === 16, `${camCfg.id} 应已写入 Python 外参`);
  assert.ok((camCfg.extrinsics?.reprojectionErrorPx ?? 99) < 2.5, `${camCfg.id} Python 重投影误差过高`);

  console.log(
    `[PASS] ${camCfg.id} 脏数据 PnP: 内点=${res.inlierCount}/9, Web误差=${res.reprojectionErrorPx} px, Python误差=${camCfg.extrinsics.reprojectionErrorPx} px, 位置误差=${posErr.toFixed(4)} m`
  );
}
