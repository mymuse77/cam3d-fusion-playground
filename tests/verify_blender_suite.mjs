import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { solveCameraPnP } from '../src/utils/webCalibrator.js';

const demoDir = path.resolve(process.cwd(), 'public/demo-blender');
const glbStat = fs.statSync(path.join(demoDir, 'blender_park.glb'));
assert.ok(glbStat.size > 1000, 'blender_park.glb 应存在且非空');

const parkCfg = JSON.parse(fs.readFileSync(path.join(demoDir, 'park_cameras_blender.json'), 'utf8'));

for (const camCfg of parkCfg.cameras) {
  const calibPts = JSON.parse(fs.readFileSync(path.join(demoDir, `calib_points_${camCfg.id}.json`), 'utf8'));
  const res = solveCameraPnP({
    pts3D: calibPts.pts_3d,
    pts2D: calibPts.pts_2d,
    intrinsics: calibPts.intrinsics,
    distortion: calibPts.distortion,
    useRansac: true
  });

  assert.equal(res.inlierCount, calibPts.pts_3d.length, `${camCfg.id} 应全为内点`);
  assert.ok(res.reprojectionErrorPx < 0.05, `${camCfg.id} 重投影误差过大: ${res.reprojectionErrorPx} px`);

  const gt = calibPts.groundTruthPosition;
  const posErr = Math.hypot(res.position.x - gt[0], res.position.y - gt[1], res.position.z - gt[2]);
  assert.ok(posErr < 0.01, `${camCfg.id} 相机位置与 Blender 真值偏差过大: ${posErr} m`);

  const pyMat = camCfg.extrinsics?.matrixWorldRowMajor;
  assert.ok(Array.isArray(pyMat) && pyMat.length === 16, `${camCfg.id} 应已由 Python calib_toolkit.py 写入外参`);
  let maxMatDiff = 0;
  for (let i = 0; i < 16; i++) {
    maxMatDiff = Math.max(maxMatDiff, Math.abs(pyMat[i] - res.matrixWorldRowMajor[i]));
  }
  assert.ok(maxMatDiff < 1e-3, `${camCfg.id} 前端与 Python 外参矩阵差异过大: ${maxMatDiff}`);

  console.log(
    `[PASS] ${camCfg.id}: Web PnP 位置误差=${posErr.toExponential(2)} m, 重投影误差=${res.reprojectionErrorPx} px, Web/Python 矩阵最大差=${maxMatDiff.toExponential(2)}`
  );
}
