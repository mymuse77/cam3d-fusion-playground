import * as THREE from 'three';

/**
 * 迭代去畸变（像素坐标 -> OpenCV 归一化相机坐标）
 */
function undistortPixel(u, v, intr, dist = {}) {
  const xDist = (u - intr.cx) / intr.fx;
  const yDist = (v - intr.cy) / intr.fy;
  const k1 = dist.k1 || 0;
  const k2 = dist.k2 || 0;
  const p1 = dist.p1 || 0;
  const p2 = dist.p2 || 0;
  if (!k1 && !k2 && !p1 && !p2) return [xDist, yDist];

  let x = xDist;
  let y = yDist;
  for (let i = 0; i < 8; i++) {
    const r2 = x * x + y * y;
    const radial = 1 + k1 * r2 + k2 * r2 * r2;
    const dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
    const dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
    x = (xDist - dx) / radial;
    y = (yDist - dy) / radial;
  }
  return [x, y];
}

/**
 * OpenCV 相机坐标系点 P_cv -> 含 Brown-Conrady 畸变的像素坐标 [u, v]
 */
function projectCVPointToPixel(pcv, intr, dist = {}) {
  const z = Math.max(pcv[2], 1e-4);
  const x = pcv[0] / z;
  const y = pcv[1] / z;
  const r2 = x * x + y * y;
  const r4 = r2 * r2;

  const k1 = dist.k1 || 0;
  const k2 = dist.k2 || 0;
  const p1 = dist.p1 || 0;
  const p2 = dist.p2 || 0;
  const radial = 1 + k1 * r2 + k2 * r4;

  const xDist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
  const yDist = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
  return [xDist * intr.fx + intr.cx, yDist * intr.fy + intr.cy];
}

/**
 * 实对称矩阵 Jacobi 特征值分解（用于 12x12 DLT 最小奇异向量求解）
 */
function symmetricJacobiSmallestEigenvector(M, n) {
  const A = M.map((row) => Float64Array.from(row));
  const V = Array.from({ length: n }, (_, i) => {
    const r = new Float64Array(n);
    r[i] = 1;
    return r;
  });

  for (let iter = 0; iter < 80; iter++) {
    let maxVal = 0;
    let p = 0;
    let q = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const val = Math.abs(A[i][j]);
        if (val > maxVal) {
          maxVal = val;
          p = i;
          q = j;
        }
      }
    }
    if (maxVal < 1e-12) break;

    const app = A[p][p];
    const aqq = A[q][q];
    const apq = A[p][q];
    const tau = (aqq - app) / (2 * apq);
    const t = tau >= 0 ? 1 / (tau + Math.sqrt(1 + tau * tau)) : -1 / (-tau + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    A[p][p] = app - t * apq;
    A[q][q] = aqq + t * apq;
    A[p][q] = 0;
    A[q][p] = 0;

    for (let k = 0; k < n; k++) {
      if (k !== p && k !== q) {
        const akp = A[k][p];
        const akq = A[k][q];
        A[k][p] = c * akp - s * akq;
        A[p][k] = A[k][p];
        A[k][q] = s * akp + c * akq;
        A[q][k] = A[k][q];
      }
      const vkp = V[k][p];
      const vkq = V[k][q];
      V[k][p] = c * vkp - s * vkq;
      V[k][q] = s * vkp + c * vkq;
    }
  }

  let minIdx = 0;
  let minEig = A[0][0];
  for (let i = 1; i < n; i++) {
    if (A[i][i] < minEig) {
      minEig = A[i][i];
      minIdx = i;
    }
  }

  const vec = new Float64Array(n);
  for (let i = 0; i < n; i++) vec[i] = V[i][minIdx];
  return vec;
}

function mat3Mul(A, B) {
  return [
    [
      A[0][0] * B[0][0] + A[0][1] * B[1][0] + A[0][2] * B[2][0],
      A[0][0] * B[0][1] + A[0][1] * B[1][1] + A[0][2] * B[2][1],
      A[0][0] * B[0][2] + A[0][1] * B[1][2] + A[0][2] * B[2][2]
    ],
    [
      A[1][0] * B[0][0] + A[1][1] * B[1][0] + A[1][2] * B[2][0],
      A[1][0] * B[0][1] + A[1][1] * B[1][1] + A[1][2] * B[2][1],
      A[1][0] * B[0][2] + A[1][1] * B[1][2] + A[1][2] * B[2][2]
    ],
    [
      A[2][0] * B[0][0] + A[2][1] * B[1][0] + A[2][2] * B[2][0],
      A[2][0] * B[0][1] + A[2][1] * B[1][1] + A[2][2] * B[2][1],
      A[2][0] * B[0][2] + A[2][1] * B[1][2] + A[2][2] * B[2][2]
    ]
  ];
}

function mat3Vec(R, v) {
  return [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]
  ];
}

function mat3Det(M) {
  return (
    M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
    M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
    M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
  );
}

function mat3InvTranspose(M) {
  const det = mat3Det(M);
  if (Math.abs(det) < 1e-12) return null;
  const invDet = 1 / det;
  // M^-T 的元素 (r, c) 等于 M 的代数余子式 (r, c) / det
  return [
    [
      (M[1][1] * M[2][2] - M[1][2] * M[2][1]) * invDet,
      -(M[1][0] * M[2][2] - M[1][2] * M[2][0]) * invDet,
      (M[1][0] * M[2][1] - M[1][1] * M[2][0]) * invDet
    ],
    [
      -(M[0][1] * M[2][2] - M[0][2] * M[2][1]) * invDet,
      (M[0][0] * M[2][2] - M[0][2] * M[2][0]) * invDet,
      -(M[0][0] * M[2][1] - M[0][1] * M[2][0]) * invDet
    ],
    [
      (M[0][1] * M[1][2] - M[0][2] * M[1][1]) * invDet,
      -(M[0][0] * M[1][2] - M[0][2] * M[1][0]) * invDet,
      (M[0][0] * M[1][1] - M[0][1] * M[1][0]) * invDet
    ]
  ];
}

function rodrigues(wx, wy, wz) {
  const theta = Math.hypot(wx, wy, wz);
  if (theta < 1e-12) {
    return [
      [1, -wz, wy],
      [wz, 1, -wx],
      [-wy, wx, 1]
    ];
  }
  const kx = wx / theta;
  const ky = wy / theta;
  const kz = wz / theta;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const v = 1 - c;
  return [
    [kx * kx * v + c, kx * ky * v - kz * s, kx * kz * v + ky * s],
    [ky * kx * v + kz * s, ky * ky * v + c, ky * kz * v - kx * s],
    [kz * kx * v - ky * s, kz * ky * v + kx * s, kz * kz * v + c]
  ];
}

function solveLinear6x6(A, b) {
  const M = A.map((r, i) => [...r, b[i]]);
  for (let col = 0; col < 6; col++) {
    let pivot = col;
    for (let r = col + 1; r < 6; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    [M[col], M[pivot]] = [M[pivot], M[col]];
    const diag = M[col][col];
    if (Math.abs(diag) < 1e-12) return null;
    for (let j = col; j <= 6; j++) M[col][j] /= diag;
    for (let r = 0; r < 6; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      for (let j = col; j <= 6; j++) M[r][j] -= factor * M[col][j];
    }
  }
  return M.map((r) => r[6]);
}

/**
 * 归一化 3D DLT 求解 OpenCV 相机初始外参 (R_cv, t_cv)
 */
function solveDLT(pts3D, norm2D, indices) {
  const m = indices.length;
  const mu = [0, 0, 0];
  for (const idx of indices) {
    mu[0] += pts3D[idx][0];
    mu[1] += pts3D[idx][1];
    mu[2] += pts3D[idx][2];
  }
  mu[0] /= m;
  mu[1] /= m;
  mu[2] /= m;

  let meanD = 0;
  for (const idx of indices) {
    meanD += Math.hypot(pts3D[idx][0] - mu[0], pts3D[idx][1] - mu[1], pts3D[idx][2] - mu[2]);
  }
  meanD /= m;
  const scale = Math.sqrt(3) / Math.max(meanD, 1e-6);

  const AtA = Array.from({ length: 12 }, () => new Float64Array(12));
  const addRow = (row) => {
    for (let i = 0; i < 12; i++) {
      if (row[i] === 0) continue;
      for (let j = i; j < 12; j++) {
        AtA[i][j] += row[i] * row[j];
      }
    }
  };

  for (const idx of indices) {
    const X = (pts3D[idx][0] - mu[0]) * scale;
    const Y = (pts3D[idx][1] - mu[1]) * scale;
    const Z = (pts3D[idx][2] - mu[2]) * scale;
    const [u, v] = norm2D[idx];
    addRow([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u]);
    addRow([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]);
  }
  for (let i = 0; i < 12; i++) {
    for (let j = 0; j < i; j++) AtA[i][j] = AtA[j][i];
  }

  const p = symmetricJacobiSmallestEigenvector(AtA, 12);
  let Mraw = [
    [p[0] * scale, p[1] * scale, p[2] * scale],
    [p[4] * scale, p[5] * scale, p[6] * scale],
    [p[8] * scale, p[9] * scale, p[10] * scale]
  ];
  let traw = [
    p[3] - (Mraw[0][0] * mu[0] + Mraw[0][1] * mu[1] + Mraw[0][2] * mu[2]),
    p[7] - (Mraw[1][0] * mu[0] + Mraw[1][1] * mu[1] + Mraw[1][2] * mu[2]),
    p[11] - (Mraw[2][0] * mu[0] + Mraw[2][1] * mu[1] + Mraw[2][2] * mu[2])
  ];

  // 保证平均深度为正
  let sumZ = 0;
  for (const idx of indices) {
    const pt = pts3D[idx];
    sumZ += Mraw[2][0] * pt[0] + Mraw[2][1] * pt[1] + Mraw[2][2] * pt[2] + traw[2];
  }
  if (sumZ < 0) {
    Mraw = Mraw.map((r) => r.map((v) => -v));
    traw = traw.map((v) => -v);
  }
  if (mat3Det(Mraw) < 0) {
    Mraw = Mraw.map((r) => r.map((v) => -v));
    traw = traw.map((v) => -v);
  }

  // 通过极分解求最近正交旋转矩阵 R ∈ SO(3) 与尺度因子
  let frob2 = 0;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) frob2 += Mraw[r][c] * Mraw[r][c];
  const initScale = Math.max(Math.sqrt(frob2 / 3), 1e-6);

  let R = Mraw.map((r) => r.map((v) => v / initScale));
  for (let iter = 0; iter < 12; iter++) {
    const invT = mat3InvTranspose(R);
    if (!invT) break;
    let diff = 0;
    const nextR = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        nextR[r][c] = 0.5 * (R[r][c] + invT[r][c]);
        diff += Math.abs(nextR[r][c] - R[r][c]);
      }
    }
    R = nextR;
    if (diff < 1e-10) break;
  }

  let trRtM = 0;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) trRtM += R[r][c] * Mraw[r][c];
  const alpha = Math.max(trRtM / 3, 1e-6);
  const t = traw.map((v) => v / alpha);
  return { R, t };
}

/**
 * Levenberg-Marquardt 6自由度非线性最小二乘精化（直接最小化含畸变像素重投影残差）
 */
function refineLM(Rinit, tinit, pts3D, pts2D, indices, intrinsics, distortion, maxIters = 35) {
  let R = Rinit.map((r) => [...r]);
  let t = [...tinit];
  let lambda = 1e-3;

  const evalResiduals = (Rc, tc) => {
    const res = new Float64Array(indices.length * 2);
    let cost = 0;
    for (let k = 0; k < indices.length; k++) {
      const idx = indices[k];
      const pcv = mat3Vec(Rc, pts3D[idx]);
      pcv[0] += tc[0];
      pcv[1] += tc[1];
      pcv[2] += tc[2];
      const [pu, pv] = projectCVPointToPixel(pcv, intrinsics, distortion);
      const du = pu - pts2D[idx][0];
      const dv = pv - pts2D[idx][1];
      res[2 * k] = du;
      res[2 * k + 1] = dv;
      cost += du * du + dv * dv;
    }
    return { res, cost };
  };

  let cur = evalResiduals(R, t);
  const eps = 1e-5;

  for (let iter = 0; iter < maxIters; iter++) {
    const m2 = indices.length * 2;
    const J = Array.from({ length: m2 }, () => new Float64Array(6));

    for (let col = 0; col < 6; col++) {
      const d = [0, 0, 0, 0, 0, 0];
      d[col] = eps;
      const dR = rodrigues(d[0], d[1], d[2]);
      const Rpert = mat3Mul(dR, R);
      const tpert = [t[0] + d[3], t[1] + d[4], t[2] + d[5]];
      const pert = evalResiduals(Rpert, tpert);
      for (let row = 0; row < m2; row++) {
        J[row][col] = (pert.res[row] - cur.res[row]) / eps;
      }
    }

    const JtJ = Array.from({ length: 6 }, () => new Float64Array(6));
    const Jtr = new Float64Array(6);
    for (let row = 0; row < m2; row++) {
      for (let i = 0; i < 6; i++) {
        Jtr[i] -= J[row][i] * cur.res[row];
        for (let j = i; j < 6; j++) {
          JtJ[i][j] += J[row][i] * J[row][j];
        }
      }
    }
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < i; j++) JtJ[i][j] = JtJ[j][i];
      JtJ[i][i] += lambda * Math.max(JtJ[i][i], 1e-6);
    }

    const step = solveLinear6x6(JtJ, Jtr);
    if (!step) {
      lambda *= 10;
      continue;
    }

    const stepNorm = Math.hypot(...step);
    if (stepNorm < 1e-8) break;

    const Rnext = mat3Mul(rodrigues(step[0], step[1], step[2]), R);
    const tnext = [t[0] + step[3], t[1] + step[4], t[2] + step[5]];
    const next = evalResiduals(Rnext, tnext);

    if (next.cost < cur.cost) {
      R = Rnext;
      t = tnext;
      cur = next;
      lambda = Math.max(lambda * 0.3, 1e-9);
    } else {
      lambda = Math.min(lambda * 5.0, 1e6);
    }
  }

  return { R, t };
}

function generate6PointSubsets(n, maxTrials = 60) {
  const subsets = [];
  if (n <= 9) {
    // 组合数最多 C(9,6)=84，枚举确定性子集确保 100% 命中无外点组合
    for (let a = 0; a < n; a++) {
      for (let b = a + 1; b < n; b++) {
        for (let c = b + 1; c < n; c++) {
          for (let d = c + 1; d < n; d++) {
            for (let e = d + 1; e < n; e++) {
              for (let f = e + 1; f < n; f++) {
                subsets.push([a, b, c, d, e, f]);
                if (subsets.length >= maxTrials) return subsets;
              }
            }
          }
        }
      }
    }
    return subsets;
  }

  let seed = 1337;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const base = Array.from({ length: n }, (_, i) => i);
  for (let t = 0; t < maxTrials; t++) {
    const arr = [...base];
    for (let i = 0; i < 6; i++) {
      const j = i + Math.floor(rand() * (n - i));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    subsets.push(arr.slice(0, 6));
  }
  return subsets;
}

/**
 * 纯前端 RANSAC + DLT + Levenberg-Marquardt PnP 外参求解器
 * 坐标系与 Python OpenCV `calib_toolkit.py` 严格一致：M[:3,:3] = R_cv^T * diag(1,-1,-1), M[:3,3] = -R_cv^T * t_cv
 */
export function solveCameraPnP({
  pts3D,
  pts2D,
  intrinsics,
  distortion = {},
  useRansac = true,
  ransacThreshPx = 4.0
}) {
  const n = Math.min(pts3D.length, pts2D.length);
  if (n < 6) {
    throw new Error(`PnP 至少需要 6 组不共面控制点（当前 ${n} 组）`);
  }

  const norm2D = pts2D.slice(0, n).map(([u, v]) => undistortPixel(u, v, intrinsics, distortion));
  const allIndices = Array.from({ length: n }, (_, i) => i);

  let bestInliers = allIndices;
  let bestR = null;
  let bestT = null;

  if (useRansac && n >= 7) {
    const subsets = generate6PointSubsets(n, 84);
    let bestCount = -1;
    let bestErrSum = Infinity;

    for (const sub of subsets) {
      const dlt = solveDLT(pts3D, norm2D, sub);
      const quick = refineLM(dlt.R, dlt.t, pts3D, pts2D, sub, intrinsics, distortion, 12);

      const inliers = [];
      let errSum = 0;
      for (let i = 0; i < n; i++) {
        const pcv = mat3Vec(quick.R, pts3D[i]);
        pcv[0] += quick.t[0];
        pcv[1] += quick.t[1];
        pcv[2] += quick.t[2];
        if (pcv[2] <= 0.1) continue;
        const [pu, pv] = projectCVPointToPixel(pcv, intrinsics, distortion);
        const err = Math.hypot(pu - pts2D[i][0], pv - pts2D[i][1]);
        if (err <= ransacThreshPx) {
          inliers.push(i);
          errSum += err;
        }
      }

      if (inliers.length > bestCount || (inliers.length === bestCount && errSum < bestErrSum)) {
        bestCount = inliers.length;
        bestErrSum = errSum;
        bestInliers = inliers;
        bestR = quick.R;
        bestT = quick.t;
      }
    }
  }

  if (!bestR || bestInliers.length < 6) {
    bestInliers = allIndices;
    const dltAll = solveDLT(pts3D, norm2D, allIndices);
    bestR = dltAll.R;
    bestT = dltAll.t;
  } else {
    const dltInliers = solveDLT(pts3D, norm2D, bestInliers);
    bestR = dltInliers.R;
    bestT = dltInliers.t;
  }

  const final = refineLM(bestR, bestT, pts3D, pts2D, bestInliers, intrinsics, distortion, 35);

  let totalErr = 0;
  for (const idx of bestInliers) {
    const pcv = mat3Vec(final.R, pts3D[idx]);
    pcv[0] += final.t[0];
    pcv[1] += final.t[1];
    pcv[2] += final.t[2];
    const [pu, pv] = projectCVPointToPixel(pcv, intrinsics, distortion);
    totalErr += Math.hypot(pu - pts2D[idx][0], pv - pts2D[idx][1]);
  }
  const reprojectionErrorPx = totalErr / bestInliers.length;

  // R_three = R_cv^T * diag(1, -1, -1), pos_three = -R_cv^T * t_cv
  const R = final.R;
  const t = final.t;
  const pos = [
    -(R[0][0] * t[0] + R[1][0] * t[1] + R[2][0] * t[2]),
    -(R[0][1] * t[0] + R[1][1] * t[1] + R[2][1] * t[2]),
    -(R[0][2] * t[0] + R[1][2] * t[1] + R[2][2] * t[2])
  ];

  const rowMajor = [
    R[0][0], -R[1][0], -R[2][0], pos[0],
    R[0][1], -R[1][1], -R[2][1], pos[1],
    R[0][2], -R[1][2], -R[2][2], pos[2],
    0,       0,        0,        1
  ].map((v) => +v.toFixed(6));

  const m4 = new THREE.Matrix4().set(...rowMajor);
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  m4.decompose(position, quaternion, new THREE.Vector3());

  return {
    matrixWorldRowMajor: rowMajor,
    position,
    quaternion,
    reprojectionErrorPx: +reprojectionErrorPx.toFixed(3),
    inliers: bestInliers,
    inlierCount: bestInliers.length,
    totalCount: n
  };
}

/**
 * 挂载轻量 Web 端 2D/3D 配对标定弹窗控制器
 */
export function mountWebCalibratorModal({
  getSelectedCamera,
  getCaptureSource,
  getPoints3D,
  onApplyCalibration
}) {
  const modal = document.getElementById('calib-modal');
  const canvas = document.getElementById('calib-canvas');
  const ctx = canvas.getContext('2d');
  const summaryEl = document.getElementById('calib-summary');
  const imgInput = document.getElementById('calib-img-upload');

  let bgImage = null;
  let camWidth = 1920;
  let camHeight = 1080;
  let pts2D = [];

  const redraw = () => {
    const cw = canvas.width;
    const ch = canvas.height;
    ctx.clearRect(0, 0, cw, ch);

    if (bgImage) {
      ctx.drawImage(bgImage, 0, 0, cw, ch);
    } else {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, cw, ch);
      ctx.fillStyle = '#94a3b8';
      ctx.font = '16px sans-serif';
      ctx.fillText('请点击「抓取当前视频帧」或上传标定参考图，然后左键按 3D 采点顺序点击 2D 对应点', 24, 40);
    }

    pts2D.forEach(([u, v], idx) => {
      const px = (u / camWidth) * cw;
      const py = (v / camHeight) * ch;
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = '#fde047';
      ctx.font = 'bold 14px monospace';
      ctx.fillText(`#${idx + 1}`, px + 9, py - 6);
    });

    const pts3D = getPoints3D();
    const camItem = getSelectedCamera();
    summaryEl.textContent = `当前相机: ${camItem?.id || '-'} (${camWidth}x${camHeight}) | 3D 点: ${pts3D.length} 个 | 2D 点: ${pts2D.length} 个`;
  };

  const captureFromCamera = () => {
    const camItem = getSelectedCamera();
    if (!camItem) return;
    camWidth = camItem.intrinsics.width || 1920;
    camHeight = camItem.intrinsics.height || 1080;
    const src = getCaptureSource(camItem.id);
    if (src) {
      const off = document.createElement('canvas');
      off.width = camWidth;
      off.height = camHeight;
      off.getContext('2d').drawImage(src, 0, 0, camWidth, camHeight);
      bgImage = off;
    }
    redraw();
  };

  canvas.addEventListener('click', (ev) => {
    const rect = canvas.getBoundingClientRect();
    const rx = (ev.clientX - rect.left) / rect.width;
    const ry = (ev.clientY - rect.top) / rect.height;
    pts2D.push([Math.round(rx * camWidth), Math.round(ry * camHeight)]);
    redraw();
  });

  canvas.addEventListener('contextmenu', (ev) => {
    ev.preventDefault();
    if (pts2D.length) {
      pts2D.pop();
      redraw();
    }
  });

  imgInput.onchange = (ev) => {
    const file = ev.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      camWidth = img.naturalWidth || camWidth;
      camHeight = img.naturalHeight || camHeight;
      bgImage = img;
      URL.revokeObjectURL(url);
      redraw();
    };
    img.src = url;
  };

  document.getElementById('btn-open-calib').onclick = () => {
    modal.style.display = 'flex';
    captureFromCamera();
  };
  document.getElementById('btn-close-calib').onclick = () => {
    modal.style.display = 'none';
  };
  document.getElementById('btn-calib-capture').onclick = captureFromCamera;
  document.getElementById('btn-calib-undo').onclick = () => {
    pts2D.pop();
    redraw();
  };
  document.getElementById('btn-calib-clear').onclick = () => {
    pts2D = [];
    redraw();
  };

  document.getElementById('btn-calib-solve').onclick = () => {
    const camItem = getSelectedCamera();
    const pts3DRaw = getPoints3D();
    if (!camItem) return alert('请先加载相机配置');
    if (pts3DRaw.length < 6 || pts2D.length < 6) {
      return alert('请确保已采集至少 6 组不共面 3D 点与对应的 2D 点');
    }
    if (pts3DRaw.length !== pts2D.length) {
      return alert(`3D 点 (${pts3DRaw.length}) 与 2D 点 (${pts2D.length}) 数量不一致，请核对顺序与数量`);
    }

    try {
      const pts3D = pts3DRaw.map((p) => [p.x, p.y, p.z]);
      const result = solveCameraPnP({
        pts3D,
        pts2D,
        intrinsics: camItem.intrinsics,
        distortion: camItem.distortion,
        useRansac: pts3D.length >= 7
      });
      onApplyCalibration(camItem, result);
      modal.style.display = 'none';
      alert(
        `求解成功！\n内点: ${result.inlierCount}/${result.totalCount}\n平均重投影误差: ${result.reprojectionErrorPx} px`
      );
    } catch (err) {
      alert('PnP 求解失败: ' + err.message);
    }
  };

  document.getElementById('btn-calib-export').onclick = () => {
    const camItem = getSelectedCamera();
    const pts3DRaw = getPoints3D();
    if (!camItem) return;
    const payload = {
      camId: camItem.id,
      imageSize: [camItem.intrinsics.width, camItem.intrinsics.height],
      intrinsics: camItem.intrinsics,
      distortion: camItem.distortion,
      pts_3d: pts3DRaw.map((p) => [p.x, p.y, p.z]),
      pts_2d: pts2D
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `calib_points_${camItem.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
}
