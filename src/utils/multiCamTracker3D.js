const INF_COST = 1e6;

/**
 * 4D 恒速卡尔曼滤波器：状态向量 [px, pz, vx, vz]^T，高程 y 随真实地形自适应平滑
 */
class KalmanTrack3D {
  constructor(globalId, cls, x, y, z, nowMs) {
    this.globalId = globalId;
    this.cls = cls;
    this.state = [x, z, 0, 0];
    this.y = y;
    this.lastSeen = nowMs;

    // 4x4 初始状态协方差（展平为长度 16 的数组）
    this.P = [
      1.0, 0,   0,   0,
      0,   1.0, 0,   0,
      0,   0,   4.0, 0,
      0,   0,   0,   4.0
    ];
  }

  get x() { return this.state[0]; }
  get z() { return this.state[1]; }
  get vx() { return this.state[2]; }
  get vz() { return this.state[3]; }

  predictPosition(dt) {
    return {
      x: this.state[0] + this.state[2] * dt,
      z: this.state[1] + this.state[3] * dt
    };
  }

  predict(dt) {
    if (dt <= 0) return;
    const [px, pz, vx, vz] = this.state;
    this.state[0] = px + vx * dt;
    this.state[1] = pz + vz * dt;

    // 过程噪声 Q（连续白噪声加速度模型）
    const sigmaA2 = 2.2;
    const dt2 = dt * dt;
    const dt3 = dt2 * dt * 0.5 * sigmaA2;
    const dt4 = dt2 * dt2 * 0.25 * sigmaA2;
    const qv = dt2 * sigmaA2;

    const F = [
      1, 0, dt, 0,
      0, 1, 0, dt,
      0, 0, 1, 0,
      0, 0, 0, 1
    ];
    const FP = mat4Mul(F, this.P);
    const FPFt = mat4MulTransposeB(FP, F);

    FPFt[0] += dt4; FPFt[2] += dt3;
    FPFt[5] += dt4; FPFt[7] += dt3;
    FPFt[8] += dt3; FPFt[10] += qv;
    FPFt[13] += dt3; FPFt[15] += qv;
    this.P = FPFt;
  }

  update(zx, zy, zz, nowMs, measNoiseVar = 0.25) {
    const dt = Math.min(Math.max((nowMs - this.lastSeen) / 1000, 0.001), 1.5);
    this.predict(dt);

    // 观测矩阵 H = [[1,0,0,0], [0,1,0,0]]
    const y0 = zx - this.state[0];
    const y1 = zz - this.state[1];

    const s00 = this.P[0] + measNoiseVar;
    const s01 = this.P[1];
    const s10 = this.P[4];
    const s11 = this.P[5] + measNoiseVar;
    const det = s00 * s11 - s01 * s10;
    if (Math.abs(det) < 1e-9) return;

    const invS00 = s11 / det;
    const invS01 = -s01 / det;
    const invS10 = -s10 / det;
    const invS11 = s00 / det;

    // 卡尔曼增益 K (4x2) = P * H^T * S^-1
    const K = new Array(8);
    for (let r = 0; r < 4; r++) {
      const p0 = this.P[r * 4];
      const p1 = this.P[r * 4 + 1];
      K[r * 2] = p0 * invS00 + p1 * invS10;
      K[r * 2 + 1] = p0 * invS01 + p1 * invS11;
    }

    // 状态更新 x = x + K * y
    for (let r = 0; r < 4; r++) {
      this.state[r] += K[r * 2] * y0 + K[r * 2 + 1] * y1;
    }

    // 协方差更新 P = (I - K H) * P
    const IKH = [
      1 - K[0], -K[1], 0, 0,
      -K[2], 1 - K[3], 0, 0,
      -K[4], -K[5], 1, 0,
      -K[6], -K[7], 0, 1
    ];
    this.P = mat4Mul(IKH, this.P);
    this.y = this.y + (zy - this.y) * 0.65;
    this.lastSeen = nowMs;
  }

  toSnapshot() {
    return {
      globalId: this.globalId,
      cls: this.cls,
      x: this.state[0],
      y: this.y,
      z: this.state[1],
      vx: this.state[2],
      vz: this.state[3],
      lastSeen: this.lastSeen
    };
  }
}

function mat4Mul(A, B) {
  const C = new Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      C[r * 4 + c] =
        A[r * 4] * B[c] +
        A[r * 4 + 1] * B[4 + c] +
        A[r * 4 + 2] * B[8 + c] +
        A[r * 4 + 3] * B[12 + c];
    }
  }
  return C;
}

function mat4MulTransposeB(A, B) {
  const C = new Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      C[r * 4 + c] =
        A[r * 4] * B[c * 4] +
        A[r * 4 + 1] * B[c * 4 + 1] +
        A[r * 4 + 2] * B[c * 4 + 2] +
        A[r * 4 + 3] * B[c * 4 + 3];
    }
  }
  return C;
}

/**
 * 标准匈牙利全局最小代价指派算法（Kuhn-Munkres 增广路，支持任意矩形代价矩阵）
 * @returns {Array<[number, number]>} 匹配成功的 [rowIdx, colIdx] 对
 */
function hungarianAssign(costMatrix, maxValidCost) {
  const rows = costMatrix.length;
  const cols = rows > 0 ? costMatrix[0].length : 0;
  if (rows === 0 || cols === 0) return [];

  const n = Math.max(rows, cols);
  const a = Array.from({ length: n + 1 }, (_, i) =>
    Array.from({ length: n + 1 }, (_, j) =>
      i > 0 && j > 0 && i <= rows && j <= cols ? costMatrix[i - 1][j - 1] : INF_COST
    )
  );

  const u = new Array(n + 1).fill(0);
  const v = new Array(n + 1).fill(0);
  const p = new Array(n + 1).fill(0);
  const way = new Array(n + 1).fill(0);

  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array(n + 1).fill(Infinity);
    const used = new Array(n + 1).fill(false);

    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = a[i0][j] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);

    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }

  const pairs = [];
  for (let j = 1; j <= n; j++) {
    const r = p[j] - 1;
    const c = j - 1;
    if (r >= 0 && r < rows && c >= 0 && c < cols && costMatrix[r][c] <= maxValidCost) {
      pairs.push([r, c]);
    }
  }
  return pairs;
}

/**
 * 多摄像头重叠区 3D 目标凝聚聚类 + 4D 卡尔曼滤波 + 匈牙利全局最优跨镜追踪器
 */
export class MultiCamTracker3D {
  constructor(options = {}) {
    this.distThresholds = {
      person: options.personThreshold ?? 1.5,
      car: options.carThreshold ?? 3.0,
      default: 2.0
    };
    this.maxAgeMs = options.maxAgeMs ?? 2000;
    this.nextGlobalId = 1;
    this.tracks = new Map();
    this.localToGlobal = new Map();
  }

  update(observations, nowMs = performance.now()) {
    const clusters = this._clusterObservations(observations);
    const trackList = Array.from(this.tracks.values());
    const matchedClusterIdx = new Set();

    if (trackList.length > 0 && clusters.length > 0) {
      const maxThresh = Math.max(...Object.values(this.distThresholds)) * 1.6;
      const costMatrix = trackList.map((track) => {
        const dt = Math.min(Math.max((nowMs - track.lastSeen) / 1000, 0), 1.2);
        const pred = track.predictPosition(dt);
        const thresh = this.distThresholds[track.cls] ?? this.distThresholds.default;
        const gate = thresh * 1.45;

        return clusters.map((c) => {
          if (track.cls !== c.cls) return INF_COST;
          const dist = Math.hypot(pred.x - c.x, pred.z - c.z);
          const hasLocalAffinity = c.localKeys.some((k) => this.localToGlobal.get(k) === track.globalId);
          if (dist > gate && !(hasLocalAffinity && dist <= gate * 1.4)) {
            return INF_COST;
          }
          // 结合预测位置距离与局部 ID 绑定亲和度（优先保持跨帧 ID 连贯）
          return Math.max(0.001, dist - (hasLocalAffinity ? 0.35 : 0));
        });
      });

      const assignments = hungarianAssign(costMatrix, maxThresh);
      for (const [tIdx, cIdx] of assignments) {
        const track = trackList[tIdx];
        const cluster = clusters[cIdx];
        // 多视角融合观测噪声显著低于单视角
        const measVar = cluster.camIds.size > 1 ? 0.12 : 0.28;
        track.update(cluster.x, cluster.y, cluster.z, nowMs, measVar);
        cluster.localKeys.forEach((k) => this.localToGlobal.set(k, track.globalId));
        matchedClusterIdx.add(cIdx);
      }
    }

    for (let i = 0; i < clusters.length; i++) {
      if (matchedClusterIdx.has(i)) continue;
      const c = clusters[i];
      const gid = this.nextGlobalId++;
      this.tracks.set(gid, new KalmanTrack3D(gid, c.cls, c.x, c.y, c.z, nowMs));
      c.localKeys.forEach((k) => this.localToGlobal.set(k, gid));
    }

    this._pruneExpiredTracks(nowMs);
    return Array.from(this.tracks.values()).map((t) => t.toSnapshot());
  }

  /**
   * 基于置信度距离权重的凝聚聚类（Agglomerative Clustering，消除观测输入顺序敏感性）
   */
  _clusterObservations(observations) {
    const clusters = observations.map((obs) => {
      const w = 1.0 / Math.pow(Math.max(obs.distToCam || 1.0, 1.0), 1.2);
      return {
        cls: obs.cls,
        x: obs.x,
        y: obs.y,
        z: obs.z,
        totalWeight: w,
        camIds: new Set([obs.camId]),
        localKeys: [`${obs.camId}:${obs.localTrackId}`]
      };
    });

    while (clusters.length > 1) {
      let bestI = -1;
      let bestJ = -1;
      let bestDist = Infinity;

      for (let i = 0; i < clusters.length; i++) {
        const ci = clusters[i];
        const thresh = this.distThresholds[ci.cls] ?? this.distThresholds.default;
        for (let j = i + 1; j < clusters.length; j++) {
          const cj = clusters[j];
          if (ci.cls !== cj.cls) continue;

          let shareCam = false;
          for (const cid of ci.camIds) {
            if (cj.camIds.has(cid)) {
              shareCam = true;
              break;
            }
          }
          if (shareCam) continue;

          const d = Math.hypot(ci.x - cj.x, ci.z - cj.z);
          if (d <= thresh && d < bestDist) {
            bestDist = d;
            bestI = i;
            bestJ = j;
          }
        }
      }

      if (bestI < 0) break;

      const a = clusters[bestI];
      const b = clusters[bestJ];
      const nw = a.totalWeight + b.totalWeight;
      a.x = (a.x * a.totalWeight + b.x * b.totalWeight) / nw;
      a.y = (a.y * a.totalWeight + b.y * b.totalWeight) / nw;
      a.z = (a.z * a.totalWeight + b.z * b.totalWeight) / nw;
      a.totalWeight = nw;
      b.camIds.forEach((cid) => a.camIds.add(cid));
      b.localKeys.forEach((k) => a.localKeys.push(k));
      clusters.splice(bestJ, 1);
    }

    return clusters;
  }

  _pruneExpiredTracks(nowMs) {
    for (const [gid, t] of this.tracks.entries()) {
      if (nowMs - t.lastSeen > this.maxAgeMs) {
        this.tracks.delete(gid);
        for (const [k, v] of this.localToGlobal.entries()) {
          if (v === gid) this.localToGlobal.delete(k);
        }
      }
    }
  }
}
