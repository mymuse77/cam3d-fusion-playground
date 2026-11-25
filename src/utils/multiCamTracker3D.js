/**
 * 多摄像头重叠区 3D 目标加权聚类去重 + 跨镜统一 ID 追踪器
 */
export class MultiCamTracker3D {
  constructor(options = {}) {
    this.distThresholds = { person: options.personThreshold ?? 1.5, car: options.carThreshold ?? 3.0, default: 2.0 };
    this.maxAgeMs = options.maxAgeMs ?? 2000;
    this.nextGlobalId = 1;
    this.tracks = new Map();
    this.localToGlobal = new Map();
  }

  update(observations, nowMs = performance.now()) {
    const clusters = this._clusterObservations(observations);
    const matched = new Set();

    for (const c of clusters) {
      let gid = this._findByLocalId(c);
      if (gid === null || matched.has(gid)) {
        gid = this._findBySpatialDistance(c, matched, nowMs);
      }
      if (gid !== null) {
        this._updateTrack(gid, c, nowMs);
        matched.add(gid);
      } else {
        matched.add(this._createTrack(c, nowMs));
      }
    }
    this._pruneExpiredTracks(nowMs);
    return Array.from(this.tracks.values());
  }

  _clusterObservations(observations) {
    const clusters = [];
    for (const obs of observations) {
      const thresh = this.distThresholds[obs.cls] ?? this.distThresholds.default;
      const target = clusters.find(
        (c) => c.cls === obs.cls && !c.camIds.has(obs.camId) && Math.hypot(c.x - obs.x, c.z - obs.z) <= thresh
      );
      const w = 1.0 / Math.max(obs.distToCam || 1.0, 1.0);
      if (!target) {
        clusters.push({
          cls: obs.cls, x: obs.x, y: obs.y, z: obs.z, totalWeight: w,
          camIds: new Set([obs.camId]), localKeys: [`${obs.camId}:${obs.localTrackId}`]
        });
      } else {
        const nw = target.totalWeight + w;
        target.x = (target.x * target.totalWeight + obs.x * w) / nw;
        target.y = (target.y * target.totalWeight + obs.y * w) / nw;
        target.z = (target.z * target.totalWeight + obs.z * w) / nw;
        target.totalWeight = nw;
        target.camIds.add(obs.camId);
        target.localKeys.push(`${obs.camId}:${obs.localTrackId}`);
      }
    }
    return clusters;
  }

  _findByLocalId(cluster) {
    for (const k of cluster.localKeys) {
      const gid = this.localToGlobal.get(k);
      if (gid !== undefined && this.tracks.has(gid)) return gid;
    }
    return null;
  }

  _findBySpatialDistance(cluster, matched, nowMs) {
    const thresh = this.distThresholds[cluster.cls] ?? this.distThresholds.default;
    let bestId = null, minDist = thresh;
    for (const [gid, t] of this.tracks.entries()) {
      if (t.cls !== cluster.cls || matched.has(gid)) continue;
      const dt = Math.min((nowMs - t.lastSeen) / 1000, 1.0);
      const d = Math.hypot(t.x + t.vx * dt - cluster.x, t.z + t.vz * dt - cluster.z);
      if (d < minDist) { minDist = d; bestId = gid; }
    }
    return bestId;
  }

  _createTrack(c, nowMs) {
    const gid = this.nextGlobalId++;
    this.tracks.set(gid, { globalId: gid, cls: c.cls, x: c.x, y: c.y, z: c.z, vx: 0, vz: 0, lastSeen: nowMs });
    c.localKeys.forEach((k) => this.localToGlobal.set(k, gid));
    return gid;
  }

  _updateTrack(gid, c, nowMs) {
    const t = this.tracks.get(gid);
    const dt = Math.max((nowMs - t.lastSeen) / 1000, 0.016);
    const alpha = 0.4;
    const nx = t.x + (c.x - t.x) * alpha, ny = t.y + (c.y - t.y) * alpha, nz = t.z + (c.z - t.z) * alpha;
    t.vx = t.vx * 0.7 + ((nx - t.x) / dt) * 0.3;
    t.vz = t.vz * 0.7 + ((nz - t.z) / dt) * 0.3;
    t.x = nx; t.y = ny; t.z = nz; t.lastSeen = nowMs;
    c.localKeys.forEach((k) => this.localToGlobal.set(k, gid));
  }

  _pruneExpiredTracks(nowMs) {
    for (const [gid, t] of this.tracks.entries()) {
      if (nowMs - t.lastSeen > this.maxAgeMs) {
        this.tracks.delete(gid);
        for (const [k, v] of this.localToGlobal.entries()) if (v === gid) this.localToGlobal.delete(k);
      }
    }
  }
}
