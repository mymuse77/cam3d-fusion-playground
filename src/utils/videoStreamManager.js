import * as THREE from 'three';

const AUTO_BG_W = 320;
const AUTO_BG_H = 180;
const AUTO_BG_SAMPLES = 9;
const AUTO_BG_INTERVAL_MS = 400;

/**
 * 多路视频流生命周期管理器：
 * 1. 支持任意相机独立绑定本地视频（MP4/WebM）、远程流（MP4/HLS/WebRTC WHEP）或测试画布
 * 2. 与 CameraScheduler 激活槽位联动：滑出物理槽位超过宽限期自动暂停解码，切回自动恢复
 * 3. 监听页面可见性（visibilitychange）全局节能，支持 WHEP 断线指数退避重连与完整资源释放
 * 4. 动静双轨纹理：每路同时持有实时视频纹理与静态背景纹理（显式 bgUrl 或自动时域中值建模），
 *    通过 setForegroundMode() 在「原始直投 / 动静分离 + 3D 立牌 / 纯静态背景」间无感切换场景投影源
 */
export class VideoStreamManager {
  constructor({ onTextureChange, onStatus } = {}) {
    this.onTextureChange = onTextureChange;
    this.onStatus = onStatus;
    this.entries = new Map();
    this.foregroundMode = 'billboard'; // 'billboard' | 'freeview' | 'avatar3d' | 'raw' | 'clean_bg'
    this._visibilitySuspended = false;

    this._onVisibilityChange = () => {
      if (typeof document === 'undefined') return;
      if (document.hidden) {
        this._visibilitySuspended = true;
        for (const entry of this.entries.values()) {
          if (entry.video && !entry.video.paused) entry.video.pause();
        }
      } else {
        this._visibilitySuspended = false;
        for (const entry of this.entries.values()) {
          if (entry.video && !entry.suspended) entry.video.play().catch(() => {});
        }
      }
    };

    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', this._onVisibilityChange);
    }
  }

  // ==========================================
  // 动静双轨：场景投影纹理选择与模式切换
  // ==========================================
  _getSceneProjectionTexture(entry) {
    if (this.foregroundMode === 'raw' || !entry.bgTexture) {
      return entry.videoTexture;
    }
    return entry.bgTexture;
  }

  _emitTexture(entry) {
    entry.texture = this._getSceneProjectionTexture(entry);
    this.onTextureChange?.(entry.camId, entry.texture, entry);
  }

  setForegroundMode(mode) {
    if (!['billboard', 'freeview', 'avatar3d', 'raw', 'clean_bg'].includes(mode)) return;
    this.foregroundMode = mode;
    for (const entry of this.entries.values()) {
      this._emitTexture(entry);
    }
  }

  /**
   * 供 3D 视频立牌模块读取：实时视频元素/纹理 + 静态背景元素/纹理
   * 仅返回正在播放且未休眠的活跃流，避免休眠相机的冻结画面注入静止重影
   */
  getLiveForegroundPair(camId) {
    const entry = this.entries.get(camId);
    if (!entry || !entry.video || entry.suspended || entry.video.paused) return null;
    return {
      videoEl: entry.video,
      bgSourceEl: entry.bgSourceEl || null,
      videoTexture: entry.videoTexture,
      bgTexture: entry.bgTexture || null
    };
  }

  getBackgroundTexture(camId) {
    return this.entries.get(camId)?.bgTexture || null;
  }

  _setBackgroundSource(entry, sourceEl) {
    if (entry.bgTexture) entry.bgTexture.dispose();
    const tex = sourceEl instanceof HTMLCanvasElement ? new THREE.CanvasTexture(sourceEl) : new THREE.Texture(sourceEl);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    tex.needsUpdate = true;
    entry.bgSourceEl = sourceEl;
    entry.bgTexture = tex;
    this._emitTexture(entry);
  }

  _loadBackgroundUrl(entry, bgUrl) {
    if (!bgUrl || typeof Image === 'undefined') return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      if (entry.disposed || this.entries.get(entry.camId) !== entry) return;
      this._setBackgroundSource(entry, img);
      this.onStatus?.(`[${entry.camId}] 已加载静态背景底图`);
    };
    img.onerror = () => {
      // 背景图加载失败时退回自动时域中值建模
      entry.autoBg = this._createAutoBgState();
    };
    img.src = bgUrl;
  }

  _createAutoBgState() {
    return { samples: [], lastSampleMs: -Infinity, done: false };
  }

  /**
   * 对未提供 bgUrl 的视频流，播放期间每 400ms 采样一帧，凑满 9 帧后计算逐像素时域中值作为背景
   */
  _tickAutoBackground(entry, nowMs) {
    const state = entry.autoBg;
    if (!state || state.done || !entry.video || entry.video.readyState < 2 || entry.video.paused) return;
    if (nowMs - state.lastSampleMs < AUTO_BG_INTERVAL_MS) return;
    state.lastSampleMs = nowMs;

    if (!this._autoCanvas) {
      this._autoCanvas = document.createElement('canvas');
      this._autoCanvas.width = AUTO_BG_W;
      this._autoCanvas.height = AUTO_BG_H;
      this._autoCtx = this._autoCanvas.getContext('2d', { willReadFrequently: true });
    }
    this._autoCtx.drawImage(entry.video, 0, 0, AUTO_BG_W, AUTO_BG_H);
    state.samples.push(this._autoCtx.getImageData(0, 0, AUTO_BG_W, AUTO_BG_H).data);
    if (state.samples.length < AUTO_BG_SAMPLES) return;

    const n = state.samples.length;
    const out = new Uint8ClampedArray(AUTO_BG_W * AUTO_BG_H * 4);
    const buf = new Uint8Array(n);
    for (let i = 0; i < out.length; i++) {
      if ((i & 3) === 3) {
        out[i] = 255;
        continue;
      }
      for (let k = 0; k < n; k++) buf[k] = state.samples[k][i];
      buf.sort();
      out[i] = buf[n >> 1];
    }
    const bgCanvas = document.createElement('canvas');
    bgCanvas.width = AUTO_BG_W;
    bgCanvas.height = AUTO_BG_H;
    bgCanvas.getContext('2d').putImageData(new ImageData(out, AUTO_BG_W, AUTO_BG_H), 0, 0);
    state.done = true;
    state.samples = [];
    this._setBackgroundSource(entry, bgCanvas);
    this.onStatus?.(`[${entry.camId}] 已自动完成时域中值背景建模`);
  }

  // ==========================================
  // 视频源绑定
  // ==========================================
  bindCanvas(camId, canvasAsset) {
    this.releaseCamera(camId);
    const entry = {
      camId,
      type: 'canvas',
      url: '',
      autoSuspend: false,
      video: null,
      canvasEl: canvasAsset.texture.image,
      videoTexture: canvasAsset.texture,
      texture: canvasAsset.texture,
      canvasAsset,
      suspended: false,
      lastActiveMs: typeof performance !== 'undefined' ? performance.now() : 0
    };
    this.entries.set(camId, entry);
    this._emitTexture(entry);
    return entry;
  }

  bindLocalFile(camId, file) {
    this.releaseCamera(camId);
    const objectUrl = URL.createObjectURL(file);
    const video = this._createVideoElement(true);
    video.src = objectUrl;
    video.play().catch(() => {});

    const videoTexture = this._createVideoTexture(video);
    const entry = {
      camId,
      type: 'mp4',
      url: file.name,
      autoSuspend: true,
      video,
      objectUrl,
      videoTexture,
      texture: videoTexture,
      autoBg: this._createAutoBgState(),
      suspended: false,
      lastActiveMs: typeof performance !== 'undefined' ? performance.now() : 0
    };
    this.entries.set(camId, entry);
    this._emitTexture(entry);
    this.onStatus?.(`[${camId}] 已绑定本地视频: ${file.name}`);
    return entry;
  }

  bindStreamUrl(camId, url, streamType = 'auto', autoSuspend = true, bgUrl = '') {
    this.releaseCamera(camId);
    const resolvedType = this._detectStreamType(url, streamType);
    const video = this._createVideoElement(resolvedType !== 'whep');
    const videoTexture = this._createVideoTexture(video);

    const entry = {
      camId,
      type: resolvedType,
      url,
      autoSuspend,
      video,
      videoTexture,
      texture: videoTexture,
      pc: null,
      reconnectTimer: null,
      retryCount: 0,
      disposed: false,
      autoBg: bgUrl ? null : this._createAutoBgState(),
      suspended: false,
      lastActiveMs: typeof performance !== 'undefined' ? performance.now() : 0
    };
    this.entries.set(camId, entry);

    if (resolvedType === 'whep') {
      this._connectWhep(entry);
    } else {
      if (resolvedType === 'hls' && video.canPlayType('application/vnd.apple.mpegurl') === '') {
        this.onStatus?.(`[${camId}] 当前浏览器不原生支持 HLS (.m3u8)，建议转用 WHEP WebRTC 或 MP4 流`);
      }
      video.src = url;
      video.play().catch(() => {});
      this.onStatus?.(`[${camId}] 已接入 ${resolvedType.toUpperCase()} 流: ${url}`);
    }

    this._emitTexture(entry);
    if (bgUrl) this._loadBackgroundUrl(entry, bgUrl);
    return entry;
  }

  _detectStreamType(url, hint) {
    if (hint && hint !== 'auto') return hint;
    const lower = (url || '').toLowerCase();
    if (lower.includes('/whep') || lower.startsWith('whep://') || lower.startsWith('webrtc://')) return 'whep';
    if (lower.includes('.m3u8')) return 'hls';
    return 'mp4';
  }

  _createVideoElement(loop) {
    const video = document.createElement('video');
    video.muted = true;
    video.loop = loop;
    video.playsInline = true;
    video.autoplay = true;
    video.crossOrigin = 'anonymous';
    return video;
  }

  _createVideoTexture(video) {
    const texture = new THREE.VideoTexture(video);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    return texture;
  }

  async _connectWhep(entry) {
    if (entry.disposed || typeof RTCPeerConnection === 'undefined') {
      this.onStatus?.(`[${entry.camId}] 当前环境不支持 WebRTC RTCPeerConnection`);
      return;
    }

    const endpoint = entry.url.replace(/^whep:\/\//i, 'https://').replace(/^webrtc:\/\//i, 'https://');
    const pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
    entry.pc = pc;
    pc.addTransceiver('video', { direction: 'recvonly' });

    pc.ontrack = (ev) => {
      if (entry.disposed) return;
      entry.retryCount = 0;
      entry.video.srcObject = ev.streams?.[0] || new MediaStream([ev.track]);
      entry.video.play().catch(() => {});
      this.onStatus?.(`[${entry.camId}] WHEP WebRTC 视频轨已连接`);
    };

    pc.onconnectionstatechange = () => {
      if (entry.disposed) return;
      if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
        this._scheduleWhepReconnect(entry);
      }
    };

    try {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await this._waitForIceComplete(pc, 1200);

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/sdp' },
        body: pc.localDescription.sdp
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const answerSdp = await res.text();
      if (!entry.disposed) {
        await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp });
      }
    } catch (err) {
      if (!entry.disposed) {
        this.onStatus?.(`[${entry.camId}] WHEP 信令连接失败: ${err.message}`);
        this._scheduleWhepReconnect(entry);
      }
    }
  }

  _waitForIceComplete(pc, timeoutMs) {
    if (pc.iceGatheringState === 'complete') return Promise.resolve();
    return new Promise((resolve) => {
      const timer = setTimeout(resolve, timeoutMs);
      const onState = () => {
        if (pc.iceGatheringState === 'complete') {
          clearTimeout(timer);
          pc.removeEventListener('icegatheringstatechange', onState);
          resolve();
        }
      };
      pc.addEventListener('icegatheringstatechange', onState);
    });
  }

  _scheduleWhepReconnect(entry) {
    if (entry.disposed || entry.reconnectTimer || entry.retryCount >= 5) return;
    const delayMs = Math.min(2000 * Math.pow(2, entry.retryCount), 16000);
    entry.retryCount += 1;
    this.onStatus?.(`[${entry.camId}] WHEP 将在 ${(delayMs / 1000).toFixed(0)}s 后第 ${entry.retryCount} 次重连...`);

    entry.reconnectTimer = setTimeout(() => {
      entry.reconnectTimer = null;
      if (entry.disposed) return;
      if (entry.pc) {
        entry.pc.close();
        entry.pc = null;
      }
      this._connectWhep(entry);
    }, delayMs);
  }

  /**
   * 获取当前首个正在播放的 MP4 主时钟，用于多路同步回放对齐与外部 3D 轨迹同步
   */
  getMasterMp4Time() {
    for (const entry of this.entries.values()) {
      if (
        entry.type === 'mp4' &&
        entry.video &&
        !entry.suspended &&
        !entry.video.paused &&
        entry.video.readyState >= 2 &&
        Number.isFinite(entry.video.duration) &&
        entry.video.duration > 1
      ) {
        return entry.video.currentTime;
      }
    }
    return null;
  }

  _getMasterMp4Time() {
    return this.getMasterMp4Time();
  }

  /**
   * 与 CameraScheduler 槽位联动：非激活流超过宽限期自动暂停解码，切回自动恢复并追齐主时钟
   */
  syncActiveCameras(activeIdSet, nowMs = performance.now(), graceMs = 3000) {
    const masterTime = this._getMasterMp4Time();
    for (const [camId, entry] of this.entries.entries()) {
      if (!entry.video || entry.autoSuspend === false) continue;

      if (activeIdSet.has(camId)) {
        entry.lastActiveMs = nowMs;
        if (entry.suspended) {
          entry.suspended = false;
          if (
            entry.type === 'mp4' &&
            masterTime !== null &&
            Number.isFinite(entry.video.duration) &&
            entry.video.duration > 1
          ) {
            entry.video.currentTime = masterTime % entry.video.duration;
          }
          if (!this._visibilitySuspended) entry.video.play().catch(() => {});
        }
      } else if (!entry.suspended && nowMs - entry.lastActiveMs > graceMs) {
        entry.suspended = true;
        entry.video.pause();
      }
    }
  }

  /**
   * 每帧驱动：刷新测试画布动画（纯静态背景模式下冻结）+ 自动背景建模采样 + 多路 MP4 时钟对齐
   */
  update(elapsed, nowMs = performance.now()) {
    if (nowMs - (this._lastClockSyncMs || 0) >= 1500) {
      this._lastClockSyncMs = nowMs;
      const masterTime = this._getMasterMp4Time();
      if (masterTime !== null) {
        for (const entry of this.entries.values()) {
          if (
            entry.type !== 'mp4' ||
            !entry.video ||
            entry.suspended ||
            entry.video.paused ||
            entry.video.readyState < 2 ||
            !Number.isFinite(entry.video.duration) ||
            entry.video.duration <= 1
          ) {
            continue;
          }
          const target = masterTime % entry.video.duration;
          const diff = Math.abs(entry.video.currentTime - target);
          const wrapDiff = Math.min(diff, entry.video.duration - diff);
          if (wrapDiff > 0.35) {
            entry.video.currentTime = target;
          }
        }
      }
    }

    for (const entry of this.entries.values()) {
      if (entry.canvasAsset) {
        if (this.foregroundMode !== 'clean_bg') entry.canvasAsset.update(elapsed);
      } else if (entry.autoBg) {
        this._tickAutoBackground(entry, nowMs);
      }
    }
  }

  updateCanvases(elapsed) {
    this.update(elapsed);
  }

  getCaptureSource(camId) {
    const entry = this.entries.get(camId);
    if (!entry) return null;
    return entry.video || entry.canvasEl || null;
  }

  releaseCamera(camId) {
    const entry = this.entries.get(camId);
    if (!entry) return;
    entry.disposed = true;
    if (entry.reconnectTimer) {
      clearTimeout(entry.reconnectTimer);
      entry.reconnectTimer = null;
    }
    if (entry.pc) {
      entry.pc.close();
      entry.pc = null;
    }
    if (entry.video) {
      entry.video.pause();
      entry.video.srcObject = null;
      entry.video.removeAttribute('src');
      entry.video.load();
    }
    if (entry.objectUrl) URL.revokeObjectURL(entry.objectUrl);
    if (entry.videoTexture) entry.videoTexture.dispose();
    if (entry.bgTexture) entry.bgTexture.dispose();
    this.entries.delete(camId);
  }

  disposeAll() {
    for (const camId of Array.from(this.entries.keys())) {
      this.releaseCamera(camId);
    }
  }
}
