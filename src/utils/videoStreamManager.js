import * as THREE from 'three';

/**
 * 多路视频流生命周期管理器：
 * 1. 支持任意相机独立绑定本地视频（MP4/WebM）、远程流（MP4/HLS/WebRTC WHEP）或测试画布
 * 2. 与 CameraScheduler 激活槽位联动：滑出物理槽位超过宽限期自动暂停解码，切回自动恢复
 * 3. 监听页面可见性（visibilitychange）全局节能，支持 WHEP 断线指数退避重连与完整资源释放
 */
export class VideoStreamManager {
  constructor({ onTextureChange, onStatus } = {}) {
    this.onTextureChange = onTextureChange;
    this.onStatus = onStatus;
    this.entries = new Map();
    this._visibilitySuspended = false;

    this._onVisibilityChange = () => {
      if (typeof document === 'undefined') return;
      if (document.hidden) {
        this._visibilitySuspended = true;
        for (const entry of this.entries.values()) {
          if (entry.video && !entry.video.paused) {
            entry.video.pause();
          }
        }
      } else {
        this._visibilitySuspended = false;
        for (const entry of this.entries.values()) {
          if (entry.video && !entry.suspended) {
            entry.video.play().catch(() => {});
          }
        }
      }
    };

    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', this._onVisibilityChange);
    }
  }

  /**
   * 绑定动态测试画布纹理
   */
  bindCanvas(camId, canvasAsset) {
    this.releaseCamera(camId);
    const entry = {
      camId,
      type: 'canvas',
      url: '',
      autoSuspend: false,
      video: null,
      canvasEl: canvasAsset.texture.image,
      texture: canvasAsset.texture,
      canvasAsset,
      suspended: false,
      lastActiveMs: typeof performance !== 'undefined' ? performance.now() : 0
    };
    this.entries.set(camId, entry);
    this.onTextureChange?.(camId, entry.texture, entry);
    return entry;
  }

  /**
   * 为指定相机绑定本地视频文件
   */
  bindLocalFile(camId, file) {
    this.releaseCamera(camId);
    const objectUrl = URL.createObjectURL(file);
    const video = this._createVideoElement(false);
    video.src = objectUrl;
    video.play().catch(() => {});

    const texture = this._createVideoTexture(video);
    const entry = {
      camId,
      type: 'mp4',
      url: file.name,
      autoSuspend: true,
      video,
      objectUrl,
      texture,
      suspended: false,
      lastActiveMs: typeof performance !== 'undefined' ? performance.now() : 0
    };
    this.entries.set(camId, entry);
    this.onTextureChange?.(camId, texture, entry);
    this.onStatus?.(`[${camId}] 已绑定本地视频: ${file.name}`);
    return entry;
  }

  /**
   * 为指定相机绑定远程流地址（支持 MP4 / HLS / WebRTC WHEP）
   */
  bindStreamUrl(camId, url, streamType = 'auto', autoSuspend = true) {
    this.releaseCamera(camId);
    const resolvedType = this._detectStreamType(url, streamType);
    const video = this._createVideoElement(resolvedType !== 'whep');
    const texture = this._createVideoTexture(video);

    const entry = {
      camId,
      type: resolvedType,
      url,
      autoSuspend,
      video,
      texture,
      pc: null,
      reconnectTimer: null,
      retryCount: 0,
      disposed: false,
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

    this.onTextureChange?.(camId, texture, entry);
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
   * 与 CameraScheduler 槽位联动：非激活流超过宽限期自动暂停解码，切回自动恢复
   */
  syncActiveCameras(activeIdSet, nowMs = performance.now(), graceMs = 3000) {
    for (const [camId, entry] of this.entries.entries()) {
      if (!entry.video || entry.autoSuspend === false) continue;

      if (activeIdSet.has(camId)) {
        entry.lastActiveMs = nowMs;
        if (entry.suspended) {
          entry.suspended = false;
          if (!this._visibilitySuspended) {
            entry.video.play().catch(() => {});
          }
        }
      } else if (!entry.suspended && nowMs - entry.lastActiveMs > graceMs) {
        entry.suspended = true;
        entry.video.pause();
      }
    }
  }

  /**
   * 驱动测试画布刷新，并返回某路当前可用的图像源（用于 Web 标定采帧）
   */
  updateCanvases(elapsed) {
    for (const entry of this.entries.values()) {
      if (entry.canvasAsset) {
        entry.canvasAsset.update(elapsed);
      }
    }
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
    if (entry.objectUrl) {
      URL.revokeObjectURL(entry.objectUrl);
    }
    if (entry.texture) {
      entry.texture.dispose();
    }
    this.entries.delete(camId);
  }

  disposeAll() {
    for (const camId of Array.from(this.entries.keys())) {
      this.releaseCamera(camId);
    }
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this._onVisibilityChange);
    }
  }
}
