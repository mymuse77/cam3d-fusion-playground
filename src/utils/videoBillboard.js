import * as THREE from 'three';

const DET_W = 160;
const DET_H = 90;

/**
 * 估算当前帧相对静态背景的全局曝光增益比（抵消室外云层/自动曝光漂移引起的背景误检与边框残留）
 */
export function estimateExposureGain(vidData, bgData) {
  let sumVid = 0;
  let sumBg = 0;
  // 抽样步长 4 像素，快速估算全局曝光比例
  for (let p = 0; p < vidData.length; p += 16) {
    const lumV = 0.299 * vidData[p] + 0.587 * vidData[p + 1] + 0.114 * vidData[p + 2];
    const lumB = 0.299 * bgData[p] + 0.587 * bgData[p + 1] + 0.114 * bgData[p + 2];
    if (lumB > 20 && lumB < 240 && Math.abs(lumV - lumB) < 42) {
      sumVid += lumV;
      sumBg += lumB;
    }
  }
  if (sumBg < 100) return 1.0;
  return Math.min(Math.max(sumVid / sumBg, 0.78), 1.25);
}

/**
 * 纯算法函数：从缩小后的实时视频帧与静态背景帧 RGBA 像素数组中提取前景目标 2D 检测框
 * 包含曝光漂移自适应补偿、Cucchiara 亮度-色度阴影抑制与上下邻近部位合并（如身体+安全帽）
 * @returns {Array<[number, number, number, number]>} 原分辨率下的 [u1, v1, u2, v2] 数组
 */
export function extractForegroundBlobsFromRgba(vidData, bgData, w, h, fullW, fullH, diffThresh = 30) {
  const mask = new Uint8Array(w * h);
  const gain = estimateExposureGain(vidData, bgData);

  for (let i = 0, p = 0; i < mask.length; i++, p += 4) {
    const r = vidData[p];
    const g = vidData[p + 1];
    const b = vidData[p + 2];
    const br = Math.min(255, bgData[p] * gain);
    const bg = Math.min(255, bgData[p + 1] * gain);
    const bb = Math.min(255, bgData[p + 2] * gain);

    const maxDiff = Math.max(Math.abs(r - br), Math.abs(g - bg), Math.abs(b - bb));
    if (maxDiff < diffThresh) continue;

    // Cucchiara 阴影抑制：仅过滤半透明地面投影暗影（亮度落在背景 48%~94% 且色度几乎不变）；
    // 深黑目标（< 48% 背景亮度，如黑色人形机器人/深色衣物）绝不误判为阴影
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    const bgLum = 0.299 * br + 0.587 * bg + 0.114 * bb;
    const dCr = Math.abs((r - lum) - (br - bgLum));
    const dCb = Math.abs((b - lum) - (bb - bgLum));
    const isCastShadow = lum < bgLum * 0.94 && lum > bgLum * 0.48 && dCr + dCb < 15;
    if (isCastShadow) continue;

    mask[i] = 1;
  }

  // 8 邻域连通域提取
  const visited = new Uint8Array(w * h);
  const queue = new Int32Array(w * h);
  const rawBoxes = [];

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const start = y * w + x;
      if (!mask[start] || visited[start]) continue;

      let head = 0;
      let tail = 0;
      queue[tail++] = start;
      visited[start] = 1;

      let minX = x, maxX = x, minY = y, maxY = y, area = 0;
      while (head < tail) {
        const idx = queue[head++];
        const cx = idx % w;
        const cy = (idx - cx) / w;
        area++;
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        for (let oy = -1; oy <= 1; oy++) {
          const ny = cy + oy;
          if (ny < 0 || ny >= h) continue;
          for (let ox = -1; ox <= 1; ox++) {
            const nx = cx + ox;
            if (nx < 0 || nx >= w) continue;
            const nIdx = ny * w + nx;
            if (mask[nIdx] && !visited[nIdx]) {
              visited[nIdx] = 1;
              queue[tail++] = nIdx;
            }
          }
        }
      }

      if (area >= 5) {
        rawBoxes.push({ minX, minY, maxX, maxY, area });
      }
    }
  }

  // 合并上下邻近同目标部件（例如巡检员身体与头顶安全帽）
  let merged = true;
  while (merged && rawBoxes.length > 1) {
    merged = false;
    for (let i = 0; i < rawBoxes.length; i++) {
      for (let j = i + 1; j < rawBoxes.length; j++) {
        const a = rawBoxes[i];
        const b = rawBoxes[j];
        const cxA = (a.minX + a.maxX) * 0.5;
        const cxB = (b.minX + b.maxX) * 0.5;
        const horizOverlap = a.minX <= b.maxX + 2 && b.minX <= a.maxX + 2;
        const horizClose = Math.abs(cxA - cxB) < 6 || horizOverlap;
        const vertGap = Math.max(0, Math.max(a.minY - b.maxY, b.minY - a.maxY));

        if (horizClose && vertGap <= 9) {
          a.minX = Math.min(a.minX, b.minX);
          a.minY = Math.min(a.minY, b.minY);
          a.maxX = Math.max(a.maxX, b.maxX);
          a.maxY = Math.max(a.maxY, b.maxY);
          a.area += b.area;
          rawBoxes.splice(j, 1);
          merged = true;
          break;
        }
      }
      if (merged) break;
    }
  }

  const sx = fullW / w;
  const sy = fullH / h;
  const pad = 0.45;

  return rawBoxes
    .filter((b) => b.area >= 8)
    .map((b) => [
      Math.max(0, (b.minX - pad) * sx),
      Math.max(0, (b.minY - pad) * sy),
      Math.min(fullW, (b.maxX + 1 + pad) * sx),
      Math.min(fullH, (b.maxY + 1 + pad) * sy)
    ]);
}

const BILLBOARD_VERT = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const BILLBOARD_FRAG = `
  uniform sampler2D uCamVideoTex;
  uniform sampler2D uCamBgTex;
  uniform sampler2D uMaskAtlasTex;
  uniform vec4 uCropRect;        // (uMin, vMinGL, uMax, vMaxGL)
  uniform vec4 uMaskTile;        // (offsetX, offsetY, scaleX, scaleY) in 2x3 AI mask atlas
  uniform vec4 uCamColorAdjust;  // (brightness, contrast, saturation, 0)
  uniform float uExposureGain;
  uniform float uKeyThreshold;
  uniform float uHasExactTrack;
  uniform float uHasMaskAtlas;
  varying vec2 vUv;

  vec3 applyColorAdjustSrgb(vec3 srgbCol, vec4 adj) {
    srgbCol += adj.x;
    srgbCol = (srgbCol - 0.5) * adj.y + 0.5;
    float luma = dot(srgbCol, vec3(0.2126, 0.7152, 0.0722));
    srgbCol = mix(vec3(luma), srgbCol, adj.z);
    return clamp(srgbCol, 0.0, 1.0);
  }

  // 计算单个采样点的原始前景置信度（精确区分深黑实体与地面半透明投影暗影）
  float evalFgConfidence(vec2 uv, vec2 localUv) {
    vec3 fg = texture2D(uCamVideoTex, uv).rgb;
    vec3 rawBg = texture2D(uCamBgTex, uv).rgb;
    vec3 bg = clamp(pow(max(rawBg, vec3(0.0)), vec3(1.0 / 2.2)) * uExposureGain, 0.0, 1.0);

    float maxDiff = max(abs(fg.r - bg.r), max(abs(fg.g - bg.g), abs(fg.b - bg.b)));
    float lumFg = dot(fg, vec3(0.299, 0.587, 0.114));
    float lumBg = dot(bg, vec3(0.299, 0.587, 0.114));
    float chromaDiff = abs((fg.r - lumFg) - (bg.r - lumBg)) + abs((fg.b - lumFg) - (bg.b - lumBg));

    // 地面投影暗影仅存在于脚底附近 (localUv.y < 0.30)、亮度位于背景 52%~93% 且无色度变化
    bool isCastShadow =
      (localUv.y < 0.30) &&
      (lumFg < lumBg * 0.93) &&
      (lumFg > lumBg * 0.52) &&
      (chromaDiff < 0.045);

    if (isCastShadow) return 0.0;

    // 当具备精确全身 3D 检测框时，对躯干中轴区降低抠像门槛，确保同色系灰衣/深色机械关节 100% 完整不破洞
    vec2 spineDist = (localUv - vec2(0.5, 0.52)) / vec2(0.26, 0.42);
    float inSpine = (uHasExactTrack > 0.5) ? (1.0 - smoothstep(0.65, 1.15, length(spineDist))) : 0.0;
    float effThresh = mix(uKeyThreshold * 0.85, uKeyThreshold * 0.48, inSpine);
    return smoothstep(effThresh, effThresh + 0.05, maxDiff);
  }

  void main() {
    vec2 camUV = mix(uCropRect.xy, uCropRect.zw, vUv);
    vec2 cropSpan = max(uCropRect.zw - uCropRect.xy, vec2(1e-4));

    // WebGL2 下 VideoTexture 始终以原始 sRGB 上传
    vec3 fgSrgb = texture2D(uCamVideoTex, camUV).rgb;
    float centerAlpha = evalFgConfidence(camUV, vUv);

    // 精细小半径十字桥接填洞：仅补全躯干内部像素级小洞，严格保持原视频中的真实人体轮廓
    vec2 stepS = cropSpan * vec2(0.025, 0.020);
    vec2 stepL = cropSpan * vec2(0.055, 0.045);

    float aL1 = evalFgConfidence(clamp(camUV - vec2(stepS.x, 0.0), uCropRect.xy, uCropRect.zw), vUv - vec2(0.025, 0.0));
    float aR1 = evalFgConfidence(clamp(camUV + vec2(stepS.x, 0.0), uCropRect.xy, uCropRect.zw), vUv + vec2(0.025, 0.0));
    float aD1 = evalFgConfidence(clamp(camUV - vec2(0.0, stepS.y), uCropRect.xy, uCropRect.zw), vUv - vec2(0.0, 0.020));
    float aU1 = evalFgConfidence(clamp(camUV + vec2(0.0, stepS.y), uCropRect.xy, uCropRect.zw), vUv + vec2(0.0, 0.020));

    float aL2 = evalFgConfidence(clamp(camUV - vec2(stepL.x, 0.0), uCropRect.xy, uCropRect.zw), vUv - vec2(0.055, 0.0));
    float aR2 = evalFgConfidence(clamp(camUV + vec2(stepL.x, 0.0), uCropRect.xy, uCropRect.zw), vUv + vec2(0.055, 0.0));
    float aD2 = evalFgConfidence(clamp(camUV - vec2(0.0, stepL.y), uCropRect.xy, uCropRect.zw), vUv - vec2(0.0, 0.045));
    float aU2 = evalFgConfidence(clamp(camUV + vec2(0.0, stepL.y), uCropRect.xy, uCropRect.zw), vUv + vec2(0.0, 0.045));

    float horizBridge = smoothstep(0.42, 0.85, max(min(aL1, aR1), min(max(aL1, aL2), max(aR1, aR2))));
    float vertBridge = smoothstep(0.42, 0.85, max(min(aD1, aU1), min(max(aD1, aD2), max(aU1, aU2))));
    float holeFill = max(horizBridge, vertBridge);

    // 保留锐利真实的原轮廓边缘，仅内部实心补洞；若存在离线 AI 分割掩膜图集则优先按语义掩膜实现绿幕级完整抠像
    float diffAlpha = max(centerAlpha, holeFill);
    float alpha = diffAlpha;
    if (uHasMaskAtlas > 0.5) {
      vec2 atlasUV = uMaskTile.xy + camUV * uMaskTile.zw;
      float aiMask = texture2D(uMaskAtlasTex, atlasUV).g;
      // AI 语义分割完整保留灰衣/黑机器人躯干，外侧与高置信度背景差分融合防帧间半帧延迟切边
      alpha = smoothstep(0.28, 0.68, max(aiMask, diffAlpha * 0.88));
    }
    vec2 edge = min(vUv, 1.0 - vUv);
    alpha *= smoothstep(0.0, 0.025, min(edge.x, edge.y));
    if (alpha < 0.26) discard;

    vec3 adjustedSrgb = applyColorAdjustSrgb(fgSrgb, uCamColorAdjust);
    vec3 finalLinear = pow(adjustedSrgb, vec3(2.2));
    gl_FragColor = vec4(finalLinear, clamp(alpha * 1.1, 0.0, 1.0));
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

/**
 * 3D 视频切片立牌管理器：
 * 从多路监控实时视频中检测运动前景，反投影到三维地形/高台并跨镜去重，
 * 在唯一真实落脚点竖起带背景差分抠像的 3D 视频立牌，消除地面 V 字躺平双影。
 */
export class VideoBillboardManager {
  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'VideoBillboards';
    this.visible = true;
    this.keyThreshold = 0.095;

    this._unitPlaneGeo = new THREE.PlaneGeometry(1, 1);
    this._unitPlaneGeo.translate(0, 0.5, 0); // 底部中心为原点（脚底落地）
    this._ringGeo = new THREE.RingGeometry(0.30, 0.40, 28);
    this._ringGeo.rotateX(-Math.PI / 2);

    this._pool = [];
    this._ringPool = [];
    this._labelPool = [];
    this._labelTexCache = new Map();
    this._tracks = [];
    this._externalTracks = null;
    this._extCamChoice = new Map(); // objId -> { camId, lastSwitchMs }
    this._maskVideo = null;
    this._maskTexture = null;
    this._maskCamTiles = new Map(); // camId -> [ox, oy, sx, sy]
    this._emptyMaskTex = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
    this._emptyMaskTex.needsUpdate = true;
    this._bgCache = new Map(); // camId -> { sourceEl, rgba }
    this._lastDetectMs = 0;

    this._ray = new THREE.Ray();
    this._dir = new THREE.Vector3();
    this._vPlane = new THREE.Plane();
    this._planeNormal = new THREE.Vector3();
    this._tmpHitA = new THREE.Vector3();
    this._tmpHitB = new THREE.Vector3();

    if (typeof document !== 'undefined') {
      this._detCanvas = document.createElement('canvas');
      this._detCanvas.width = DET_W;
      this._detCanvas.height = DET_H;
      this._detCtx = this._detCanvas.getContext('2d', { willReadFrequently: true });
    }
  }

  setVisible(visible) {
    this.visible = visible;
    this.group.visible = visible;
    if (!visible) {
      this._pool.forEach((m) => (m.visible = false));
      this._ringPool.forEach((r) => (r.visible = false));
      this._labelPool.forEach((l) => (l.visible = false));
    }
  }

  setExternalTracks(tracksData) {
    this._externalTracks = tracksData;
    this._extCamChoice.clear();
    this._tracks = [];
  }

  /**
   * 绑定 2x3 拼合的离线 AI 实例分割掩膜视频 (如 YOLOv11-seg 生成的 masks_w16.mp4)
   */
  setMaskAtlas(url, camOrder = ['camera', 'camera_01', 'camera_02', 'camera_03', 'camera_05', 'camera_06']) {
    this._disposeMaskAtlas();
    if (!url || typeof document === 'undefined') return;

    const vid = document.createElement('video');
    vid.crossOrigin = 'anonymous';
    vid.loop = true;
    vid.muted = true;
    vid.playsInline = true;
    vid.preload = 'auto';
    vid.src = url;
    vid.play().catch(() => {});

    const tex = new THREE.VideoTexture(vid);
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;

    this._maskVideo = vid;
    this._maskTexture = tex;
    this._maskCamTiles.clear();
    // 2 列 x 3 行网格 (col=0..1, row=0..2 从顶到底；OpenGL V 轴自底向上，因此顶行 row=0 对应 oy = 2/3)
    const cols = 2;
    const rows = 3;
    camOrder.forEach((cid, idx) => {
      const col = idx % cols;
      const row = Math.floor(idx / cols);
      const ox = col / cols;
      const oy = (rows - 1 - row) / rows;
      this._maskCamTiles.set(cid, [ox, oy, 1 / cols, 1 / rows]);
    });
  }

  _disposeMaskAtlas() {
    if (this._maskVideo) {
      this._maskVideo.pause();
      this._maskVideo.removeAttribute('src');
      this._maskVideo.load?.();
      this._maskVideo = null;
    }
    if (this._maskTexture) {
      this._maskTexture.dispose();
      this._maskTexture = null;
    }
    this._maskCamTiles.clear();
  }

  clear() {
    this._tracks = [];
    this._externalTracks = null;
    this._extCamChoice.clear();
    this._disposeMaskAtlas();
    this._bgCache.clear();
    this._pool.forEach((m) => (m.visible = false));
    this._ringPool.forEach((r) => (r.visible = false));
    this._labelPool.forEach((l) => (l.visible = false));
  }

  _getBgRgba(camId, bgSourceEl) {
    const cached = this._bgCache.get(camId);
    if (cached && cached.sourceEl === bgSourceEl) return cached.rgba;
    this._detCtx.drawImage(bgSourceEl, 0, 0, DET_W, DET_H);
    const rgba = new Uint8ClampedArray(this._detCtx.getImageData(0, 0, DET_W, DET_H).data);
    this._bgCache.set(camId, { sourceEl: bgSourceEl, rgba });
    return rgba;
  }

  _intersectVerticalPlane(proj, u, v, verticalPlane, outVec) {
    const { xn, yn } = proj._undistortPixel(u, v);
    this._dir.set(xn, -yn, -1).normalize().applyQuaternion(proj.camera.quaternion).normalize();
    this._ray.origin.copy(proj.camera.position);
    this._ray.direction.copy(this._dir);
    return this._ray.intersectPlane(verticalPlane, outVec);
  }

  _acquireMesh(index) {
    while (this._pool.length <= index) {
      const mat = new THREE.ShaderMaterial({
        vertexShader: BILLBOARD_VERT,
        fragmentShader: BILLBOARD_FRAG,
        uniforms: {
          uCamVideoTex: { value: null },
          uCamBgTex: { value: null },
          uMaskAtlasTex: { value: this._emptyMaskTex },
          uCropRect: { value: new THREE.Vector4(0, 0, 1, 1) },
          uMaskTile: { value: new THREE.Vector4(0, 0, 1, 1) },
          uCamColorAdjust: { value: new THREE.Vector4(0, 1, 1, 0) },
          uExposureGain: { value: 1.0 },
          uKeyThreshold: { value: this.keyThreshold },
          uHasExactTrack: { value: 0.0 },
          uHasMaskAtlas: { value: 0.0 }
        },
        side: THREE.DoubleSide,
        transparent: true,
        depthWrite: true,
        depthTest: true
      });
      const mesh = new THREE.Mesh(this._unitPlaneGeo, mat);
      mesh.rotation.order = 'YXZ';
      mesh.frustumCulled = false;
      mesh.visible = false;
      this.group.add(mesh);
      this._pool.push(mesh);
    }
    return this._pool[index];
  }

  _acquireRing(index) {
    while (this._ringPool.length <= index) {
      const mat = new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.85,
        side: THREE.DoubleSide,
        depthWrite: false
      });
      const ring = new THREE.Mesh(this._ringGeo, mat);
      ring.frustumCulled = false;
      ring.visible = false;
      this.group.add(ring);
      this._ringPool.push(ring);
    }
    return this._ringPool[index];
  }

  _getLabelTexture(cls, id) {
    const key = `${cls}_${id}`;
    const cached = this._labelTexCache.get(key);
    if (cached) return cached;

    const canvas = document.createElement('canvas');
    canvas.width = 220;
    canvas.height = 56;
    const ctx = canvas.getContext('2d');

    const style =
      cls === 'robot'
        ? { bg: 'rgba(234, 88, 12, 0.90)', border: '#fdba74', text: `机器人 #${id}` }
        : cls === 'agv'
          ? { bg: 'rgba(101, 163, 13, 0.90)', border: '#d9f99d', text: `AGV #${id}` }
          : { bg: 'rgba(2, 132, 199, 0.90)', border: '#7dd3fc', text: `工人 #${id}` };

    ctx.fillStyle = style.bg;
    ctx.strokeStyle = style.border;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.roundRect(6, 6, 208, 44, 12);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 26px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(style.text, 110, 29);

    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    this._labelTexCache.set(key, tex);
    return tex;
  }

  _acquireLabel(index) {
    while (this._labelPool.length <= index) {
      const mat = new THREE.SpriteMaterial({
        transparent: true,
        depthTest: false,
        depthWrite: false
      });
      const sprite = new THREE.Sprite(mat);
      sprite.frustumCulled = false;
      sprite.visible = false;
      this.group.add(sprite);
      this._labelPool.push(sprite);
    }
    return this._labelPool[index];
  }

  /**
   * 外部高精度 3D 轨迹模式（如 AI 3D 检测器 / NVIDIA ground_truth）：
   * 毫米级 3D 精确落位 + 完整全身 2D 裁切 + 区分普通工人 / 人形机器人 / AGV
   */
  _updateFromExternalTracks(nowMs, logicalCameras, streamMgr) {
    const ext = this._externalTracks;
    const frames = ext?.frames;
    if (!frames || !frames.length) return;

    const masterTime = streamMgr.getMasterMp4Time?.();
    if (masterTime === null || masterTime === undefined) return;

    const fps = ext.fps || 15;
    const fFloat = (masterTime % (ext.duration || 20)) * fps;
    const idx0 = Math.floor(fFloat) % frames.length;
    const idx1 = (idx0 + 1) % frames.length;
    const frac = fFloat - Math.floor(fFloat);

    const list0 = frames[idx0] || [];
    const list1 = frames[idx1] || [];
    const map1 = new Map(list1.map((o) => [o.id, o]));

    const camMap = new Map(logicalCameras.map((c) => [c.id, c]));
    const nextTracks = [];

    for (const o0 of list0) {
      const o1 = map1.get(o0.id);
      const x = o1 ? o0.p[0] + (o1.p[0] - o0.p[0]) * frac : o0.p[0];
      const y = o1 ? o0.p[1] + (o1.p[1] - o0.p[1]) * frac : o0.p[1];
      const z = o1 ? o0.p[2] + (o1.p[2] - o0.p[2]) * frac : o0.p[2];
      const worldH = Math.max(0.35, o0.s[1]);

      // 在当前活跃且正在播放的相机中，优选低俯角、全身完整可见的裁切视角
      const candidates = [];
      for (const [camId, crop0] of Object.entries(o0.b || {})) {
        const camItem = camMap.get(camId);
        if (!camItem) continue;
        const pair = streamMgr.getLiveForegroundPair(camId);
        if (!pair || !pair.bgTexture || pair.videoEl.readyState < 2) continue;

        const crop1 = o1?.b?.[camId];
        const crop = crop1
          ? [
              crop0[0] + (crop1[0] - crop0[0]) * frac,
              crop0[1] + (crop1[1] - crop0[1]) * frac,
              crop0[2] + (crop1[2] - crop0[2]) * frac,
              crop0[3] + (crop1[3] - crop0[3]) * frac
            ]
          : crop0;

        const cropW = Math.max(0.005, crop[2] - crop[0]);
        const cropH = Math.max(0.005, crop[3] - crop[1]);
        const fullW = camItem.intrinsics.width || 1280;
        const fullH = camItem.intrinsics.height || 720;
        const cropAspect = (cropW * fullW) / Math.max(1, cropH * fullH);

        const distToCam = Math.hypot(
          camItem.camera.position.x - x,
          camItem.camera.position.y - y,
          camItem.camera.position.z - z
        );
        const camHoriz = Math.hypot(camItem.camera.position.x - x, camItem.camera.position.z - z);
        const camDrop = Math.max(0.1, camItem.camera.position.y - y);
        const pitchDeg = THREE.MathUtils.radToDeg(Math.atan2(camDrop, Math.max(0.1, camHoriz)));
        const pitchPenalty = 1.0 + Math.pow(Math.max(0, pitchDeg - 30) / 16, 2.0);

        // 计算世界竖直铅锤线 (+Y) 在该相机图像平面上的倾斜角：既用于惩罚大倾斜角相机，也用于把歪斜人形扶正
        this._tmpHitA.set(x, y, z).applyMatrix4(camItem.camera.matrixWorldInverse);
        this._tmpHitB.set(x, y + worldH, z).applyMatrix4(camItem.camera.matrixWorldInverse);
        let leanRad = 0.0;
        if (this._tmpHitA.z < -0.2 && this._tmpHitB.z < -0.2) {
          const fx = camItem.intrinsics.fx || 900;
          const fy = camItem.intrinsics.fy || 900;
          const uFoot = (this._tmpHitA.x / -this._tmpHitA.z) * fx;
          const vFoot = (this._tmpHitA.y / this._tmpHitA.z) * fy;
          const uHead = (this._tmpHitB.x / -this._tmpHitB.z) * fx;
          const vHead = (this._tmpHitB.y / this._tmpHitB.z) * fy;
          const du = uHead - uFoot;
          const dvUp = vFoot - vHead;
          if (Math.hypot(du, dvUp) > 2.0) {
            leanRad = Math.atan2(du, Math.max(1.0, dvUp));
          }
        }
        const leanDeg = Math.abs(THREE.MathUtils.radToDeg(leanRad));
        const leanPenalty = 1.0 + Math.pow(leanDeg / 18.0, 2.0);

        // 偏向全身裁切更高的视角（避开被托盘遮掉一半腿的视角）
        const sizeBonus = 1.0 / Math.max(0.12, Math.sqrt(cropH));
        const viewCost = distToCam * pitchPenalty * leanPenalty * sizeBonus;

        candidates.push({
          camId,
          crop,
          cropAspect,
          leanRad,
          viewCost,
          pair,
          colorAdjust: camItem.colorAdjust
        });
      }

      if (!candidates.length) continue;
      candidates.sort((a, b) => a.viewCost - b.viewCost);

      // 900ms 相机源滞回锁定，防止在多镜重叠区来回切视角闪烁
      let chosen = candidates[0];
      const prevChoice = this._extCamChoice.get(o0.id);
      if (prevChoice) {
        const sameCand = candidates.find((c) => c.camId === prevChoice.camId);
        if (sameCand) {
          const shouldSwitch =
            chosen.camId !== sameCand.camId &&
            chosen.viewCost < sameCand.viewCost * 0.72 &&
            nowMs - prevChoice.lastSwitchMs > 900;
          if (!shouldSwitch) {
            chosen = sameCand;
          } else {
            this._extCamChoice.set(o0.id, { camId: chosen.camId, lastSwitchMs: nowMs });
          }
        } else {
          this._extCamChoice.set(o0.id, { camId: chosen.camId, lastSwitchMs: nowMs });
        }
      } else {
        this._extCamChoice.set(o0.id, { camId: chosen.camId, lastSwitchMs: nowMs });
      }

      const worldW = THREE.MathUtils.clamp(worldH * chosen.cropAspect, 0.35, 2.4);
      const rollZ = THREE.MathUtils.clamp(chosen.leanRad || 0, -0.52, 0.52);
      nextTracks.push({
        id: o0.id,
        cls: o0.cls,
        exactTrack: true,
        rollZ,
        camId: chosen.camId,
        x,
        y,
        z,
        width: worldW,
        height: worldH,
        crop: chosen.crop,
        videoTexture: chosen.pair.videoTexture,
        bgTexture: chosen.pair.bgTexture,
        colorAdjust: chosen.colorAdjust,
        exposureGain: 1.0,
        hits: 10,
        lastSeen: nowMs
      });
    }

    this._tracks = nextTracks;
  }

  /**
   * 主渲染循环调用：若加载了外部高精度 3D 轨迹则按 60Hz 同步插值，否则以 ~30Hz 视觉检测前景
   */
  update(nowMs, logicalCameras, bboxProjectors, streamMgr, mainCamera) {
    if (!this.visible || !this._detCtx) return;

    if (this._externalTracks) {
      // 同步 AI 分割掩膜图集视频时钟到主 MP4 时钟
      if (this._maskVideo && this._maskVideo.readyState >= 2 && Number.isFinite(this._maskVideo.duration)) {
        const mt = streamMgr.getMasterMp4Time?.();
        if (mt !== null && mt !== undefined) {
          const target = mt % this._maskVideo.duration;
          const diff = Math.abs(this._maskVideo.currentTime - target);
          if (Math.min(diff, this._maskVideo.duration - diff) > 0.18) {
            this._maskVideo.currentTime = target;
          }
          if (this._maskVideo.paused) this._maskVideo.play().catch(() => {});
        }
      }
      this._updateFromExternalTracks(nowMs, logicalCameras, streamMgr);
    } else if (nowMs - this._lastDetectMs >= 33) {
      this._lastDetectMs = nowMs;
      this._detectAndCluster(nowMs, logicalCameras, bboxProjectors, streamMgr);
    }

    // 每帧更新立牌朝向，保持底部着地并平滑面向主相机
    let poolIdx = 0;
    let badgeIdx = 0;
    for (let i = 0; i < this._tracks.length; i++) {
      const tr = this._tracks[i];
      // 连续命中 2 帧以上才显示，彻底消除单帧噪点闪烁
      if (tr.hits < 2) continue;
      const mesh = this._acquireMesh(poolIdx++);
      mesh.visible = true;
      mesh.position.set(tr.x, tr.y, tr.z);
      mesh.scale.set(tr.width, tr.height, 1);

      const dx = mainCamera.position.x - tr.x;
      const dz = mainCamera.position.z - tr.z;
      const dy = mainCamera.position.y - (tr.y + tr.height * 0.5);
      const yaw = Math.atan2(dx, dz);
      const horizDist = Math.hypot(dx, dz);
      const pitchToCam = Math.atan2(dy, Math.max(horizDist, 0.01));
      // 保持人物自然直立于地面，并自动回正高空斜角相机的铅锤倾斜角 rollZ
      const tiltX = -Math.min(Math.max(pitchToCam, 0) * 0.20, 0.22);
      const rollZ = tr.rollZ || 0;
      mesh.rotation.set(tiltX, yaw, rollZ);
      // 绕底部中心旋转 rollZ 时，微调水平基座使人物重心与脚底严格落回光环正中
      const rightX = Math.cos(yaw);
      const rightZ = -Math.sin(yaw);
      const shift = Math.sin(rollZ) * tr.height * 0.42;
      mesh.position.set(tr.x + rightX * shift, tr.y, tr.z + rightZ * shift);

      const u = mesh.material.uniforms;
      u.uCamVideoTex.value = tr.videoTexture;
      u.uCamBgTex.value = tr.bgTexture;
      u.uCropRect.value.set(tr.crop[0], tr.crop[1], tr.crop[2], tr.crop[3]);
      u.uCamColorAdjust.value.set(
        tr.colorAdjust.brightness ?? 0,
        tr.colorAdjust.contrast ?? 1,
        tr.colorAdjust.saturation ?? 1,
        0
      );
      u.uExposureGain.value = tr.exposureGain ?? 1.0;
      u.uKeyThreshold.value = this.keyThreshold;
      u.uHasExactTrack.value = tr.exactTrack ? 1.0 : 0.0;

      const tile = tr.camId ? this._maskCamTiles.get(tr.camId) : null;
      if (tile && this._maskTexture && this._maskVideo && this._maskVideo.readyState >= 2) {
        u.uMaskAtlasTex.value = this._maskTexture;
        u.uMaskTile.value.set(tile[0], tile[1], tile[2], tile[3]);
        u.uHasMaskAtlas.value = 1.0;
      } else {
        u.uMaskAtlasTex.value = this._emptyMaskTex;
        u.uHasMaskAtlas.value = 0.0;
      }

      // 区分普通工人 / 人形机器人 / AGV：脚底彩色身份光环 + 顶部类型标签
      if (tr.cls && tr.id !== undefined) {
        const ring = this._acquireRing(badgeIdx);
        ring.visible = true;
        ring.position.set(tr.x, tr.y + 0.03, tr.z);
        const ringColor = tr.cls === 'robot' ? 0xf97316 : tr.cls === 'agv' ? 0xa3e635 : 0x38bdf8;
        ring.material.color.setHex(ringColor);
        const rScale = tr.cls === 'agv' ? 1.55 : 1.0;
        ring.scale.set(rScale, 1, rScale);

        const label = this._acquireLabel(badgeIdx);
        label.visible = true;
        label.material.map = this._getLabelTexture(tr.cls, tr.id);
        label.material.needsUpdate = true;
        label.position.set(tr.x, tr.y + tr.height + 0.28, tr.z);
        label.scale.set(1.15, 0.30, 1);
        badgeIdx++;
      }
    }

    for (let i = poolIdx; i < this._pool.length; i++) {
      this._pool[i].visible = false;
    }
    for (let i = badgeIdx; i < this._ringPool.length; i++) {
      this._ringPool[i].visible = false;
    }
    for (let i = badgeIdx; i < this._labelPool.length; i++) {
      this._labelPool[i].visible = false;
    }
  }

  _detectAndCluster(nowMs, logicalCameras, bboxProjectors, streamMgr) {
    const observations = [];

    for (const item of logicalCameras) {
      const pair = streamMgr.getLiveForegroundPair(item.id);
      if (!pair || !pair.bgSourceEl || !pair.bgTexture || pair.videoEl.readyState < 2) continue;

      const proj = bboxProjectors.get(item.id);
      if (!proj) continue;

      const bgRgba = this._getBgRgba(item.id, pair.bgSourceEl);
      this._detCtx.drawImage(pair.videoEl, 0, 0, DET_W, DET_H);
      const vidRgba = this._detCtx.getImageData(0, 0, DET_W, DET_H).data;

      const fullW = item.intrinsics.width;
      const fullH = item.intrinsics.height;
      const exposureGain = estimateExposureGain(vidRgba, bgRgba);
      const detDiffThresh = Math.max(24, Math.round(this.keyThreshold * 240));
      const bboxes = extractForegroundBlobsFromRgba(vidRgba, bgRgba, DET_W, DET_H, fullW, fullH, detDiffThresh);

      for (const bbox of bboxes) {
        const hit = proj.projectToScene(bbox, 0);
        if (!hit) continue;

        const distToCam = hit.distanceTo(item.camera.position);
        if (distToCam < 1.0 || distToCam > (item.camera.far || 60) * 1.1) continue;

        // 构造过脚底 hit 且水平朝向相机的竖直平面，精确解算真实三维宽高（消除俯仰/滚转/畸变误差）
        this._planeNormal.set(item.camera.position.x - hit.x, 0, item.camera.position.z - hit.z);
        if (this._planeNormal.lengthSq() < 1e-6) this._planeNormal.set(0, 0, 1);
        this._planeNormal.normalize();
        this._vPlane.setFromNormalAndCoplanarPoint(this._planeNormal, hit);

        const [u1, v1, u2, v2] = bbox;
        const uMid = (u1 + u2) * 0.5;
        const cropPxW = Math.max(1, u2 - u1);
        const cropPxH = Math.max(1, v2 - v1);
        const cropAspect = cropPxW / cropPxH;

        // 计算相机对该目标的俯拍角：优先选用俯角平缓的侧拍视角，避开正上方大俯角透视变形
        const camHoriz = Math.hypot(item.camera.position.x - hit.x, item.camera.position.z - hit.z);
        const camDrop = Math.max(0.1, item.camera.position.y - hit.y);
        const pitchDeg = THREE.MathUtils.radToDeg(Math.atan2(camDrop, Math.max(0.1, camHoriz)));
        const pitchPenalty = 1.0 + Math.pow(Math.max(0, pitchDeg - 30) / 16, 2.0);
        const viewCost = distToCam * pitchPenalty;

        let worldH = 1.65;
        if (this._intersectVerticalPlane(proj, uMid, v1, this._vPlane, this._tmpHitA)) {
          worldH = THREE.MathUtils.clamp(this._tmpHitA.y - hit.y, 0.55, 2.25);
        }
        // 严格按原视频裁切区 1:1 像素宽高比计算物理宽度，确保人物绝不拉伸或压扁
        const worldW = THREE.MathUtils.clamp(worldH * cropAspect, 0.35, 2.6);

        observations.push({
          camId: item.id,
          x: hit.x,
          y: hit.y,
          z: hit.z,
          width: worldW,
          height: worldH,
          distToCam,
          viewCost,
          crop: [u1 / fullW, 1.0 - v2 / fullH, u2 / fullW, 1.0 - v1 / fullH],
          videoTexture: pair.videoTexture,
          bgTexture: pair.bgTexture,
          colorAdjust: item.colorAdjust,
          exposureGain
        });
      }
    }

    // 跨镜水平距离聚类去重（重叠区多路相机合并为 1 个唯一 3D 实体，优先选用低俯角清晰侧拍视角）
    const clusters = [];
    for (const obs of observations) {
      const w = 1.0 / Math.max(obs.viewCost, 1.0);
      const target = clusters.find(
        (c) => !c.obsByCam.has(obs.camId) && Math.hypot(c.x - obs.x, c.z - obs.z) <= 2.1
      );
      if (!target) {
        const obsByCam = new Map([[obs.camId, obs]]);
        clusters.push({
          x: obs.x,
          y: obs.y,
          z: obs.z,
          width: obs.width,
          height: obs.height,
          totalWeight: w,
          obsByCam,
          bestObs: obs
        });
      } else {
        const nw = target.totalWeight + w;
        target.x = (target.x * target.totalWeight + obs.x * w) / nw;
        target.y = (target.y * target.totalWeight + obs.y * w) / nw;
        target.z = (target.z * target.totalWeight + obs.z * w) / nw;
        target.totalWeight = nw;
        target.obsByCam.set(obs.camId, obs);
        if (obs.viewCost < target.bestObs.viewCost) {
          target.bestObs = obs;
          target.width = obs.width;
          target.height = obs.height;
        }
      }
    }

    // 与上一帧立牌轨迹匹配：加入相机源滞回锁定 + UV 裁剪框平滑 + 420ms 遮挡保持，根治多镜切换闪烁
    const nextTracks = [];
    const usedPrev = new Set();
    for (const c of clusters) {
      let bestPrevIdx = -1;
      let bestDist = 2.4;
      for (let i = 0; i < this._tracks.length; i++) {
        if (usedPrev.has(i)) continue;
        const d = Math.hypot(this._tracks[i].x - c.x, this._tracks[i].z - c.z);
        if (d < bestDist) {
          bestDist = d;
          bestPrevIdx = i;
        }
      }

      if (bestPrevIdx >= 0) {
        usedPrev.add(bestPrevIdx);
        const prev = this._tracks[bestPrevIdx];
        const prevCamObs = c.obsByCam.get(prev.camId);
        let chosenObs = c.bestObs;
        let lastCamSwitchMs = prev.lastCamSwitchMs ?? 0;

        // 滞回锁定：若原相机仍能看到目标，除非新相机综合视角代价低 25% 以上且距上次切换已过 900ms，否则不切视角
        if (prevCamObs) {
          const canSwitch =
            c.bestObs.camId !== prev.camId &&
            c.bestObs.viewCost < prevCamObs.viewCost * 0.75 &&
            nowMs - lastCamSwitchMs > 900;
          if (canSwitch) {
            chosenObs = c.bestObs;
            lastCamSwitchMs = nowMs;
          } else {
            chosenObs = prevCamObs;
          }
        } else {
          lastCamSwitchMs = nowMs;
        }

        const sameCam = chosenObs.camId === prev.camId;
        const a = 0.40;
        const crop = sameCam
          ? [
              prev.crop[0] + (chosenObs.crop[0] - prev.crop[0]) * 0.45,
              prev.crop[1] + (chosenObs.crop[1] - prev.crop[1]) * 0.45,
              prev.crop[2] + (chosenObs.crop[2] - prev.crop[2]) * 0.45,
              prev.crop[3] + (chosenObs.crop[3] - prev.crop[3]) * 0.45
            ]
          : chosenObs.crop;

        nextTracks.push({
          camId: chosenObs.camId,
          x: prev.x + (c.x - prev.x) * a,
          y: prev.y + (c.y - prev.y) * a,
          z: prev.z + (c.z - prev.z) * a,
          width: prev.width + (chosenObs.width - prev.width) * 0.25,
          height: prev.height + (chosenObs.height - prev.height) * 0.25,
          crop,
          videoTexture: chosenObs.videoTexture,
          bgTexture: chosenObs.bgTexture,
          colorAdjust: chosenObs.colorAdjust,
          exposureGain: chosenObs.exposureGain,
          hits: Math.min(prev.hits + 1, 100),
          lastCamSwitchMs,
          lastSeen: nowMs
        });
      } else {
        nextTracks.push({
          camId: c.bestObs.camId,
          x: c.x,
          y: c.y,
          z: c.z,
          width: c.bestObs.width,
          height: c.bestObs.height,
          crop: c.bestObs.crop,
          videoTexture: c.bestObs.videoTexture,
          bgTexture: c.bestObs.bgTexture,
          colorAdjust: c.bestObs.colorAdjust,
          exposureGain: c.bestObs.exposureGain,
          hits: 1,
          lastCamSwitchMs: nowMs,
          lastSeen: nowMs
        });
      }
    }

    // 短暂遮挡保持 420ms，防止经过货架/托盘边缘时瞬断闪烁
    for (let i = 0; i < this._tracks.length; i++) {
      if (!usedPrev.has(i) && nowMs - this._tracks[i].lastSeen < 420) {
        nextTracks.push(this._tracks[i]);
      }
    }

    this._tracks = nextTracks;
  }

  dispose() {
    this._unitPlaneGeo.dispose();
    for (const mesh of this._pool) {
      mesh.material.dispose();
    }
    this._pool = [];
    this._tracks = [];
    this._bgCache.clear();
  }
}
