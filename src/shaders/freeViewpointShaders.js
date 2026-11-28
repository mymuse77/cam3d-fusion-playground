import { MAX_CAMS } from '../utils/multiProjector.js';
import { GLSL_CAMERA_MODEL } from './injectMultiProjection.js';

/**
 * 生成按槽位静态下标分发的采样函数（GLSL ES 3.0 不允许用循环变量索引采样器数组）
 */
export function glslSlotSampler(fnName, uniformName, retType, swizzle) {
  const branches = Array.from({ length: MAX_CAMS }, (_, i) => {
    const head = i === 0 ? 'if (idx == 0)' : i === MAX_CAMS - 1 ? 'else' : `else if (idx == ${i})`;
    return `    ${head} return textureLod(${uniformName}[${i}], uv, 0.0).${swizzle};`;
  });
  return `  ${retType} ${fnName}(int idx, vec2 uv) {\n${branches.join('\n')}\n  }`;
}

export const SCREEN_QUAD_VERT = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

/**
 * 抠像阶段 A：逐像素前景置信度 (R)、地面投影暗影标记 (G) 与深色歧义标记 (B)
 * 背景差分 + 曝光增益补偿；存在离线 AI 实例分割掩膜时按三分图 (Trimap) 融合：
 * 语义掩膜通常比真实轮廓松一圈，直接取并集会带出一圈地面光晕，
 * 因此掩膜之外判为确定背景（去噪点与远处阴影），掩膜之内由背景差分决定精确轮廓，
 * 其中掩膜腐蚀后的内核区域大幅降低差分门限以补全同色衣物破洞
 * （不直接判为前景：掩膜常把脚下阴影一并圈入，内核里完全没有差异的像素仍是地面）
 */
export const MATTE_RAW_FRAG = `
  uniform sampler2D uVideo;
  uniform sampler2D uBg;
  uniform sampler2D uMask;
  uniform vec4 uMaskTile;   // (offsetX, offsetY, scaleX, scaleY)
  uniform vec2 uMaskTexel;
  uniform float uTrimapPx;  // 三分图过渡带半宽（掩膜像素）
  uniform float uHasMask;
  uniform float uGain;
  uniform float uKey;
  varying vec2 vUv;

  float maskTap(vec2 offPx) {
    vec2 lo = uMaskTile.xy + 0.5 * uMaskTexel;
    vec2 hi = uMaskTile.xy + uMaskTile.zw - 0.5 * uMaskTexel;
    return texture2D(uMask, clamp(uMaskTile.xy + vUv * uMaskTile.zw + offPx * uMaskTexel, lo, hi)).g;
  }

  void main() {
    // VideoTexture 以原始 sRGB 上传；背景底图经硬件解码为 Linear，需转回 sRGB 后再比较
    vec3 fg = texture2D(uVideo, vUv).rgb;
    vec3 bg = clamp(pow(max(texture2D(uBg, vUv).rgb, vec3(0.0)), vec3(1.0 / 2.2)) * uGain, 0.0, 1.0);

    float maxDiff = max(abs(fg.r - bg.r), max(abs(fg.g - bg.g), abs(fg.b - bg.b)));
    float lumFg = dot(fg, vec3(0.299, 0.587, 0.114));
    float lumBg = dot(bg, vec3(0.299, 0.587, 0.114));
    float chromaDiff = abs((fg.r - lumFg) - (bg.r - lumBg)) + abs((fg.b - lumFg) - (bg.b - lumBg));

    float conf = smoothstep(uKey * 0.85, uKey * 0.85 + 0.05, maxDiff);
    // 投影暗影：亮度落在背景 52%~93% 且色度几乎不变（是否剔除由着色阶段按离地高度决定）
    // 深色歧义：比背景暗得多且色度不变，可能是深色衣物/机身，也可能是脚下的接触阴影或地面倒影
    bool lowChroma = chromaDiff < 0.045;
    float shadow = (lowChroma && lumFg < lumBg * 0.93 && lumFg > lumBg * 0.52) ? conf : 0.0;
    float dark = (lowChroma && lumFg <= lumBg * 0.52) ? conf : 0.0;

    if (uHasMask > 0.5) {
      float r = uTrimapPx;
      float d = r * 0.7071;
      float ai = maskTap(vec2(0.0));
      float t0 = maskTap(vec2(r, 0.0)), t1 = maskTap(vec2(-r, 0.0)), t2 = maskTap(vec2(0.0, r)), t3 = maskTap(vec2(0.0, -r));
      float t4 = maskTap(vec2(d, d)), t5 = maskTap(vec2(-d, d)), t6 = maskTap(vec2(d, -d)), t7 = maskTap(vec2(-d, -d));
      float core = min(ai, min(min(min(t0, t1), min(t2, t3)), min(min(t4, t5), min(t6, t7))));
      float gate = smoothstep(0.28, 0.68, ai);
      float sure = smoothstep(0.28, 0.68, core);
      float filled = gate * sure * smoothstep(uKey * 0.4, uKey * 0.4 + 0.04, maxDiff);
      conf = max(gate * conf, filled);
      // 靠低门限补进来的像素若只是略暗且色度不变，同样可能是淡影：打上暗影标记，交由近地面规则处理
      shadow = max(gate * shadow, (lowChroma && lumFg < lumBg && lumFg > lumBg * 0.52) ? filled : 0.0);
      dark *= gate;
    }
    gl_FragColor = vec4(conf, shadow, dark, 1.0);
  }
`;

/**
 * 抠像阶段 B：形态学精修，输出抠像图集
 * R = 精细抠像（内部小洞桥接，决定最终轮廓 Alpha）
 * G = 支撑抠像（成片前景略微外扩并滤除孤立噪点，用于跨相机一致性表决）
 * B = 暗影标记，A = 深色歧义标记
 */
export const MATTE_REFINE_FRAG = `
  uniform sampler2D uRaw;
  uniform vec4 uTileRect;   // 图集内瓦片 (offsetX, offsetY, sizeX, sizeY)
  uniform vec2 uTexel;
  uniform float uDilatePx;
  varying vec2 vUv;

  vec2 tap(vec2 offPx) {
    vec2 uv = uTileRect.xy + vUv * uTileRect.zw + offPx * uTexel;
    uv = clamp(uv, uTileRect.xy + 0.5 * uTexel, uTileRect.xy + uTileRect.zw - 0.5 * uTexel);
    return texture2D(uRaw, uv).rg;
  }

  void main() {
    float r1 = uDilatePx * 0.5;
    float r2 = uDilatePx;
    float d1 = r1 * 0.7071;
    float d2 = r2 * 0.7071;

    vec3 c = texture2D(uRaw, uTileRect.xy + vUv * uTileRect.zw).rgb;
    vec2 e1 = tap(vec2(r1, 0.0)), w1 = tap(vec2(-r1, 0.0)), n1 = tap(vec2(0.0, r1)), s1 = tap(vec2(0.0, -r1));
    vec2 e2 = tap(vec2(r2, 0.0)), w2 = tap(vec2(-r2, 0.0)), n2 = tap(vec2(0.0, r2)), s2 = tap(vec2(0.0, -r2));
    vec2 a1 = tap(vec2(d1, d1)), b1 = tap(vec2(-d1, d1)), g1 = tap(vec2(d1, -d1)), h1 = tap(vec2(-d1, -d1));
    vec2 a2 = tap(vec2(d2, d2)), b2 = tap(vec2(-d2, d2)), g2 = tap(vec2(d2, -d2)), h2 = tap(vec2(-d2, -d2));

    float hBridge = min(max(w1.r, w2.r), max(e1.r, e2.r));
    float vBridge = min(max(s1.r, s2.r), max(n1.r, n2.r));
    float sharp = max(c.r, smoothstep(0.42, 0.85, max(hBridge, vBridge)));

    // 取邻域占比而非最大值：成片前景向外扩，孤立噪点与细线（压缩噪声、地面标线边缘）被滤除
    vec2 sum1 = e1 + w1 + n1 + s1 + a1 + b1 + g1 + h1;
    vec2 sum2 = e2 + w2 + n2 + s2 + a2 + b2 + g2 + h2;
    float support = smoothstep(0.12, 0.3, (c.r + sum1.r + sum2.r) / 17.0);

    gl_FragColor = vec4(sharp, support, c.g, c.b);
  }
`;

/**
 * 抠像图集采样（需要 uMatteAtlas / uAtlasGrid / uAtlasPad）
 */
export const GLSL_SAMPLE_MATTE = `
  vec4 sampleMatte(int idx, vec2 uv) {
    int cols = int(uAtlasGrid.x);
    int row = idx / cols;
    vec2 cell = vec2(float(idx - row * cols), float(row));
    return textureLod(uMatteAtlas, (cell + clamp(uv, uAtlasPad, 1.0 - uAtlasPad)) / uAtlasGrid, 0.0);
  }
`;

/**
 * 静态场景遮挡判定（需要 uProjProjectionMatrix 与 sampleDepthSlot）：
 * 该点是否被静态场景挡住（线性深度比较，偏移随距离增长，避免透视深度在远处失去分辨力）
 */
export const GLSL_STATIC_OCCLUDED = `
  bool staticOccluded(int i, vec4 viewPos) {
    mat4 pm = uProjProjectionMatrix[i];
    vec4 clip = pm * viewPos;
    vec3 ndc = clip.xyz / clip.w;
    vec2 duv = ndc.xy * 0.5 + 0.5;
    if (duv.x < 0.0 || duv.x > 1.0 || duv.y < 0.0 || duv.y > 1.0) return false;
    float sceneDist = pm[3][2] / (sampleDepthSlot(i, duv) * 2.0 - 1.0 + pm[2][2]);
    return -viewPos.z > sceneDist + 0.12 + 0.01 * sceneDist;
  }
`;

export const MAX_TARGETS = 24;

export const CARD_VERT = `
  varying vec3 vWorldPos;
  varying vec2 vUv;
  void main() {
    vUv = uv;
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vWorldPos = wp.xyz;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;

/**
 * 自由视点前景片元着色器：
 * 面片上每个三维点按真实标定参数回投到各路相机，取实时视频颜色与抠像 Alpha，
 * 再按 CPU 为该目标选定的主相机（及切换时淡出的前一路相机）权重着色，其余相机只在主相机弃权处补全。
 * 借助全部目标的三维位置做像素归属判定：某路相机的视线若先穿过别的目标，该像素拍到的其实是别人，
 * 此时该相机对这一点弃权，由其他相机补全，从而消除目标之间相互串入的重影。
 */
export const CARD_FRAG = `
  #define MAX_CAMS ${MAX_CAMS}
  #define MAX_TARGETS ${MAX_TARGETS}

  uniform int uActiveCams;
  uniform sampler2D uLiveTex[MAX_CAMS];
  uniform float uLiveOn[MAX_CAMS];
  uniform sampler2D uDepthMap[MAX_CAMS];
  uniform sampler2D uMatteAtlas;
  uniform vec2 uAtlasGrid;   // (列数, 行数)
  uniform vec2 uAtlasPad;    // 瓦片内半像素留边
  uniform mat4 uProjViewMatrix[MAX_CAMS];
  uniform mat4 uProjProjectionMatrix[MAX_CAMS];
  uniform vec3 uProjCamPos[MAX_CAMS];
  uniform vec4 uIntrinsics[MAX_CAMS];
  uniform vec4 uDistortion[MAX_CAMS];
  uniform vec4 uColorAdjust[MAX_CAMS];
  uniform vec3 uColorGain[MAX_CAMS];
  uniform float uCamWeight[MAX_CAMS];  // 每路相机在本面片上的着色权重（由 CPU 按目标选定主相机并随时间淡入淡出）
  uniform float uSupportRadius;  // 一致性表决的空间容差（米）
  uniform vec4 uTargets[MAX_TARGETS];        // 全部目标 (x, 脚底 y, z, 躯干半径)
  uniform float uTargetHeights[MAX_TARGETS]; // 目标高度；<= 0 表示该目标定位不够可靠，不参与像素归属判定
  uniform int uTargetCount;
  uniform int uSelf;             // 本面片对应的目标下标
  uniform float uCardWidth;
  uniform float uCardHeight;
  uniform vec2 uCardNormal;      // 竖直面片的水平法向 (x, z)
  uniform float uFlat;           // 1 = 低矮目标的水平面片（盖在目标顶面高度）
  varying vec3 vWorldPos;
  varying vec2 vUv;

  ${GLSL_CAMERA_MODEL}
${glslSlotSampler('sampleLiveSlot', 'uLiveTex', 'vec3', 'rgb')}
${glslSlotSampler('sampleDepthSlot', 'uDepthMap', 'float', 'r')}

${GLSL_SAMPLE_MATTE}
${GLSL_STATIC_OCCLUDED}

  // 该相机在此处拍到的是否其实是别的目标（视线穿过了另一目标的躯干柱体）：
  // 返回 2 = 该目标的躯干挡在前面，或此点已超出本目标躯干范围，像素归对方；
  // 返回 1 = 此点位于本目标躯干范围内但视线擦过对方，归属存疑（躯干柱体内的空隙会透出后面的人）；
  // 返回 0 = 视线上没有其他目标
  int claimedByOther(vec3 camPos, vec3 p, bool insideSelfBody) {
    vec2 origin = camPos.xz;
    vec2 dir = p.xz - origin;
    float len2 = max(dot(dir, dir), 1e-6);
    float len = sqrt(len2);
    int result = 0;
    for (int j = 0; j < MAX_TARGETS; j++) {
      if (j >= uTargetCount) break;
      if (j == uSelf || uTargetHeights[j] <= 0.0) continue;
      vec4 tg = uTargets[j];
      float s = dot(tg.xz - origin, dir) / len2;   // 视线参数：0 在相机处，1 在本点处
      if (s <= 0.02) continue;
      // 四肢会伸出躯干柱体：柱体外扩一圈内都视为可能拍到对方
      float gap = distance(origin + dir * s, tg.xz);
      if (gap > tg.w + 0.2) continue;
      float yAt = camPos.y + (p.y - camPos.y) * s;
      if (yAt < tg.y || yAt > tg.y + uTargetHeights[j]) continue;
      bool inFront = (1.0 - s) * len > tg.w;
      if (!insideSelfBody || (inFront && gap <= tg.w)) return 2;
      result = 1;
    }
    return result;
  }

  // 该相机在容差范围内是否也看到了前景：容忍多路视频不同步、标定误差与落脚点误差
  float nearbySupport(int i, vec2 uv, float dist, float low) {
    vec4 mc = sampleMatte(i, uv);
    float best = mc.g * (1.0 - mc.b * low);
    if (best >= 0.5) return best;
    vec2 r = uSupportRadius * uIntrinsics[i].xy / dist;
    for (int k = 1; k <= 2; k++) {
      float f = float(k) * 0.5;
      vec4 m0 = sampleMatte(i, uv + vec2(r.x * f, 0.0));
      vec4 m1 = sampleMatte(i, uv - vec2(r.x * f, 0.0));
      vec4 m2 = sampleMatte(i, uv + vec2(0.0, r.y * f));
      vec4 m3 = sampleMatte(i, uv - vec2(0.0, r.y * f));
      best = max(best, max(max(m0.g * (1.0 - m0.b * low), m1.g * (1.0 - m1.b * low)),
                           max(m2.g * (1.0 - m2.b * low), m3.g * (1.0 - m3.b * low))));
    }
    return best;
  }

  void main() {
    vec3 p = vWorldPos;
    float heightAboveFoot = vUv.y * uCardHeight;

    // 面片大部分区域是空的：先用每路一次抠像采样快速判断，任何相机都没看到前景就直接丢弃
    bool anyForeground = false;
    for (int i = 0; i < MAX_CAMS; i++) {
      if (i >= uActiveCams) break;
      if (uLiveOn[i] < 0.5) continue;
      vec4 viewPos = uProjViewMatrix[i] * vec4(p, 1.0);
      if (viewPos.z >= -0.05) continue;
      vec2 uv = computeDistortedUV(viewPos.xyz, uIntrinsics[i], uDistortion[i]);
      if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) continue;
      if (sampleMatte(i, uv).r > 0.02) {
        anyForeground = true;
        break;
      }
    }
    if (!anyForeground) discard;
    // 是否位于本目标躯干范围内：范围外的近地面深色像素按接触阴影/地面倒影处理
    float bodyRadius = uTargets[uSelf].w;
    float lateral = uFlat > 0.5 ? length(vUv - 0.5) * uCardWidth : abs(vUv.x - 0.5) * uCardWidth;
    float inBody = 1.0 - smoothstep(bodyRadius, bodyRadius + 0.08, lateral);

    vec3 accColor = vec3(0.0);
    float accAlpha = 0.0;
    float accWeight = 0.0;
    int approvals = 0;   // 归属明确且看到前景的相机数
    int vetoes = 0;      // 看得见这里却没看到前景的相机数
    float approveWeight = 0.0;
    float vetoWeight = 0.0;

    for (int i = 0; i < MAX_CAMS; i++) {
      if (i >= uActiveCams) break;
      if (uLiveOn[i] < 0.5) continue;

      vec4 viewPos = uProjViewMatrix[i] * vec4(p, 1.0);
      if (viewPos.z >= -0.05) continue;
      vec2 uv = computeDistortedUV(viewPos.xyz, uIntrinsics[i], uDistortion[i]);
      if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) continue;
      if (staticOccluded(i, viewPos)) continue;
      int claim = claimedByOther(uProjCamPos[i], p, inBody > 0.5);
      if (claim == 2) continue;

      vec3 toCam = normalize(uProjCamPos[i] - p);
      float camW = uCamWeight[i];

      // 地面上的影子被该相机"画"到竖直面片上的高度随俯角增大而升高，据此确定暗影剔除区
      // （水平面片本身就贴近地面，不做此处理）
      float shadowTop = clamp(0.7 * toCam.y / max(length(toCam.xz), 1e-3), 0.3, 1.2);
      float low = (1.0 - uFlat) * (1.0 - smoothstep(shadowTop * 0.6, shadowTop, heightAboveFoot));

      if (nearbySupport(i, uv, -viewPos.z, low) < 0.5) {
        vetoes++;
        vetoWeight += camW;
        continue;
      }
      // 归属存疑的相机仍参与着色，但不计入赞成票：只要有别的相机看见这里是空的就会被否决
      if (claim == 0) {
        approvals++;
        approveWeight += camW;
      }

      // 视线几乎贴着面片掠过（含近乎垂直俯拍竖直面片）的相机会把画面拉成长条，不参与着色
      float facing = uFlat > 0.5 ? abs(toCam.y) : abs(dot(toCam.xz, uCardNormal));
      float frontal = smoothstep(0.3, 0.55, facing);
      if (frontal <= 0.0) continue;
      float w = frontal * camW;

      vec4 matte = sampleMatte(i, uv);
      float alpha = matte.r * (1.0 - matte.b * low) * (1.0 - matte.a * low * (1.0 - inBody));

      vec3 color = pow(max(sampleLiveSlot(i, uv), vec3(0.0)), vec3(2.2));
      color = applyColorAdjust(color, uColorAdjust[i]) * uColorGain[i];

      accColor += color * (w * alpha);
      accAlpha += w * alpha;
      accWeight += w;
    }

    // 一致性表决：反对票多于赞成票则否决；平票时躯干范围内听从更贴近观察方向的一方，
    // 躯干范围外（四肢伸展区）更易混入重影与噪点，平票即否决
    bool tie = vetoes > 0 && vetoes == approvals;
    bool rejected = vetoes > approvals || (tie && (inBody < 0.5 || vetoWeight > approveWeight));
    if (rejected || accWeight <= 0.0) discard;
    // 次要相机在主相机判为背景处的零星贡献会形成半透明残影，收紧 Alpha 曲线将其滤除
    float alpha = smoothstep(0.08, 0.6, accAlpha / accWeight);
    if (alpha < 0.02) discard;

    gl_FragColor = vec4(accColor / max(accAlpha, 1e-30), alpha);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

/**
 * 接触阴影顶点着色器：在顶点阶段查询各路抠像，确认目标躯干位置确实有前景才显示阴影，
 * 避免误检轨迹在空地上留下孤零零的影子
 */
export const CONTACT_SHADOW_VERT = `
  #define MAX_CAMS ${MAX_CAMS}

  uniform int uActiveCams;
  uniform float uLiveOn[MAX_CAMS];
  uniform sampler2D uMatteAtlas;
  uniform vec2 uAtlasGrid;
  uniform vec2 uAtlasPad;
  uniform mat4 uProjViewMatrix[MAX_CAMS];
  uniform vec4 uIntrinsics[MAX_CAMS];
  uniform vec4 uDistortion[MAX_CAMS];
  uniform vec3 uFoot;
  uniform float uBodyHeight;
  varying vec2 vUv;
  varying float vPresence;

  ${GLSL_CAMERA_MODEL}

  float supportAt(int idx, vec2 uv) {
    int cols = int(uAtlasGrid.x);
    int row = idx / cols;
    vec2 cell = vec2(float(idx - row * cols), float(row));
    return textureLod(uMatteAtlas, (cell + clamp(uv, uAtlasPad, 1.0 - uAtlasPad)) / uAtlasGrid, 0.0).g;
  }

  void main() {
    int seenFg = 0;
    int seenEmpty = 0;
    for (int i = 0; i < MAX_CAMS; i++) {
      if (i >= uActiveCams) break;
      if (uLiveOn[i] < 0.5) continue;
      bool inView = false;
      float best = 0.0;
      for (int k = 0; k < 3; k++) {
        vec3 q = uFoot + vec3(0.0, uBodyHeight * (0.3 + 0.25 * float(k)), 0.0);
        vec4 viewPos = uProjViewMatrix[i] * vec4(q, 1.0);
        if (viewPos.z >= -0.05) continue;
        vec2 uv = computeDistortedUV(viewPos.xyz, uIntrinsics[i], uDistortion[i]);
        if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) continue;
        inView = true;
        best = max(best, supportAt(i, uv));
      }
      if (!inView) continue;
      if (best >= 0.5) seenFg++; else seenEmpty++;
    }
    vPresence = (seenFg > 0 && seenFg >= seenEmpty) ? 1.0 : 0.0;

    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export const CONTACT_SHADOW_FRAG = `
  uniform float uStrength;
  varying vec2 vUv;
  varying float vPresence;
  void main() {
    if (vPresence < 0.5) discard;
    float r = length(vUv - 0.5) * 2.0;
    float falloff = 1.0 - smoothstep(0.0, 1.0, r);
    gl_FragColor = vec4(vec3(0.0), falloff * falloff * uStrength);
  }
`;
