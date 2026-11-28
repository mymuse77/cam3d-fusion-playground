import { MAX_CAMS } from '../utils/multiProjector.js';
import { GLSL_CAMERA_MODEL } from './injectMultiProjection.js';
import { glslSlotSampler, GLSL_SAMPLE_MATTE, GLSL_STATIC_OCCLUDED } from './freeViewpointShaders.js';

export const OCCUPANCY_SAMPLE_COUNT = 8;
export const OCCUPANCY_NO_GROUND = -1e4;
// 落地接触检验：落脚点下方的屏幕空间偏移（占画面高度的比例）
export const OCCUPANCY_FOOT_BELOW_UV = 0.03;
// 逐相机前景查询的空间容差（米）
export const OCCUPANCY_SUPPORT_TOLERANCE = 0.06;

/**
 * 地面高度图渲染：把可行走网格的向上朝向三角形按世界 XZ 直接铺到网格纹理上，世界 Y 写入 R，并以 Y 作深度。
 * 两遍：第一遍保留每个格子里最低的向上表面（地板）；第二遍在地板以上 GROUND_MAX_STEP 米内保留最高的向上表面，
 * 这样平台 / 坡道 / 托盘顶面算作行走面，而货架顶、屋顶不算
 */
export const GROUND_MAX_STEP = 1.0;

export const GROUND_HEIGHT_VERT = `
  uniform vec2 uGridOrigin;
  uniform vec2 uGridSize;
  uniform vec2 uYRange;   // (minY, maxY)
  uniform float uDepthSign;   // 1 = 保留最低，-1 = 保留最高
  varying float vWorldY;
  varying vec2 vGridUv;
  void main() {
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vWorldY = wp.y;
    vec2 ndc = (wp.xz - uGridOrigin) / uGridSize * 2.0 - 1.0;
    vGridUv = ndc * 0.5 + 0.5;
    float depth = (wp.y - uYRange.x) / max(uYRange.y - uYRange.x, 1e-3) * 2.0 - 1.0;
    gl_Position = vec4(ndc, depth * uDepthSign, 1.0);
  }
`;

export const GROUND_HEIGHT_FRAG = `
  uniform sampler2D uLowTex;
  uniform float uMaxStep;   // <= 0 表示第一遍，不做地板过滤
  varying float vWorldY;
  varying vec2 vGridUv;
  void main() {
    if (uMaxStep > 0.0 && vWorldY > texture2D(uLowTex, vGridUv).r + uMaxStep) discard;
    gl_FragColor = vec4(vWorldY, 0.0, 0.0, 1.0);
  }
`;

/**
 * 多视角占据度：对地面网格的每个格子，沿身体高度取 8 个采样点（0.15 + 0.25k 米），
 * 逐路相机查询前景支持度，取"所有看见该点的相机里最低者"（一票否决）作为该高度的列占据。
 * R = 占据度，G = 看见躯干下部的相机数 / MAX_CAMS，B = 连续被占据的采样层数 / 8
 */
export const OCCUPANCY_FRAG = `
  #define MAX_CAMS ${MAX_CAMS}
  #define SAMPLES ${OCCUPANCY_SAMPLE_COUNT}
  #define SUPPORT_TOLERANCE ${OCCUPANCY_SUPPORT_TOLERANCE.toFixed(3)}
  #define OCCLUDED_FEET_PENALTY 0.9
  #define FOOT_BELOW_UV ${OCCUPANCY_FOOT_BELOW_UV.toFixed(3)}

  uniform int uActiveCams;
  uniform float uLiveOn[MAX_CAMS];
  uniform sampler2D uDepthMap[MAX_CAMS];
  uniform sampler2D uMatteAtlas;
  uniform vec2 uAtlasGrid;
  uniform vec2 uAtlasPad;
  uniform mat4 uProjViewMatrix[MAX_CAMS];
  uniform mat4 uProjProjectionMatrix[MAX_CAMS];
  uniform vec4 uIntrinsics[MAX_CAMS];
  uniform vec4 uDistortion[MAX_CAMS];
  uniform sampler2D uHeightTex;
  uniform vec2 uGridOrigin;
  uniform vec2 uGridSize;
  varying vec2 vUv;

  ${GLSL_CAMERA_MODEL}
${glslSlotSampler('sampleDepthSlot', 'uDepthMap', 'float', 'r')}
${GLSL_SAMPLE_MATTE}
${GLSL_STATIC_OCCLUDED}

  float matteSupport(int i, vec2 uv) {
    vec4 m = sampleMatte(i, uv);
    return m.g * (1.0 - m.b);
  }

  // 前景支持度（支持抠像去除阴影标记）；tol > 0 时取该半径（米）内的邻域最大值，
  // 容忍标定误差与多路视频不同步；该相机看不见此点时返回 -1
  float supportAt(int i, vec3 p, bool checkStatic, float tol) {
    vec4 viewPos = uProjViewMatrix[i] * vec4(p, 1.0);
    if (viewPos.z >= -0.05) return -1.0;
    vec2 uv = computeDistortedUV(viewPos.xyz, uIntrinsics[i], uDistortion[i]);
    if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) return -1.0;
    if (checkStatic && staticOccluded(i, viewPos)) return -1.0;
    float best = matteSupport(i, uv);
    if (tol <= 0.0 || best >= 0.5) return best;
    vec2 r = tol * uIntrinsics[i].xy / -viewPos.z;
    vec2 d = r * 0.7071;
    best = max(best, max(max(matteSupport(i, uv + vec2(r.x, 0.0)), matteSupport(i, uv - vec2(r.x, 0.0))),
                         max(matteSupport(i, uv + vec2(0.0, r.y)), matteSupport(i, uv - vec2(0.0, r.y)))));
    best = max(best, max(max(matteSupport(i, uv + d), matteSupport(i, uv - d)),
                         max(matteSupport(i, uv + vec2(d.x, -d.y)), matteSupport(i, uv + vec2(-d.x, d.y)))));
    return best;
  }

  // 该相机看来落脚点是否正落在前景下沿：下方一小段（屏幕空间，横向放宽到容差）必须是空地。
  // 两路相机交汇出的"幻影"点落在某个人身体中段或身后，在多数相机里下方仍是身体
  bool footContact(int i, vec2 xz, float ground) {
    vec4 footView = uProjViewMatrix[i] * vec4(xz.x, ground, xz.y, 1.0);
    if (footView.z >= -0.05) return false;
    vec2 footUv = computeDistortedUV(footView.xyz, uIntrinsics[i], uDistortion[i]);
    if (footUv.y - FOOT_BELOW_UV < 0.0 || footUv.x < 0.0 || footUv.x > 1.0) return false;
    vec2 belowUv = footUv - vec2(0.0, FOOT_BELOW_UV);
    float rx = SUPPORT_TOLERANCE * uIntrinsics[i].x / -footView.z;
    float below = max(matteSupport(i, belowUv), max(matteSupport(i, belowUv + vec2(rx, 0.0)), matteSupport(i, belowUv - vec2(rx, 0.0))));
    return below < 0.5;
  }

  void main() {
    float ground = texture2D(uHeightTex, vUv).r;
    if (ground < ${OCCUPANCY_NO_GROUND.toFixed(1)} * 0.5) {
      gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
      return;
    }
    vec2 xz = uGridOrigin + vUv * uGridSize;

    float col[SAMPLES];
    int nCams = 0;
    int contacts = 0;
    for (int k = 0; k < SAMPLES; k++) {
      vec3 p = vec3(xz.x, ground + 0.15 + 0.25 * float(k), xz.y);
      float minSupport = 2.0;
      int seen = 0;
      for (int i = 0; i < MAX_CAMS; i++) {
        if (i >= uActiveCams) break;
        if (uLiveOn[i] < 0.5) continue;
        float s = supportAt(i, p, true, SUPPORT_TOLERANCE);
        if (s < 0.0) continue;
        seen++;
        if (k == 1 && footContact(i, xz, ground)) contacts++;
        minSupport = min(minSupport, s);
      }
      col[k] = seen > 0 ? minSupport : 0.0;
      if (k == 1) {
        nCams = seen;
      }
    }

    // 脚部常被货架 / 道具 / 另一目标挡住：允许占据度从更高的相邻两层起算（小幅折减），落脚点由格子本身给定
    float occ = 0.5 * (col[0] + col[1]);
    int startK = 0;
    for (int k = 1; k < 4; k++) {
      float v = 0.5 * (col[k] + col[k + 1]) * OCCLUDED_FEET_PENALTY;
      if (v > occ) {
        occ = v;
        startK = k;
      }
    }
    // 只有两路及以上相机同时看见才可信；单相机区域沿视线存在深度歧义，交给单目落脚点估计
    if (nCams < 2) occ = 0.0;
    // 多相机时至少要有足够多路相机确认落地接触（容忍其中一路被别人挡住脚）
    if (contacts < nCams - 1) occ = 0.0;

    float heightCount = float(startK);
    for (int k = 0; k < SAMPLES; k++) {
      if (k < startK) continue;
      if (col[k] < 0.5) break;
      heightCount += 1.0;
    }
    gl_FragColor = vec4(occ, float(nCams) / float(MAX_CAMS), heightCount / float(SAMPLES), 1.0);
  }
`;
