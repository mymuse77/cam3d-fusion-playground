import { MAX_CAMS } from '../utils/multiProjector.js';
import { GLSL_CAMERA_MODEL } from './injectMultiProjection.js';
import { glslSlotSampler, GLSL_STATIC_OCCLUDED } from './freeViewpointShaders.js';
import { OCCUPANCY_NO_GROUND } from './occupancyShaders.js';

export const COLOR_SAMPLE_BORDER = 0.04;

/**
 * 自动色彩匹配采样：把指定槽位相机的静态背景底图俯视投到地面网格上（采样结果已套用该路的亮度 / 对比度 / 饱和度），
 * 供 CPU 比较多路相机在同一地面点的颜色差异。alpha = 0 表示该格子此相机无效
 */
export const COLOR_SAMPLE_FRAG = `
  #define MAX_CAMS ${MAX_CAMS}

  uniform int uSlot;
  uniform sampler2D uBg;
  uniform sampler2D uDepthMap[MAX_CAMS];
  uniform mat4 uProjViewMatrix[MAX_CAMS];
  uniform mat4 uProjProjectionMatrix[MAX_CAMS];
  uniform vec4 uIntrinsics[MAX_CAMS];
  uniform vec4 uDistortion[MAX_CAMS];
  uniform vec4 uColorAdjust[MAX_CAMS];
  uniform sampler2D uHeightTex;
  uniform vec2 uGridOrigin;
  uniform vec2 uGridSize;
  varying vec2 vUv;

  ${GLSL_CAMERA_MODEL}
${glslSlotSampler('sampleDepthSlot', 'uDepthMap', 'float', 'r')}
${GLSL_STATIC_OCCLUDED}

  void main() {
    float ground = texture2D(uHeightTex, vUv).r;
    if (ground < ${OCCUPANCY_NO_GROUND.toFixed(1)} * 0.5) {
      gl_FragColor = vec4(0.0);
      return;
    }
    vec3 p = vec3(uGridOrigin.x + vUv.x * uGridSize.x, ground, uGridOrigin.y + vUv.y * uGridSize.y);
    vec4 viewPos = uProjViewMatrix[uSlot] * vec4(p, 1.0);
    if (viewPos.z >= -0.05) {
      gl_FragColor = vec4(0.0);
      return;
    }
    vec2 uv = computeDistortedUV(viewPos.xyz, uIntrinsics[uSlot], uDistortion[uSlot]);
    float b = ${COLOR_SAMPLE_BORDER.toFixed(3)};
    if (uv.x < b || uv.x > 1.0 - b || uv.y < b || uv.y > 1.0 - b || staticOccluded(uSlot, viewPos)) {
      gl_FragColor = vec4(0.0);
      return;
    }
    // 背景底图为 sRGB 色彩空间纹理，采样结果已是线性
    vec3 c = applyColorAdjust(texture2D(uBg, uv).rgb, uColorAdjust[uSlot]);
    gl_FragColor = vec4(pow(c, vec3(1.0 / 2.2)), 1.0);
  }
`;
