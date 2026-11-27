import { MAX_CAMS } from '../utils/multiProjector.js';

/**
 * 相机成像模型 GLSL 片段：OpenCV 正向畸变投影 + 感知空间色彩调节（场景投影与自由视点融合共用）
 */
export const GLSL_CAMERA_MODEL = `
         // OpenCV Brown-Conrady 正向畸变映射
         vec2 computeDistortedUV(vec3 viewPos, vec4 intr, vec4 dist) {
           float x = viewPos.x / (-viewPos.z);
           float y = viewPos.y / viewPos.z;
           float r2 = x * x + y * y;
           if (r2 > 2.5) return vec2(-1.0);

           float r4 = r2 * r2;
           float k1 = dist.x, k2 = dist.y, p1 = dist.z, p2 = dist.w;
           float radial = 1.0 + k1 * r2 + k2 * r4;

           float xDist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
           float yDist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

           float u = xDist * intr.x + intr.z;
           float vCV = yDist * intr.y + intr.w;
           return vec2(u, 1.0 - vCV);
         }

         // 在感知均匀的 sRGB 空间执行亮度 / 对比度 / 饱和度调整，再转回 Linear 空间参与混合
         vec3 applyColorAdjust(vec3 linearCol, vec4 adj) {
           vec3 srgbCol = pow(max(linearCol, vec3(0.0)), vec3(1.0 / 2.2));
           srgbCol += adj.x;
           srgbCol = (srgbCol - 0.5) * adj.y + 0.5;
           float luma = dot(srgbCol, vec3(0.2126, 0.7152, 0.0722));
           srgbCol = mix(vec3(luma), srgbCol, adj.z);
           srgbCol = clamp(srgbCol, 0.0, 1.0);
           return pow(srgbCol, vec3(2.2));
         }
`;

/**
 * v3 材质投影注入器：
 * 1. 多路视频投影与重叠区加权融合（静态槽位采样分发，兼容 ANGLE/移动端，支持最高 6 路并发槽位）
 * 2. OpenCV Brown-Conrady 正向广角畸变矫正
 * 3. PCF 3x3 软边缘深度遮挡剔除（消除锯齿与穿模）
 * 4. 掠射角（Grazing Angle）法线衰减（支持非等比缩放、镜像模型、InstancedMesh 与 Unlit 材质）
 * 5. Linear <-> sRGB 感知空间亮度 / 对比度 / 饱和度色彩匹配
 */
export function injectMultiProjection(material, sharedUniforms) {
  if (
    !material ||
    material.userData.isProjInjected ||
    material.userData.skipProjection ||
    material.isLineBasicMaterial ||
    material.isLineDashedMaterial ||
    material.isPointsMaterial ||
    material.isSpriteMaterial ||
    material.wireframe
  ) {
    return;
  }
  material.userData.isProjInjected = true;

  const isBasic = !!material.isMeshBasicMaterial;
  const localNormalExpr = isBasic ? 'vec3(normal)' : 'objectNormal';

  material.customProgramCacheKey = () => `multi-video-proj-v3-${MAX_CAMS}-${isBasic ? 'basic' : 'lit'}`;

  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, sharedUniforms);

    // 顶点着色器：向片元传递世界空间位置与精确逆转置世界空间法线（兼容 InstancedMesh 与非等比缩放）
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
         varying vec4 vProjWorldPos;
         varying vec3 vProjWorldNormal;`
      )
      .replace(
        '#include <project_vertex>',
        `#include <project_vertex>
         #ifdef USE_INSTANCING
           mat4 projWorldMat = modelMatrix * instanceMatrix;
         #else
           mat4 projWorldMat = modelMatrix;
         #endif
         vProjWorldPos = projWorldMat * vec4(transformed, 1.0);
         vec3 localProjN = ${localNormalExpr};
         mat3 m3 = mat3(projWorldMat);
         vec3 c0 = m3[0], c1 = m3[1], c2 = m3[2];
         vec3 adj0 = cross(c1, c2);
         float det = dot(c0, adj0);
         vec3 rawWorldN = sign(det) * (adj0 * localProjN.x + cross(c2, c0) * localProjN.y + cross(c0, c1) * localProjN.z);
         vProjWorldNormal = abs(det) > 1e-8 ? normalize(rawWorldN) : vec3(0.0, 1.0, 0.0);`
      );

    // 片元着色器：注入多路 Uniform、静态下标采样器、畸变函数与 Linear-sRGB 色彩调节函数
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
         #define MAX_CAMS ${MAX_CAMS}
         uniform int uActiveCams;
         uniform sampler2D uVideoTex[MAX_CAMS];
         uniform float uIsVideoTex[MAX_CAMS];
         uniform sampler2D uDepthMap[MAX_CAMS];
         uniform mat4 uProjViewMatrix[MAX_CAMS];
         uniform mat4 uProjProjectionMatrix[MAX_CAMS];
         uniform vec3 uProjCamPos[MAX_CAMS];
         uniform vec4 uIntrinsics[MAX_CAMS];   // (fx/w, fy/h, cx/w, cy/h)
         uniform vec4 uDistortion[MAX_CAMS];   // (k1, k2, p1, p2)
         uniform vec4 uColorAdjust[MAX_CAMS];  // (brightness, contrast, saturation, 0)
         uniform float uOpacity;
         uniform float uFeather;
         uniform float uDepthBias;
         uniform float uGrazingThreshold;
         uniform float uDepthTexelSize;
         uniform bool uEnableOcclusion;
         uniform bool uEnablePCF;
         varying vec4 vProjWorldPos;
         varying vec3 vProjWorldNormal;

         vec4 sampleVideoSlot(int idx, vec2 uv) {
           if (idx == 0) return texture2D(uVideoTex[0], uv);
           else if (idx == 1) return texture2D(uVideoTex[1], uv);
           else if (idx == 2) return texture2D(uVideoTex[2], uv);
           else if (idx == 3) return texture2D(uVideoTex[3], uv);
           else if (idx == 4) return texture2D(uVideoTex[4], uv);
           else return texture2D(uVideoTex[5], uv);
         }

         float sampleDepthSlot(int idx, vec2 uv) {
           if (idx == 0) return texture2D(uDepthMap[0], uv).r;
           else if (idx == 1) return texture2D(uDepthMap[1], uv).r;
           else if (idx == 2) return texture2D(uDepthMap[2], uv).r;
           else if (idx == 3) return texture2D(uDepthMap[3], uv).r;
           else if (idx == 4) return texture2D(uDepthMap[4], uv).r;
           else return texture2D(uDepthMap[5], uv).r;
         }

         ${GLSL_CAMERA_MODEL}`
      )
      .replace(
        '#include <tonemapping_fragment>',
        `
        vec3 accumVideo = vec3(0.0);
        float totalWeight = 0.0;
        vec3 worldN = normalize(vProjWorldNormal);

        for (int i = 0; i < MAX_CAMS; i++) {
          if (i >= uActiveCams) break;

          // 1. 位于投影相机后方直接剔除
          vec4 viewPos = uProjViewMatrix[i] * vProjWorldPos;
          if (viewPos.z >= 0.0) continue;

          // 2. 掠射角（Grazing Angle）淡出：消除侧墙与背面的画面拉伸
          vec3 toCamDir = normalize(uProjCamPos[i] - vProjWorldPos.xyz);
          float ndotv = dot(worldN, toCamDir);
          float grazingWeight = smoothstep(uGrazingThreshold, uGrazingThreshold + 0.25, ndotv);
          if (grazingWeight <= 0.001) continue;

          // 3. 理想针孔投影坐标（专门用于查询深度图）
          vec4 projCoord = uProjProjectionMatrix[i] * viewPos;
          vec3 projNDC = projCoord.xyz / projCoord.w;
          vec2 depthUV = projNDC.xy * 0.5 + 0.5;
          float currentDepth = projNDC.z * 0.5 + 0.5;

          bool inDepthFrustum =
            depthUV.x >= 0.0 && depthUV.x <= 1.0 &&
            depthUV.y >= 0.0 && depthUV.y <= 1.0 &&
            currentDepth >= 0.0 && currentDepth <= 1.0;
          if (!inDepthFrustum) continue;

          // 4. 正向畸变视频采样 UV
          vec2 videoUV = computeDistortedUV(viewPos.xyz, uIntrinsics[i], uDistortion[i]);
          if (videoUV.x < 0.0 || videoUV.x > 1.0 || videoUV.y < 0.0 || videoUV.y > 1.0) {
            continue;
          }

          // 5. 深度遮挡剔除（支持 PCF 3x3 软边缘滤波，使用静态下标采样器）
          float visibility = 1.0;
          if (uEnableOcclusion) {
            if (uEnablePCF) {
              float visSum = 0.0;
              for (int ox = -1; ox <= 1; ox++) {
                for (int oy = -1; oy <= 1; oy++) {
                  vec2 sampleUV = clamp(depthUV + vec2(float(ox), float(oy)) * uDepthTexelSize, 0.0, 1.0);
                  float closestDepth = sampleDepthSlot(i, sampleUV);
                  visSum += (currentDepth - uDepthBias <= closestDepth) ? 1.0 : 0.0;
                }
              }
              visibility = visSum / 9.0;
            } else {
              float closestDepth = sampleDepthSlot(i, depthUV);
              visibility = (currentDepth - uDepthBias <= closestDepth) ? 1.0 : 0.0;
            }
            if (visibility <= 0.001) continue;
          }

          // 6. 边缘羽化 + 掠射角 + 软遮挡综合权重
          vec2 edgeDist = min(videoUV, 1.0 - videoUV);
          float edgeWeight = smoothstep(0.0, max(uFeather, 0.001), min(edgeDist.x, edgeDist.y));
          float weight = edgeWeight * grazingWeight * visibility;

          vec3 rawVideo = sampleVideoSlot(i, videoUV).rgb;
          // WebGL2 下 VideoTexture 未经硬件 sRGB->Linear 解码，需手动解码以与静态背景/PBR 色彩空间一致
          if (uIsVideoTex[i] > 0.5) {
            rawVideo = pow(max(rawVideo, vec3(0.0)), vec3(2.2));
          }
          vec3 videoColor = applyColorAdjust(rawVideo, uColorAdjust[i]);

          accumVideo += videoColor * weight;
          totalWeight += weight;
        }

        if (totalWeight > 0.001) {
          vec3 blendedVideo = accumVideo / totalWeight;
          float finalAlpha = clamp(totalWeight, 0.0, 1.0) * uOpacity;
          gl_FragColor.rgb = mix(gl_FragColor.rgb, blendedVideo, finalAlpha);
        }

        #include <tonemapping_fragment>
        `
      );
  };

  material.needsUpdate = true;
}
