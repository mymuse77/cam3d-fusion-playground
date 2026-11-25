import { MAX_CAMS } from '../utils/multiProjector.js';

/**
 * v2 PBR 材质投影注入器：
 * 1. 多路视频投影与重叠区加权融合
 * 2. OpenCV Brown-Conrady 正向广角畸变矫正
 * 3. PCF 3x3 软边缘深度遮挡剔除（消除锯齿与穿模）
 * 4. 掠射角（Grazing Angle）法线衰减（消除侧墙拉丝变形）
 * 5. 每路独立亮度 / 对比度 / 饱和度色彩匹配
 */
export function injectMultiProjection(material, sharedUniforms) {
  if (material.userData.isProjInjected) return;
  material.userData.isProjInjected = true;

  material.customProgramCacheKey = () => `multi-video-proj-v3-${MAX_CAMS}`;

  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, sharedUniforms);

    // 顶点着色器：向片元传递世界空间位置与世界空间法线
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
         vProjWorldPos = modelMatrix * vec4(transformed, 1.0);
         vProjWorldNormal = normalize(mat3(modelMatrix) * objectNormal);`
      );

    // 片元着色器：注入多路 Uniform、畸变函数与色彩调节函数
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
         #define MAX_CAMS ${MAX_CAMS}
         uniform int uActiveCams;
         uniform sampler2D uVideoTex[MAX_CAMS];
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

         // 亮度 / 对比度 / 饱和度微调
         vec3 applyColorAdjust(vec3 col, vec4 adj) {
           col += adj.x;
           col = (col - 0.5) * adj.y + 0.5;
           float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
           col = mix(vec3(luma), col, adj.z);
           return clamp(col, 0.0, 1.0);
         }`
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

          // 5. 深度遮挡剔除（支持 PCF 3x3 软边缘滤波）
          float visibility = 1.0;
          if (uEnableOcclusion) {
            if (uEnablePCF) {
              float visSum = 0.0;
              for (int ox = -1; ox <= 1; ox++) {
                for (int oy = -1; oy <= 1; oy++) {
                  vec2 sampleUV = clamp(depthUV + vec2(float(ox), float(oy)) * uDepthTexelSize, 0.0, 1.0);
                  float closestDepth = texture2D(uDepthMap[i], sampleUV).r;
                  visSum += (currentDepth - uDepthBias <= closestDepth) ? 1.0 : 0.0;
                }
              }
              visibility = visSum / 9.0;
            } else {
              float closestDepth = texture2D(uDepthMap[i], depthUV).r;
              visibility = (currentDepth - uDepthBias <= closestDepth) ? 1.0 : 0.0;
            }
            if (visibility <= 0.001) continue;
          }

          // 6. 边缘羽化 + 掠射角 + 软遮挡综合权重
          vec2 edgeDist = min(videoUV, 1.0 - videoUV);
          float edgeWeight = smoothstep(0.0, max(uFeather, 0.001), min(edgeDist.x, edgeDist.y));
          float weight = edgeWeight * grazingWeight * visibility;

          vec3 rawVideo = texture2D(uVideoTex[i], videoUV).rgb;
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
