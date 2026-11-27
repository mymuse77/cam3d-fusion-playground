# 后续优化与高阶演进待办清单 (`TODO.md`)

本文档记录 `cam3d-fusion-playground` v3.0 阶段之后的潜在改进方向，按**实施优先级与投入产出比**分为三个层次，可根据后续接入真实监控或业务演示需求按需推进。

---

## 一、近期高性价比打磨项（接入真实视频 / 现场验收刚需）

- [ ] **球机 PTZ 巡航同步刷新调度器空间包围球**
  - **现状**：[src/main.js](../src/main.js) 主循环中球机巡航更新了 `helper` 和 `BBoxProjector`，但未触发 `scheduler.refreshCameraBounds(ptzItem)`。
  - **方案**：在球机姿态/变焦变化超过角度阈值（如 `> 2°`）或按低频节流时调用 `scheduler.refreshCameraBounds(ptzItem)`，保证 30 路多相机调度下视锥剔除与评分始终与球机实时朝向一致。
- [ ] **深度图升级 24/32-bit 精度 + 斜率自适应偏移（Slope-Scaled Depth Bias）**
  - **现状**：[src/utils/multiProjector.js](../src/utils/multiProjector.js) 使用 16 位 `UnsignedShortType` 深度纹理，[src/shaders/injectMultiProjection.js](../src/shaders/injectMultiProjection.js) 使用固定常量 `uDepthBias`，远视距（`far > 150m`）易出现精度抖动或侧面斑马纹。
  - **方案**：升级为 WebGL2 原生支持的 `UnsignedIntType` / `FloatType` 深度纹理，并在片元着色器中按表面法线夹角动态计算偏移 `bias = max(uDepthBias * (1.0 - ndotv), uDepthBias * 0.25)`。
- [ ] **「一键切入监控第一视角（Fly-to）+ 2D 原画画中画（PIP）」同屏验收模式**
  - **现状**：外参标定后仅靠鼠标自由视角难以直观核对建筑边缘贴合精度。
  - **方案**：支持点击任意监控相机将主观察相机平滑移动至该路监控安装位姿与 `FOV`，同时在屏幕角落悬浮显示原始 2D 视频画中画（PIP）与叠加透明度快速对比开关。

---

## 二、交互完善与大型模型性能优化项

- [ ] **交互式多边形电子围栏绘制与配置持久化**
  - **方案**：复用射线采点机制，支持在地面点击绘制/编辑任意多边形电子围栏及光墙高度，并随 `park_cameras.json` 一并导入导出。
- [ ] **大型精细模型射线求交加速（BVH）**
  - **方案**：针对 30 万～50 万面以上的精细园区 `.glb` / 倾斜摄影模型，引入 `three-mesh-bvh`（或空间包围盒预筛）加速 [src/utils/pointPicker.js](../src/utils/pointPicker.js) 与 [src/utils/bboxProjector.js](../src/utils/bboxProjector.js) 的逐帧求交性能。
- [ ] **Vite 生产构建分包优化**
  - **方案**：在 [vite.config.js](../vite.config.js) 中配置 `build.rollupOptions.output.manualChunks`，将 `three` 拆分为独立 vendor chunk，消除大体积 bundle 警告并提升浏览器静态缓存复用率。

---

## 三、生产环境对接与图形学高阶演进项

- [ ] **真实 AI 推理流接入与媒体帧级同步（PTS 对齐）**
  - **方案**：接入 WebSocket / MQTT 实时推送的边缘 AI 检测框与局部跟踪 ID，结合 `video.requestVideoFrameCallback` 获取解码帧 `mediaTime`，与 AI 队列插值对齐以消除 WebRTC 传输抖动导致的框画错位。
- [ ] **精确偏心内参非对称投影矩阵（Off-Center Frustum）**
  - **方案**：直接依据标定内参 `(fx, fy, cx, cy, width, height)` 构造非对称投影矩阵替代对称视场角近似放大，最大化 `1024×1024` 深度图的有效像素利用率。
- [x] **消除动态目标“墙面拉长鬼影”（3D Video Billboard 立牌抠像）**
  - **已实现**：新增「前景融合模式」（`动静分离 + 3D 视频立牌` / `原始视频直投` / `纯静态背景`），背景层采用时域中值静态底图消除躺地 V 字双影，前景通过 `VideoBillboardManager` 实时运动检测、跨镜去重并在唯一三维落脚点竖起带背景差分抠像的 3D 视频立牌。
- [ ] **延迟贴花投影（Deferred Projective Decals）+ WebGL2 `sampler2DArray`**
  - **方案**：基于场景主深度缓冲与视锥包围盒执行屏幕空间延迟投影贴花，配合 `sampler2DArray` 批量采样纹理，零侵入兼容倾斜摄影 `3D Tiles` 并突破前向渲染同屏 4 路采样器限制（扩展至 16～32 路）。
