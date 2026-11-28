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
- [x] **多视角自由视点前景融合（视点相关投影 + 像素归属判定）**
  - **已实现**：新增前景融合模式「多视角自由视点融合」（[src/utils/freeViewpointFusion.js](../src/utils/freeViewpointFusion.js)）。每帧在 GPU 上生成各路相机的前景抠像图集（背景差分 + AI 掩膜三分图），在目标落脚点竖起承载面片并按真实标定回投各路实时视频，以视点相关权重选用最贴近当前观察方向的相机（面片法向取相机方向与观察方向的角平分线，宽度不失真）；结合外部 3D 轨迹做像素归属判定与多相机一致性表决以剔除目标间重影，低矮目标（AGV）改用水平面片，并叠加接触阴影。
  - **后续补充**：① 目标标签（[src/utils/targetBadges.js](../src/utils/targetBadges.js)）与立牌模式共用，自由视点下只显示标签（接触阴影已标出落脚点）；② 每个目标由 CPU 选定主相机（分数边际 + 400 ms 保持滞回，切换时 300 ms 交叉淡化，面片法向随权重连续转动），着色器只按每面片的相机权重着色，不再逐像素抢相机，主相机弃权处才由其余相机补全；③ 无外部 3D 轨迹时改用混合定位（[src/utils/groundGrid.js](../src/utils/groundGrid.js) + [src/utils/occupancyLocator.js](../src/utils/occupancyLocator.js)）：被两路及以上相机同时看见的地面格子用多视角占据度（各路抠像沿身高多层采样、一票否决，并要求多路相机确认落地接触）提取峰值，只被一路相机看见的区域沿用单目落脚点估计，二者经卡尔曼 + 匈牙利统一跟踪赋 ID，多相机确认的目标参与像素归属判定；地面高度图取地板以上 1 米内的最高向上表面（平台 / 坡道算行走面，货架顶不算）；④ 跨相机自动色彩匹配（[src/utils/autoColorMatcher.js](../src/utils/autoColorMatcher.js)）：由背景底图在共同地面点上的颜色比解出每路相机线性 RGB 增益，场景投影与自由视点共用。
  - **已知局限**：人群密集时两路 / 多路相机交汇仍会产生幻影峰（EPFL Terrace 约 8 人同框时可见轨迹可达 15~20 条，人少时 5~10 条，约 3 个新 ID/秒）；单相机区域靠单目落脚点，位置误差大、人形遮挡时可能漏检，视觉定位的目标不分类型，统一标为"目标"；切换主相机时面片朝向仍会在 0.3 秒内转过较大角度（最大约 18°/30 ms，已无翻面跳变）；占据度定位依赖抠像质量，抠像漏检的身体部位会让目标整体丢失。
- [ ] **延迟贴花投影（Deferred Projective Decals）+ WebGL2 `sampler2DArray`**
  - **方案**：基于场景主深度缓冲与视锥包围盒执行屏幕空间延迟投影贴花，配合 `sampler2DArray` 批量采样纹理，零侵入兼容倾斜摄影 `3D Tiles` 并突破前向渲染同屏 4 路采样器限制（扩展至 16～32 路）。
