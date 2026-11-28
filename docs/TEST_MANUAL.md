# 数字孪生视频融合：逐步骤测试实操手册 (`TEST_MANUAL.md`)

本手册配合 `cam3d-fusion-playground` v3.0 工程使用，涵盖从环境启动、6 大核心功能练习、Web 联合标定与 Python OpenCV `solvePnPRansac` 联调到常见报错排查的完整操作步骤与推荐参数。

---

## 一、环境准备与启动自检

### 1. 前端启动与算法自检
确保本地已安装 **Node.js v18+**，在项目根目录执行：
```bash
npm install
npm test
npm run dev
```
启动后用 Chrome / Edge 访问终端输出的本地地址（默认 `http://localhost:5173`）。

### 2. 启动自检标准
页面打开后应立刻看到：
- 3D 工业园区场景（深灰地面、后方主厂房、左侧配电室、前排棕色立柱、绿色设备箱，以及东侧带斜坡的 `0.8m` 巡检高台）。
- 3 个带彩色边框和扫描线的动态测试画布投影在场景中（`CAM_FIXED_01` 青色、`CAM_FIXED_02` 橙色、`CAM_PTZ_03` 紫色）。
- 左上角状态栏显示：`激活槽位: 3 / 4 (池总数: 3)`。

### 3. 一键运行 Blender 5.3 成套真值资产联调（可选）
若本地已安装 Blender 5.3 与 FFmpeg，可执行以下命令一键生成真实同源园区模型 `public/demo-blender/blender_park.glb`、双路 720P 监控视频 `cam_fixed_01.mp4` / `cam_fixed_02.mp4` 及参考帧与控制点真值，再用 Python OpenCV 求解外参：
```bash
"D:\Env\Blender 5.3\blender.exe" --background --factory-startup --python blender-tools/generate_blender_suite.py
python python-tools/calib_toolkit.py pnp --input public/demo-blender/calib_points_cam_fixed_01.json --config public/demo-blender/park_cameras_blender.json
python python-tools/calib_toolkit.py pnp --input public/demo-blender/calib_points_cam_fixed_02.json --config public/demo-blender/park_cameras_blender.json
```
随后在页面左上角点击 **「加载 Blender 5.3 实景套组」**，即可直接查看两路 Blender 监控视频像素级投射到同源 3D 建筑上的融合效果（也可用 `public/demo-blender/cam_fixed_01_ref.png` 在 Web 标定台亲手练习 2D/3D 采点求解）。
- **体验「前景融合模式」切换**：在左上角或右上角 GUI 切换 **「动静分离 + 3D 视频立牌（推荐）」**、**「动静分离 + 多视角自由视点融合」**（绕着目标转动视角，应看到目标随之换成对应侧相机拍到的真实正面 / 侧面 / 背面（切换时约 0.3 秒交叉淡化），脚下带接触阴影，头顶带类型 / ID 标签；无外部 3D 轨迹的实景（EPFL / Blender）由多视角占据度自动定位目标，GUI 中「自动色彩匹配（跨相机）」可对比开关前后的相机色差接缝；NVIDIA 仓库实景下效果最佳）、**「原始视频直投（对比躺地 V 字双影）」** 与 **「纯静态背景（隐藏动态前景 / 专心标定）」**，直观对比消除地面重影前后将动态巡检员与 AGV 真正立在三维地面/高台上的效果。

### 4. 实景“脏数据 / 容错”压测套组（广角畸变、棋盘格标定、色差与模实偏差）
如需压测真实工地常见痛点（桶形畸变+偏心光心、双机位冷暖曝光反差、施工移位+树木遮挡、支架滚转歪斜及采点手抖+误点 Outlier），可执行：
```bash
# 1. 生成 chess.mp4 棋盘格标定视频与两路脏数据压测视频
python blender-tools/generate_dirty_suite.py

# 2. 用 chess.mp4 标定相机真实内参与畸变系数
python python-tools/calib_toolkit.py intrinsics --video public/demo-blender-dirty/chess.mp4 --export-intrinsics public/demo-blender-dirty/intrinsics_calibrated.json

# 3. 运行带手抖噪声与误点剔除的 RANSAC+LM PnP 标定
python python-tools/calib_toolkit.py pnp --input public/demo-blender-dirty/calib_points_cam_dirty_01.json --config public/demo-blender-dirty/park_cameras_dirty.json
python python-tools/calib_toolkit.py pnp --input public/demo-blender-dirty/calib_points_cam_dirty_02.json --config public/demo-blender-dirty/park_cameras_dirty.json
```
在页面左上角点击 **「加载实景脏数据压测套组」** 即可体验对比：
- 在 GUI 中将 `cam_dirty_01` 的 **「径向畸变 k1」** 临时归零 `0.0`，立即可观察未矫正时厂房边缘的桶形弯曲；调回 `-0.26` 后自动拉直对齐。
- 将 `cam_dirty_01` / `cam_dirty_02` 的 **画面亮度 / 对比度 / 饱和度** 恢复默认值 `(0, 1, 1)`，可对比冷暖曝光色差与开启色彩匹配后的接缝融合效果；同时观察现场违建雨棚、树木和移位设备箱在设计白模上的投影表现。

---

## 二、6 大实操练习逐步骤指南

### 练习 1：多路视频流接入、PCF 软遮挡与去掠射角拉伸
**目标**：掌握多路本地/远程视频接入（含槽位休眠）、双通道深度遮挡防穿模、PCF 3×3 边缘柔化，以及侧墙法线衰减消除拉丝感。

1. **多路视频 / 流媒体接入**：
   - 在左上角「练习 1：选择目标监控并接入本地视频 / 远程流」下拉框中选中任意监控（如 `cam_fixed_01` 或 `cam_fixed_02`），上传本地 `.mp4`（H.264）文件，或输入远程流地址（支持 `.mp4`、`.m3u8`、WebRTC `/whep`）点击 **「接入流地址」**。
   - 若需还原，点击 **「恢复测试画布」** 即可。
2. **验证防穿模（深度遮挡）**：
   - 拖动鼠标左键，将视角旋转到**前排棕色立柱的后方**（看向主厂房墙面）。
   - 在右上角 GUI 展开「练习 1：投影画质、PCF 与去拉伸」，**取消勾选「开启深度遮挡」**，可见视频穿透立柱印在后墙；重新勾选后立柱背后恢复正常阴影。
3. **验证 PCF 3×3 软边缘与 Bias**：
   - 拉近观察立柱在主厂房上的遮挡边界，切换 **「PCF 3×3 软边缘」** 对比锯齿与柔化差异；将 **「深度偏移 Bias」** 调至 `0.0020 ~ 0.0030` 消除斑马纹。
4. **验证消除侧墙拉丝（掠射角淡出）**：
   - 旋转到主厂房或立柱侧面，对比「掠射角淡出阈值」在 `0.0` 与 `0.12 ~ 0.25` 时的侧墙贴图拉伸抑制效果。

---

### 练习 2：广角镜头畸变矫正与画质调色
**目标**：理解 OpenCV 正向畸变映射原理，并通过 Linear↔sRGB 感知空间色彩调节消除视频的“贴纸感”。

1. 展开右上角 GUI「练习 2 & 4：相机微调与 JSON 导出」，确认「选中相机」为 `cam_fixed_01`。
2. **测试广角畸变**：将 **「径向畸变 k1」** 调至 `-0.35`（桶形畸变）或 `+0.30`（枕形畸变），观察墙面投影弯曲而立柱深度遮挡边界依然严丝合缝。
3. **色彩匹配调优**：依次微调 **画面亮度**（如 `-0.08`）、**画面对比度**（如 `1.15`）、**画面饱和度**（如 `0.85`）。调节过程在 sRGB 感知空间执行，不会出现线性空间对比度偏移导致的暗部死黑。

---

### 练习 3：球机（PTZ）平滑巡航与光学变焦联动
**目标**：体验云台延迟对齐缓冲队列、光学变焦与局部单槽位深度图更新。

1. 展开右上角 GUI「练习 3：球机 PTZ 巡航与光学变焦」，勾选 **「开启球机自动巡航(含变焦)」**。
2. *预期现象*：
   - 3 号球机视锥体开始平滑扫动并周期性变焦，自动联动更新 `FOV`、`fx/fy`、`BBoxProjector` 标定参数与畸变衰减比例。
   - 即使在巡航中途点击 **「一键导出 park_cameras.json」**，导出的外参矩阵仍保持纯净的安装基准姿态，不会被巡航瞬时姿态污染。

---

### 练习 4：Web 端 2D/3D 联合标定与 Python `solvePnPRansac` 联调
**目标**：跑通从控制点采集到 RANSAC+LM 求解与 JSON 热重载的完整空间配准闭环。

#### 方式 A：纯前端 Web 2D/3D 联合标定台（推荐）
1. 在 3D 场景中按住 **`Shift` + 鼠标左键**，依次采集 6~8 个**不共面**的清晰特征点（如主厂房角点 + 立柱顶点 + 东侧平台角点）。
2. 点击左侧 **「Web 2D/3D 标定台」** 打开弹窗，点击「抓取当前视频帧」或上传监控参考图，**严格按相同顺序**左键点击对应 2D 像素点（右键可撤销）。
3. 点击 **「前端求解并应用外参」**，系统自动运行 `RANSAC + DLT + Levenberg-Marquardt` 剔除误点并精化外参，实时更新画面投影与深度图。
4. 如需复核或持久化，可点击 **「导出标定配点 JSON」**，或点击 GUI 中的 **「一键导出 park_cameras.json」** 并在左侧热重载上传。

#### 方式 B：Python OpenCV 标定工具箱（支持 `--input` 直接读取前端配点 JSON）
1. 安装 Python 依赖：
   ```bash
   cd python-tools
   pip install -r requirements.txt
   ```
2. （可选）棋盘格内参标定与去畸变检验：
   ```bash
   python calib_toolkit.py intrinsics --video chess.mp4
   python calib_toolkit.py verify --video cam01.mp4
   ```
3. 直接基于前端导出的配点 JSON 运行 `solvePnPRansac` + `solvePnPRefinementLM` 并自动回写 `park_cameras.json`：
   ```bash
   python calib_toolkit.py pnp --input calib_points_cam_fixed_01.json --cam-id cam_fixed_01
   ```
   终端会输出内点数量以及 RANSAC 与 LM 精化前后的平均重投影误差（建议 `< 2.0 px`）。

---

### 练习 5：离线 Draco GLB 模型与 30 路视线焦点感知调度
**目标**：验证外部园区模型接入（含 Unlit / 非等比缩放兼容性）、离线 Draco 解码及视线焦点感知调度。

1. **上传自定义 GLB**：上传本地 `.glb`（支持 Draco 压缩与 Unlit 材质；即使在纯内网离线环境也可由本地 `/draco/` 解码）。
2. **30 路视线焦点感知压测**：
   - 在 GUI 展开「练习 5 & 6：30路调度与 AI 闭环」，勾选 **「注入 27 路虚拟监控(共30路)」**。
   - 右键平移观察焦点（`controls.target`）对准不同楼栋并旋转视角，状态栏显示物理激活槽位始终 `<= 4 / 4`，且视线焦点落入的监控区域优先分配槽位；滑出槽位超过 3 秒的非测试流视频会自动暂停解码以节省 CPU/GPU 资源。

---

### 练习 6：3D 地形反投影、卡尔曼+匈牙利跨镜追踪与独立电子围栏告警
**目标**：验证带高差斜坡/平台的 3D 射线反投影、双目标交叉防 ID Switch 与业务解耦告警。

1. 在 GUI「练习 5 & 6：30路调度与 AI 闭环」中，勾选 **「开启模拟 YOLO 跨镜追踪」**，同时切换 **「显示 3D 电子围栏光墙」** 与 **「开启电子围栏越界检测」**。
2. *预期现象*：
   - 场景同时出现两个追踪目标：目标 `#1` 往返穿越 1、2 号枪机重叠区与中央电子围栏；目标 `#2` 沿东侧斜坡走上 `0.8m` 巡检平台再走下，圆柱底座与尾迹紧贴斜坡及平台表面（`Y = 0 ~ 0.8m`）。
   - 即使关闭「显示 3D 电子围栏光墙」，只要「开启电子围栏越界检测」保持勾选，目标闯入禁区时顶部仍正常弹出红色越界告警横幅。
   - 当调节 1 号或 2 号相机畸变参数 `k1` 时，模拟检测框与 `BBoxProjector` 同步更新，两目标交会穿越时全局 ID 保持稳定不串号。

---

## 三、常见报错与排查清单

| 现象 / 报错 | 常见原因 | 解决方案 |
| :--- | :--- | :--- |
| 上传本地视频后投影区呈**纯黑**或无画面 | 视频编码为 H.265 (HEVC) 或 10-bit，浏览器无法解码 | 使用 `docs/ASSET_REQUIREMENTS.md` 中的 FFmpeg 命令转为 H.264 + `yuv420p` |
| 上传 `.glb` 后**某面墙死活投不上视频** | 该墙面法线朝向建筑内部，被掠射角算法剔除 | 在建模软件中重新计算外向法线，或临时将「掠射角淡出阈值」调低验证 |
| 远程 `.m3u8` 流在 Chrome 桌面端无法播放 | Chrome 桌面端原生不支持 HLS（仅 Safari 支持） | 改用 MediaMTX / ZLMediaKit 提供的 WebRTC **WHEP** 地址或 MP4 流 |
| `solvePnP` / Web 标定重投影误差 > 10px | 选点共面（全在同一面墙）、2D/3D 点顺序错位或未标定广角畸变 | 重新采集立体分布的 6~8 个点，严格核对 `#1~#6` 对应顺序，广角镜头先标定 `k1/k2` |
| `npm install` 安装极慢或超时 | 默认 npm 源网络波动 | 执行 `npm install --registry=https://registry.npmmirror.com` |
