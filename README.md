# 数字孪生：视频与 3D 模型融合全链路实操工程 (`cam3d-fusion-playground` v3.0)

本项目基于 **Vite + Three.js + GLSL + Python OpenCV**，完整集成了数字孪生视频融合的 **6 大核心实操练习** 与 **生产级画质/算法/流媒体优化**，内置含斜坡与 0.8m 巡检高台的 PBR 工业园区练习场景、动态测试画布、球机变焦巡航、Web 在线 RANSAC+LM PnP 标定台与模拟 YOLO 双目标跨镜追踪，**零硬件依赖、开箱即用**。

---

## 配套文档导航

- **逐步骤测试实操手册**：[`docs/TEST_MANUAL.md`](./docs/TEST_MANUAL.md)（包含 6 大练习详细步骤、Web 联合标定与 Python `solvePnPRansac` 联调流程、推荐调试参数与报错排查清单）
- **视频与 GLB 模型规范**：[`docs/ASSET_REQUIREMENTS.md`](./docs/ASSET_REQUIREMENTS.md)（包含监控视频 H.264 转码 FFmpeg 命令、GLB 材质/法线/Unlit 兼容/坐标系/本地 Draco 说明）
- **后续优化待办清单**：[`docs/TODO.md`](./docs/TODO.md)（记录 PTZ 调度联动、24/32-bit 自适应深度、第一视角比对、BVH 加速、3D 视频立牌与延迟贴花等演进规划）

---

## 快速启动

### 1. 启动前端 3D 融合项目（需 Node.js v18+）
```bash
npm install
npm run dev
```
启动后在浏览器打开终端输出的地址（通常为 `http://localhost:5173`）。构建阶段由 `vite.config.js` 自动将 Three.js 自带的 Draco 解码器拷贝至 `public/draco/`，纯内网离线亦可直接解压 Draco `.glb` 模型。

### 2. 运行核心算法自动化测试
```bash
npm test
```
自动验证正逆畸变往返精度、3D 地形/高台网格射线反投影、卡尔曼滤波+匈牙利跨镜追踪防串号、视线焦点感知调度、PTZ 基准隔离以及 Web 端 RANSAC+LM PnP 标定求解。

### 3. Python OpenCV 离线标定工具（可选，用于外部复核与批处理）
```bash
cd python-tools
pip install -r requirements.txt

python calib_toolkit.py intrinsics --video chess.mp4                    # 棋盘格内参标定
python calib_toolkit.py verify --video cam01.mp4                        # 去畸变效果验证
python calib_toolkit.py pick2d --source cam01.mp4                       # 2D 画面采点
python calib_toolkit.py pnp --input calib_points_cam_fixed_01.json      # RANSAC+LM PnP 求解并写入 JSON
```

---

## v3.0 完整能力清单

1. **视觉画质与 Shader 兼容性优化**：掠射角（Grazing Angle）精确逆转置法线衰减（支持非等比缩放、`InstancedMesh` 与 Unlit 材质）、PCF 3×3 软边缘深度遮挡防穿模、静态下标纹理采样分发、Linear↔sRGB 感知空间色彩匹配。
2. **流媒体生命周期与工程鲁棒性**：`VideoStreamManager` 支持多路独立绑定本地视频与远程流（MP4 / HLS / WebRTC WHEP 指数退避重连），并与调度槽位联动实现非活跃流自动暂停休眠；显存防泄漏（`disposeObject3D` + `URL.revokeObjectURL`）、本地离线 Draco 解码、`PTZController` 基准姿态隔离导出。
3. **高阶调度、标定闭环与 AI 追踪**：紧致视锥包围球 + 视线焦点感知（Focus-Aware）30 路调度；Web 端 2D/3D 联合 RANSAC+LM PnP 标定台；3D 可行走地形/斜坡/高台网格射线反投影 + 凝聚聚类 + 4D 卡尔曼滤波 + 匈牙利全局最优跨镜追踪 + 显隐解耦的 3D 电子围栏告警。
