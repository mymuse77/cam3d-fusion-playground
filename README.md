# 数字孪生：视频与 3D 模型融合全链路实操工程 (`dt-video-fusion-pro-v3`)

本项目基于 **Vite + Three.js + GLSL + Python OpenCV**，完整集成了数字孪生视频融合的 **6 大核心实操练习** 与 **生产级画质/鲁棒性优化**，内置 PBR 工业园区练习场景、动态测试画布、球机变焦巡航与模拟 YOLO 跨镜数据，**零硬件依赖、开箱即用**。

---

## 配套文档导航

- **逐步骤测试实操手册**：[`docs/TEST_MANUAL.md`](./docs/TEST_MANUAL.md)（包含 6 大练习详细步骤、推荐调试参数、Python 标定联调流程与报错排查清单）
- **视频与 GLB 模型规范**：[`docs/ASSET_REQUIREMENTS.md`](./docs/ASSET_REQUIREMENTS.md)（包含监控视频 H.264 转码 FFmpeg 命令、GLB 材质法线/坐标系/Draco 导出核对清单）

---

## 快速启动

### 1. 启动前端 3D 融合项目（需 Node.js v18+）
```bash
npm install
npm run dev
```
启动后在浏览器打开终端输出的地址（通常为 `http://localhost:5173`）。

### 2. Python OpenCV 离线标定工具（可选，用于练习 4 真实监控接入）
```bash
cd python-tools
pip install -r requirements.txt

python calib_toolkit.py intrinsics --video chess.mp4   # 棋盘格内参标定
python calib_toolkit.py verify --video cam01.mp4       # 去畸变效果验证
python calib_toolkit.py pick2d --source cam01.mp4      # 2D 画面采点
python calib_toolkit.py pnp                            # solvePnP 外参求解并写入 JSON
```

---

## v3 完整能力清单

1. **视觉画质优化**：掠射角（Grazing Angle）法线衰减消除侧墙拉丝、PCF 3×3 软边缘深度遮挡防穿模、每路独立亮度/对比度/饱和度色彩匹配。
2. **工程鲁棒性**：显存防泄漏（`disposeObject3D` + `URL.revokeObjectURL`）、支持 Draco 压缩的 `.glb` 上传与自动居中落底、球机 `Pan / Tilt / Zoom` 全通道联动与畸变衰减。
3. **现场实施与 AI 闭环**：可视化相机微调与 `park_cameras.json` 热重载/一键导出、30 路视锥体剔除动态槽位调度、YOLO 2D 反投影、跨镜去重、历史运动尾迹、头顶 ID 标签与 3D 电子围栏越界告警。
