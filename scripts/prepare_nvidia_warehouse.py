#!/usr/bin/env python3
"""
NVIDIA PhysicalAI-SmartSpaces (MTMC_Tracking_2025 / val / Warehouse_016) 一键联调准备脚本：
1. 从 Hugging Face 公开仓库按需拉取 calibration.json 与 map.png（若本地尚未缓存）；
2. 选取 6 路视场高度重叠的核心仓库相机（6 路 > 4 物理投影槽位，完整验证视线焦点调度 + 流休眠恢复）：
   Camera, Camera_01, Camera_02, Camera_03, Camera_05, Camera_06
3. 使用 FFmpeg HTTP Range 直接远程截取每路 20 秒 720p H.264 MP4（无需下载 130MB+ 原片）；
4. 从每路 20 秒成片均匀采样 50 帧做时域中值滤波，生成完全匹配曝光的静态背景底板 *_bg.png；
5. 将 NVIDIA Z-Up OpenCV 内外参严格转换到 Three.js Y-Up（仓库中心 (0, -10, 0) 归零），输出 park_cameras_w16.json；
6. 调用本地 Blender 5.3 根据 map.png 精确坐标生成与 6 路相机毫米级对齐的 warehouse_016.glb。
"""

import json
import math
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = ROOT / ".tmp_nvidia_w16"
OUT_DIR = ROOT / "public" / "demo-nvidia-w16"
BLENDER_EXE = Path(r"D:\Env\Blender 5.3\blender.exe")

HF_BASE = (
    "https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/resolve/main/"
    "MTMC_Tracking_2025/val/Warehouse_016"
)

# 选取 6 路覆盖中心与南侧托盘作业区的高重叠相机（测试 Top-4 调度与多视角融合）
SELECTED_CAMS = [
    ("Camera", "仓库西侧主视角 (Camera)"),
    ("Camera_01", "仓库东侧对射 (Camera_01)"),
    ("Camera_02", "仓库中区俯视 (Camera_02)"),
    ("Camera_03", "仓库西北斜视 (Camera_03)"),
    ("Camera_05", "仓库西南北望 (Camera_05)"),
    ("Camera_06", "仓库南墙西北望 (Camera_06)"),
]

CLIP_START = "00:00:05"
CLIP_DURATION = "20"
TARGET_W = 1280
TARGET_H = 720

# 仓库中心偏移 (NVIDIA Z-Up 世界坐标: X in [-11, 11], Y in [-21, 1] => 中心 (0, -10, 0))
CENTER_OFFSET_W = np.array([0.0, -10.0, 0.0], dtype=np.float64)


def ensure_metadata():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ("calibration.json", "map.png"):
        dst = TMP_DIR / fname
        if dst.exists() and dst.stat().st_size > 1000:
            continue
        url = f"{HF_BASE}/{fname}"
        print(f"[NVIDIA] 下载元数据: {fname} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dst, "wb") as f:
            shutil.copyfileobj(resp, f)


def pull_clip_and_bg(cam_id: str) -> dict:
    """使用 FFmpeg HTTP Range 截取 20 秒 720p H.264 片段并提取 50 帧时域中值静态背景"""
    out_mp4 = OUT_DIR / f"{cam_id.lower()}.mp4"
    out_bg = OUT_DIR / f"{cam_id.lower()}_bg.png"
    remote_url = f"{HF_BASE}/videos/{cam_id}.mp4"

    if not out_mp4.exists() or out_mp4.stat().st_size < 100_000:
        print(f"[NVIDIA] HTTP Range 截取 {cam_id} ({CLIP_DURATION}s @ {TARGET_W}x{TARGET_H}) ...")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", CLIP_START,
            "-t", CLIP_DURATION,
            "-i", remote_url,
            "-vf", f"scale={TARGET_W}:{TARGET_H}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True)
        size_mb = out_mp4.stat().st_size / (1024 * 1024)
        print(f"[NVIDIA] 完成片段 {out_mp4.name} ({size_mb:.2f} MB)")
    else:
        print(f"[NVIDIA] 复用已存在片段 {out_mp4.name}")

    # 从同一段 MP4 均匀采样 50 帧做时域中值，确保与视频流色域/曝光完全一致
    cap = cv2.VideoCapture(str(out_mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_count = min(50, max(12, total))
    indices = np.linspace(0, max(0, total - 1), sample_count, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"无法从 {out_mp4} 读取视频帧")
    bg = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    cv2.imwrite(str(out_bg), bg)
    print(f"[NVIDIA] 已生成背景底板 {out_bg.name} ({len(frames)} 帧中值, {w}x{h})")
    return {"id": cam_id, "width": w, "height": h}


def convert_sensor_to_three(sensor: dict, actual_w: int, actual_h: int) -> dict:
    """将 NVIDIA Z-Up OpenCV 内外参转换为 Three.js Y-Up matrixWorldRowMajor"""
    attrs = {a["name"]: a["value"] for a in sensor.get("attributes", [])}
    src_w = float(attrs.get("frameWidth", 1920))
    src_h = float(attrs.get("frameHeight", 1080))
    sx = actual_w / src_w
    sy = actual_h / src_h

    K = np.array(sensor["intrinsicMatrix"], dtype=np.float64)
    fx = float(K[0, 0]) * sx
    fy = float(K[1, 1]) * sy
    cx = float(K[0, 2]) * sx
    cy = float(K[1, 2]) * sy
    fov_y = math.degrees(2.0 * math.atan(actual_h / (2.0 * fy)))

    M_ext = np.array(sensor["extrinsicMatrix"], dtype=np.float64)
    R_w2c = M_ext[:, :3]
    t_w2c = M_ext[:, 3]

    # Z-Up 世界相机位置
    cam_pos_w = -R_w2c.T @ t_w2c

    # Z-Up -> Three.js Y-Up: (X, Y, Z) -> (X, Z, -Y), det = +1
    T_z2y = np.array([
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64)

    # OpenCV 相机系 (+x右, +y下, +z前) -> Three.js 相机系 (+x右, +y上, -z前)
    R_three = T_z2y @ R_w2c.T @ np.diag([1.0, -1.0, -1.0])
    pos_three = T_z2y @ (cam_pos_w - CENTER_OFFSET_W)

    M = np.eye(4)
    M[:3, :3] = R_three
    M[:3, 3] = pos_three

    return {
        "intrinsics": {
            "width": actual_w,
            "height": actual_h,
            "fx": round(fx, 4),
            "fy": round(fy, 4),
            "cx": round(cx, 4),
            "cy": round(cy, 4),
            "fovY": round(fov_y, 4),
            "depthFovScale": 1.08,
            "near": 0.3,
            "far": 45.0,
        },
        "distortion": {
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
        },
        "extrinsics": {
            "matrixWorldRowMajor": [round(float(v), 6) for v in M.flatten()],
        },
        "posThree": [round(float(v), 3) for v in pos_three],
    }


def px_box_to_three(x0, y0, x1, y1, sf, tx, ty, img_h=1080):
    """将 map.png 像素矩形 (x0,y0)-(x1,y1) 转为 Three.js 中心与尺寸 (cx, cz, sx, sz)"""
    wx0 = x0 / sf - tx
    wx1 = x1 / sf - tx
    wy0 = (img_h - y1) / sf - ty
    wy1 = (img_h - y0) / sf - ty
    # Z_three = -(Y_world + 10.0)
    z0 = -(wy1 + 10.0)
    z1 = -(wy0 + 10.0)
    cx = (wx0 + wx1) * 0.5
    cz = (z0 + z1) * 0.5
    sx = abs(wx1 - wx0)
    sz = abs(z1 - z0)
    return (round(cx, 3), round(cz, 3), round(sx, 3), round(sz, 3))


def build_warehouse_glb(sf: float, tx: float, ty: float):
    """使用 Blender 5.3 根据 map.png 精确布局生成仓库三维白模 warehouse_016.glb"""
    glb_path = OUT_DIR / "warehouse_016.glb"

    # 从 map.png 精确量取的静态货架、工作台与周界墙体像素坐标（1920x1080, 仓库区 x=[420..1500], y=[0..1080]）
    # (name, (x0, y0, x1, y1), height_m, color_rgba)
    static_props_px = [
        # 北侧靠墙三排高货架（左上）
        ("Rack_NW_1", (555, 155, 618, 326), 2.8, (0.42, 0.36, 0.30, 1.0)),
        ("Rack_NW_2", (685, 155, 748, 326), 2.8, (0.42, 0.36, 0.30, 1.0)),
        ("Rack_NW_3", (815, 155, 878, 326), 2.8, (0.42, 0.36, 0.30, 1.0)),
        # 北墙长货架（右上）
        ("Rack_NorthWall", (998, 84, 1390, 138), 2.6, (0.38, 0.35, 0.32, 1.0)),
        # 东北大型木箱堆垛区
        ("CrateStack_NE_Main", (1152, 232, 1255, 405), 1.85, (0.58, 0.46, 0.32, 1.0)),
        ("CrateStack_NE_Side", (1255, 260, 1306, 366), 1.35, (0.58, 0.46, 0.32, 1.0)),
        # 中央两排货架/工作台
        ("Table_Center_N", (1046, 190, 1084, 314), 1.05, (0.48, 0.50, 0.52, 1.0)),
        ("Table_Center_S", (1040, 355, 1082, 478), 1.05, (0.48, 0.50, 0.52, 1.0)),
        # 西侧两张工作台
        ("Table_West_1", (568, 400, 675, 466), 0.90, (0.52, 0.54, 0.56, 1.0)),
        ("Table_West_2", (770, 408, 864, 468), 0.90, (0.52, 0.54, 0.56, 1.0)),
        # 东墙两段长货架（贴紧东墙内侧，高约 2.05m）
        ("Rack_East_N", (1362, 308, 1420, 502), 2.05, (0.40, 0.36, 0.32, 1.0)),
        ("Rack_East_S", (1362, 750, 1420, 946), 2.05, (0.40, 0.36, 0.32, 1.0)),
        # 南墙东侧低矮货架/箱垛
        ("Rack_South_E", (1142, 955, 1272, 1006), 1.15, (0.56, 0.46, 0.34, 1.0)),
        # 西南 6 个深蓝料箱
        ("Bin_SW_1", (563, 608, 606, 668), 0.85, (0.20, 0.24, 0.30, 1.0)),
        ("Bin_SW_2", (669, 608, 712, 668), 0.85, (0.20, 0.24, 0.30, 1.0)),
        ("Bin_SW_3", (563, 734, 606, 794), 0.85, (0.20, 0.24, 0.30, 1.0)),
        ("Bin_SW_4", (669, 734, 712, 794), 0.85, (0.20, 0.24, 0.30, 1.0)),
        ("Bin_SW_5", (563, 868, 606, 928), 0.85, (0.20, 0.24, 0.30, 1.0)),
        ("Bin_SW_6", (669, 868, 712, 928), 0.85, (0.20, 0.24, 0.30, 1.0)),
        # 三个黄色托盘作业区内的 11 组独立木托盘+纸箱体块（消除多视角投影到平地产生的拖影）
        # 西黄框（3 组）
        ("Pallet_W1", (809, 628, 868, 683), 0.72, (0.62, 0.50, 0.36, 1.0)),
        ("Pallet_W2", (807, 758, 869, 813), 0.74, (0.62, 0.50, 0.36, 1.0)),
        ("Pallet_W3", (798, 849, 859, 902), 0.70, (0.62, 0.50, 0.36, 1.0)),
        # 中黄框（4 组）
        ("Pallet_M1", (1043, 621, 1103, 662), 0.70, (0.62, 0.50, 0.36, 1.0)),
        ("Pallet_M2", (1043, 689, 1106, 745), 0.76, (0.62, 0.50, 0.36, 1.0)),
        ("Pallet_M3", (1043, 761, 1106, 817), 0.76, (0.62, 0.50, 0.36, 1.0)),
        ("Pallet_M4", (1032, 833, 1094, 881), 0.72, (0.62, 0.50, 0.36, 1.0)),
        # 东黄框（4 组）
        ("Pallet_E1", (1214, 638, 1274, 678), 0.70, (0.62, 0.50, 0.36, 1.0)),
        ("Pallet_E2", (1214, 696, 1272, 748), 0.74, (0.62, 0.50, 0.36, 1.0)),
        ("Pallet_E3", (1214, 756, 1272, 809), 0.74, (0.62, 0.50, 0.36, 1.0)),
        ("Pallet_E4", (1214, 822, 1274, 868), 0.70, (0.62, 0.50, 0.36, 1.0)),
    ]

    props_three = []
    for name, (x0, y0, x1, y1), h, rgba in static_props_px:
        cx, cz, sx, sz = px_box_to_three(x0, y0, x1, y1, sf, tx, ty)
        props_three.append((name, cx, cz, sx, sz, h, rgba))

    props_json = json.dumps(props_three)

    blender_code = f"""
import bpy, json
bpy.ops.wm.read_factory_settings(use_empty=True)

def make_mat(name, rgba, rough=0.82, metal=0.05):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = rough
        bsdf.inputs["Metallic"].default_value = metal
    return m

def add_box(name, center_three, size_three, mat, walkable=False):
    # Blender Z-Up -> glTF Y-Up 导出规则: Three(x, y, z) 对应 Blender(x, -z, y)
    bx, by, bz = center_three[0], -center_three[2], center_three[1]
    sx, sy, sz = size_three[0], size_three[2], size_three[1]
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(bx, by, bz))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (sx, sy, sz)
    obj.data.materials.append(mat)
    if walkable:
        obj["isWalkable"] = True
    return obj

mat_floor = make_mat("Warehouse_Floor_Mat", (0.46, 0.47, 0.49, 1.0), rough=0.88)
mat_wall = make_mat("Warehouse_Wall_Mat", (0.72, 0.74, 0.76, 1.0), rough=0.85)
mat_dado = make_mat("Warehouse_Dado_Mat", (0.14, 0.32, 0.56, 1.0), rough=0.65)

# 1. 可行走仓库混凝土地面 (22m x 22m, 顶面严格位于 Y=0.0)
add_box("Warehouse_Floor", (0.0, -0.15, 0.0), (22.0, 0.30, 22.0), mat_floor, walkable=True)

# 2. 四周围墙（内壁精确对齐实测 map.png 的 ±9.82m，高 6.6m，留开放顶面方便俯瞰）
wall_h = 6.6
wall_t = 0.45
inner = 9.82
wall_c = inner + wall_t * 0.5
span = inner * 2.0 + wall_t * 2.0

add_box("Wall_North", (0.0, wall_h * 0.5, -wall_c), (span, wall_h, wall_t), mat_wall)
add_box("Wall_South", (0.0, wall_h * 0.5,  wall_c), (span, wall_h, wall_t), mat_wall)
add_box("Wall_West",  (-wall_c, wall_h * 0.5, 0.0), (wall_t, wall_h, span), mat_wall)
add_box("Wall_East",  ( wall_c, wall_h * 0.5, 0.0), (wall_t, wall_h, span), mat_wall)

# 墙裙蓝色装饰带（高度 1.20m~1.65m，与实景蓝带精确吻合）
add_box("Dado_North", (0.0, 1.42, -inner + 0.015), (inner * 2.0, 0.45, 0.03), mat_dado)
add_box("Dado_South", (0.0, 1.42,  inner - 0.015), (inner * 2.0, 0.45, 0.03), mat_dado)
add_box("Dado_West",  (-inner + 0.015, 1.42, 0.0), (0.03, 0.45, inner * 2.0), mat_dado)
add_box("Dado_East",  ( inner - 0.015, 1.42, 0.0), (0.03, 0.45, inner * 2.0), mat_dado)

# 3. 精确对齐的静态货架、工作台与箱垛体块
props = json.loads({props_json!r})
for name, cx, cz, sx, sz, h, rgba in props:
    m = make_mat(f"Mat_{{name}}", tuple(rgba), rough=0.80)
    add_box(name, (cx, h * 0.5, cz), (sx, h, sz), m, walkable=False)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.export_scene.gltf(
    filepath=r"{str(glb_path)}",
    export_format='GLB',
    use_selection=True,
    export_yup=True,
    export_apply=True,
    export_extras=True,
)
print("EXPORT_NVIDIA_W16_GLB_OK")
"""
    script_path = TMP_DIR / "build_warehouse_glb.py"
    script_path.write_text(blender_code, encoding="utf-8")

    if not BLENDER_EXE.exists():
        raise FileNotFoundError(f"未找到 Blender: {BLENDER_EXE}")

    print("[NVIDIA] 正在调用 Blender 5.3 构建对齐三维白模 warehouse_016.glb ...")
    res = subprocess.run(
        [str(BLENDER_EXE), "--background", "--factory-startup", "--python", str(script_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    if "EXPORT_NVIDIA_W16_GLB_OK" not in res.stdout:
        print(res.stdout[-500:])
        print(res.stderr[-500:])
        raise RuntimeError("Blender GLB 导出未确认成功")
    print(f"[NVIDIA] 成功生成 {glb_path.name} ({glb_path.stat().st_size / 1024:.1f} KB)")


def main():
    ensure_metadata()
    cal = json.loads((TMP_DIR / "calibration.json").read_text(encoding="utf-8"))
    sensors_by_id = {s["id"]: s for s in cal["sensors"]}

    # 1. 并行截取 6 路 20 秒 720p 视频并生成时域中值静态背景
    clip_meta = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(pull_clip_and_bg, cid): cid for cid, _ in SELECTED_CAMS}
        for fut in as_completed(futs):
            info = fut.result()
            clip_meta[info["id"]] = info

    # 2. 生成 park_cameras_w16.json
    cameras_cfg = []
    for cid, label in SELECTED_CAMS:
        sensor = sensors_by_id[cid]
        info = clip_meta[cid]
        conv = convert_sensor_to_three(sensor, info["width"], info["height"])
        cameras_cfg.append({
            "id": cid.lower(),
            "name": label,
            "type": "fixed",
            "enabled": True,
            "intrinsics": conv["intrinsics"],
            "distortion": conv["distortion"],
            "extrinsics": conv["extrinsics"],
            "stream": {
                "url": f"/demo-nvidia-w16/{cid.lower()}.mp4",
                "bgUrl": f"/demo-nvidia-w16/{cid.lower()}_bg.png",
                "type": "mp4",
                "autoSuspend": True,
            },
        })
        print(f"  -> {cid:10s} Three.js pos={conv['posThree']} fovY={conv['intrinsics']['fovY']}°")

    out_json = OUT_DIR / "park_cameras_w16.json"
    out_json.write_text(
        json.dumps(
            {
                "version": "3.0-nvidia-w16",
                "dataset": "NVIDIA PhysicalAI-SmartSpaces / MTMC_Tracking_2025 / val / Warehouse_016 (6-cam subset)",
                "scene": {
                    "glbUrl": "/demo-nvidia-w16/warehouse_016.glb",
                    "description": "NVIDIA Warehouse_016 22m x 22m 仓储场景（6路高重叠相机，测试 Top-4 调度与 3D 融合）",
                },
                "cameras": cameras_cfg,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[NVIDIA] 已写出相机配置: {out_json}")

    # 3. 构建三维白模 warehouse_016.glb
    s0 = cal["sensors"][0]
    build_warehouse_glb(
        s0["scaleFactor"],
        s0["translationToGlobalCoordinates"]["x"],
        s0["translationToGlobalCoordinates"]["y"],
    )
    print("\n[NVIDIA] 全部准备就绪！请在前端点击「加载 NVIDIA 仓库实景(6路)」按钮体验。")


if __name__ == "__main__":
    main()
