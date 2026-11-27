"""
自动下载真实多路监控数据集 EPFL Terrace（4 路同步室外露台视频 + Tsai 3D 标定），
完成以下全链路转换：
1. 下载 terrace-tsai.zip 与 terrace1-c0..c3.avi 至 .tmp_epfl/
2. 使用 FFmpeg 截取多行人活跃片段（25s）转码为 Web 兼容 H.264 MP4，并从全片均匀抽帧计算零行人静态背景底图 (*_bg.png)
3. 精确解析 Tsai 相机标定（mm -> m, Z-Up -> Three.js Y-Up, kappa1 -> OpenCV Brown-Conrady k1/k2），输出 park_cameras_terrace.json
4. 调用本地 Blender 5.3 按同一世界坐标系生成露台实景白模 epfl_terrace.glb
"""
import io
import json
import math
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
BLENDER_EXE = Path(r"D:\Env\Blender 5.3\blender.exe")
TMP_DIR = ROOT_DIR / ".tmp_epfl"
OUT_DIR = ROOT_DIR / "public" / "demo-epfl-terrace"

TSAI_ZIP_URL = "https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/terrace-tsai.zip"
VIDEO_URLS = {
    "terrace_c0": "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace1-c0.avi",
    "terrace_c1": "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace1-c1.avi",
    "terrace_c2": "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace1-c2.avi",
    "terrace_c3": "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace1-c3.avi",
}

# EPFL Terrace 露台中心偏移（原世界系 mm: X∈[-500, 7000], Y∈[-1500, 9500], Z=0 -> 中心 (3.25m, 4.0m, 0)）
CENTER_OFFSET_M = np.array([3.25, 4.0, 0.0], dtype=np.float64)


def download_file(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 通过 HEAD 获取精确字节数，支持断点续传与断线重试
    expected_size = 0
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as r:
            expected_size = int(r.headers.get("Content-Length", 0))
    except Exception:
        pass

    if dst.exists() and expected_size > 0 and dst.stat().st_size == expected_size:
        print(f"[EPFL] 已完整存在跳过下载: {dst.name} ({dst.stat().st_size / 1048576:.1f} MB)")
        return

    print(f"[EPFL] 正在断点续传下载 {dst.name} ...")
    subprocess.run(
        [
            "curl.exe", "-fL", "-C", "-", "--retry", "8", "--retry-delay", "2",
            "--connect-timeout", "20", "-sS", "-o", str(dst), url,
        ],
        check=True,
    )
    print(f"[EPFL] 下载完成: {dst.name} ({dst.stat().st_size / 1048576:.1f} MB)")


def fit_opencv_k1_k2_from_tsai(focal_mm: float, kappa1: float):
    """
    Tsai 径向模型: r_u = r_d * (1 + kappa1 * r_d^2) (mm)
    归一化坐标: r_n = r_nd * (1 + alpha * r_nd^2), alpha = kappa1 * focal^2
    最小二乘拟合 OpenCV 正向畸变: r_nd = r_n * (1 + k1 * r_n^2 + k2 * r_n^4)
    """
    alpha = kappa1 * (focal_mm ** 2)
    r_nd = np.linspace(0.01, 0.65, 200)
    r_n = r_nd * (1.0 + alpha * (r_nd ** 2))
    # 目标: r_nd / r_n - 1 = k1 * r_n^2 + k2 * r_n^4
    A = np.column_stack([r_n ** 2, r_n ** 4])
    b = (r_nd / r_n) - 1.0
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return float(sol[0]), float(sol[1])


def parse_tsai_xml(xml_bytes: bytes, actual_w: int, actual_h: int):
    root = ET.fromstring(xml_bytes)
    geom = root.find("Geometry")
    intr = root.find("Intrinsic")
    extr = root.find("Extrinsic")

    xml_w = float(geom.attrib["width"])
    xml_h = float(geom.attrib["height"])
    dpx = float(geom.attrib["dpx"])
    dpy = float(geom.attrib["dpy"])

    focal = float(intr.attrib["focal"])
    kappa1 = float(intr.attrib["kappa1"])
    cx_xml = float(intr.attrib["cx"])
    cy_xml = float(intr.attrib["cy"])
    sx = float(intr.attrib.get("sx", "1.0"))

    # 按实际视频分辨率缩放内参
    scale_x = actual_w / xml_w
    scale_y = actual_h / xml_h
    fx = (focal * sx / dpx) * scale_x
    fy = (focal / dpy) * scale_y
    cx = cx_xml * scale_x
    cy = cy_xml * scale_y
    fov_y = 2.0 * math.degrees(math.atan(actual_h / (2.0 * fy)))
    k1, k2 = fit_opencv_k1_k2_from_tsai(focal, kappa1)

    # Tsai 外参: R = Rz(rz) @ Ry(ry) @ Rx(rx), t 单位 mm -> m
    rx = float(extr.attrib["rx"])
    ry = float(extr.attrib["ry"])
    rz = float(extr.attrib["rz"])
    tx = float(extr.attrib["tx"]) / 1000.0
    ty = float(extr.attrib["ty"]) / 1000.0
    tz = float(extr.attrib["tz"]) / 1000.0

    sa, ca = math.sin(rx), math.cos(rx)
    sb, cb = math.sin(ry), math.cos(ry)
    sg, cg = math.sin(rz), math.cos(rz)
    R_w2c = np.array([
        [cb * cg, cg * sa * sb - ca * sg, sa * sg + ca * cg * sb],
        [cb * sg, sa * sb * sg + ca * cg, ca * sb * sg - cg * sa],
        [-sb,     cb * sa,                ca * cb],
    ], dtype=np.float64)
    t_w2c = np.array([tx, ty, tz], dtype=np.float64)

    # 原 EPFL 世界系 (Z-Up) 相机位置
    cam_pos_epfl = -R_w2c.T @ t_w2c
    # 平移至露台中心原点，再转 Three.js Y-Up: T_z2y = [[1,0,0],[0,0,1],[0,-1,0]]
    T_z2y = np.array([
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64)

    # P_cv = R_w2c * (T_z2y^T * P_three + offset) + t_w2c
    # => Three.js matrixWorld: R_three = T_z2y @ R_w2c.T @ diag(1, -1, -1), pos_three = T_z2y @ (cam_pos_epfl - offset)
    R_three = T_z2y @ R_w2c.T @ np.diag([1.0, -1.0, -1.0])
    pos_three = T_z2y @ (cam_pos_epfl - CENTER_OFFSET_M)

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
            "depthFovScale": 1.18,
            "near": 0.3,
            "far": 45.0,
        },
        "distortion": {
            "k1": round(k1, 6),
            "k2": round(k2, 6),
            "p1": 0.0,
            "p2": 0.0,
        },
        "extrinsics": {
            "matrixWorldRowMajor": [round(float(v), 6) for v in M.flatten()],
        },
        "posThree": [round(float(v), 3) for v in pos_three],
    }


def extract_full_video_median_bg(avi_path: Path, out_bg_path: Path, sample_count: int = 45):
    """从整段 3 分钟 AVI 均匀抽取 45 帧计算时域中值，得到完全无行人的干净露台静态背景"""
    cap = cv2.VideoCapture(str(avi_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, total // sample_count)
    frames = []
    for f_idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        if len(frames) >= sample_count:
            break
    cap.release()
    bg = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    cv2.imwrite(str(out_bg_path), bg)
    print(f"[EPFL] 已提取静态背景 ({len(frames)} 帧中值, {w}x{h}): {out_bg_path.name}")
    return w, h


def build_terrace_glb():
    """调用 Blender 5.3 生成与 EPFL Terrace 标定坐标系完全对齐的露台三维白模 epfl_terrace.glb"""
    glb_path = OUT_DIR / "epfl_terrace.glb"
    blender_code = f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

def make_mat(name, rgba, rough=0.85):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = rough
        bsdf.inputs["Metallic"].default_value = 0.05
    return m

def add_box(name, center_three, size_three, mat, walkable=False):
    # Three.js Y-Up -> Blender Z-Up: (Xt, Yt, Zt) -> (Xt, -Zt, Yt)
    cx, cy, cz = center_three[0], -center_three[2], center_three[1]
    sx, sy, sz = size_three[0], size_three[2], size_three[1]
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(cx, cy, cz))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (sx, sy, sz)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    if walkable:
        obj["isWalkable"] = True
    return obj

mat_floor = make_mat("TerraceFloor", (0.36, 0.38, 0.40, 1.0), 0.9)
mat_wall = make_mat("TerraceWall", (0.52, 0.54, 0.56, 1.0), 0.85)
mat_parapet = make_mat("TerraceParapet", (0.44, 0.46, 0.48, 1.0), 0.8)

objs = []
# 露台主地面（覆盖 EPFL Terrace 标定区 7.5m x 11m 及周边，中心对齐 (0, 0, 0)）
bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
ground = bpy.context.active_object
ground.name = "TerraceGround"
ground.scale = (24.0, 26.0, 1.0)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
ground.data.materials.append(mat_floor)
ground["isWalkable"] = True
objs.append(ground)

# 露台西侧与北侧建筑外立面围墙（内墙面严格对齐 EPFL 露台地砖边界 X=-3.85m, Z=-5.6m）
objs.append(add_box("TerraceWestBuilding", (-4.65, 2.8, -0.5), (1.6, 5.6, 13.5), mat_wall))
objs.append(add_box("TerraceNorthBuilding", (0.2, 2.8, -6.4), (11.5, 5.6, 1.6), mat_wall))

bpy.ops.object.select_all(action="DESELECT")
for o in objs:
    o.select_set(True)
bpy.context.view_layer.objects.active = ground
bpy.ops.export_scene.gltf(
    filepath=r"{str(glb_path)}",
    export_format="GLB",
    use_selection=True,
    export_yup=True,
    export_apply=True,
    export_extras=True,
)
print("[EPFL] 已导出露台白模 GLB:", r"{str(glb_path)}")
"""
    script_path = TMP_DIR / "_build_terrace_glb.py"
    script_path.write_text(blender_code, encoding="utf-8")
    subprocess.run(
        [str(BLENDER_EXE), "--background", "--factory-startup", "--python", str(script_path)],
        check=True,
    )


def main():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 下载 Tsai 标定压缩包
    tsai_zip_path = TMP_DIR / "terrace-tsai.zip"
    if not tsai_zip_path.exists():
        print("[EPFL] 正在下载 terrace-tsai.zip ...")
        urllib.request.urlretrieve(TSAI_ZIP_URL, str(tsai_zip_path))

    tsai_xmls = {}
    with zipfile.ZipFile(tsai_zip_path) as zf:
        for idx in range(4):
            tsai_xmls[f"terrace_c{idx}"] = zf.read(f"terrace-tsai-c{idx}.xml")

    # 2. 并行断点续传下载 4 路 AVI
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download_file, url, TMP_DIR / f"{cid}.avi") for cid, url in VIDEO_URLS.items()]
        for fut in futures:
            fut.result()

    # 3. 转码 25s 活跃片段为 H.264 MP4、提取静态背景并转换标定
    cam_names = {
        "terrace_c0": "EPFL 露台 0 号实景相机",
        "terrace_c1": "EPFL 露台 1 号实景相机",
        "terrace_c2": "EPFL 露台 2 号实景相机",
        "terrace_c3": "EPFL 露台 3 号实景相机",
    }
    cameras_json = []

    for cid in VIDEO_URLS.keys():
        avi_path = TMP_DIR / f"{cid}.avi"
        mp4_path = OUT_DIR / f"{cid}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", "00:00:20", "-t", "25",
                "-i", str(avi_path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "20", "-movflags", "+faststart", "-an",
                str(mp4_path),
            ],
            check=True,
        )
        print(f"[EPFL] 已转码 25s 实景片段: {mp4_path.name} ({mp4_path.stat().st_size / 1048576:.2f} MB)")

        # 从同段 25s MP4 均匀抽取 50 帧计算时域中值，保证背景与播放片段曝光完全一致
        bg_path = OUT_DIR / f"{cid}_bg.png"
        w, h = extract_full_video_median_bg(mp4_path, bg_path, sample_count=50)

        calib = parse_tsai_xml(tsai_xmls[cid], w, h)
        print(
            f"[EPFL] {cid} 转换完成: pos={calib['posThree']} m, "
            f"fx={calib['intrinsics']['fx']}, k1={calib['distortion']['k1']}"
        )

        cameras_json.append({
            "id": cid,
            "name": cam_names[cid],
            "type": "fixed",
            "enabled": True,
            "stream": {
                "type": "mp4",
                "url": f"/demo-epfl-terrace/{cid}.mp4",
                "bgUrl": f"/demo-epfl-terrace/{cid}_bg.png",
                "autoSuspend": True,
            },
            "intrinsics": calib["intrinsics"],
            "distortion": calib["distortion"],
            "colorAdjust": {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0},
            "extrinsics": calib["extrinsics"],
        })

    # 3. 写出 park_cameras_terrace.json
    cfg = {
        "schemaVersion": "3.0.0",
        "scene": {
            "coordinateSystem": "threejs-y-up",
            "unit": "meter",
            "glbUrl": "/demo-epfl-terrace/epfl_terrace.glb",
        },
        "cameras": cameras_json,
    }
    cfg_path = OUT_DIR / "park_cameras_terrace.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[EPFL] 已写入实景相机配置: {cfg_path}")

    # 4. 生成露台白模 GLB
    build_terrace_glb()
    print("[EPFL] EPFL Terrace 真实多路实景数据集准备完毕！")


if __name__ == "__main__":
    main()
