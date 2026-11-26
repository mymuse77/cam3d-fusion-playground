"""
实景“脏数据 / 容错”压测套组总控脚本：
1. 合成带真实广角桶形畸变与偏心光心的棋盘格标定视频 public/demo-blender-dirty/chess.mp4
2. 调用 Blender 5.3 渲染含“模实偏差（移位设备箱、违建雨棚、遮挡树）+ 支架滚转角”的广角底片
3. 使用 OpenCV 对底片施加真实 Brown-Conrady 广角畸变、偏心光心、冷暖白平衡与曝光噪点，并压制 H.264 MP4
4. 生成含手抖噪声 (±1.5px) 与 1 个严重遮挡误点 (+68px, -45px) 的标定控制点 JSON 与初始 park_cameras_dirty.json
"""
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
BLENDER_EXE = Path(r"D:\Env\Blender 5.3\blender.exe")
OUT_DIR = ROOT_DIR / "public" / "demo-blender-dirty"
TMP_DIR = ROOT_DIR / ".tmp_dirty_frames"

OUT_W, OUT_H = 1280, 720
FPS = 24

# 真实广角镜头内参与偏心光心真值
K_TRUE = np.array([
    [866.6667, 0.0, 648.0],
    [0.0, 866.6667, 354.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
DIST_TRUE = np.array([-0.26, 0.07, 0.0015, -0.0015], dtype=np.float64)


def generate_chessboard_video():
    """使用精确逆畸变光线求交渲染 30 帧 10x7 棋盘格 (9x6 内角点) 广角视频 chess.mp4"""
    chess_dir = TMP_DIR / "chess"
    chess_dir.mkdir(parents=True, exist_ok=True)

    # 为输出图每个像素预先计算精确去畸变归一化射线
    grid_u, grid_v = np.meshgrid(np.arange(OUT_W, dtype=np.float32), np.arange(OUT_H, dtype=np.float32))
    pts = np.stack([grid_u, grid_v], axis=-1).reshape(-1, 1, 2)
    undist = cv2.undistortPoints(pts, K_TRUE, DIST_TRUE)
    xn = undist[:, 0, 0].reshape(OUT_H, OUT_W)
    yn = undist[:, 0, 1].reshape(OUT_H, OUT_W)
    rays = np.stack([xn, yn, np.ones_like(xn)], axis=-1)  # (H, W, 3)

    sq = 0.03
    cols, rows = 10, 7  # 9x6 内角点
    inner_hw = cols * sq * 0.5
    inner_hh = rows * sq * 0.5
    board_hw = inner_hw + sq * 1.2
    board_hh = inner_hh + sq * 1.2

    # 30 个覆盖中心与四角边缘、带多角度倾角的标定姿态
    rng = np.random.default_rng(2026)
    num_frames = 30
    for f in range(num_frames):
        theta = 2.0 * math.pi * f / num_frames
        # 让棋盘格在视野中心与四个象限均匀游走
        tx = 0.16 * math.sin(theta) + float(rng.uniform(-0.02, 0.02))
        ty = 0.09 * math.cos(2.0 * theta) + float(rng.uniform(-0.02, 0.02))
        tz = 0.50 + 0.09 * math.sin(theta * 1.5)

        rx = math.radians(18.0 * math.sin(theta * 2.0 + 0.5))
        ry = math.radians(20.0 * math.cos(theta))
        rz = math.radians(15.0 * math.sin(theta + 1.0))
        R, _ = cv2.Rodrigues(np.array([rx, ry, rz], dtype=np.float64))

        normal = R[:, 2]
        t_vec = np.array([tx, ty, tz], dtype=np.float64)
        denom = rays @ normal
        numer = float(normal @ t_vec)

        valid_ray = np.abs(denom) > 1e-4
        t_hit = np.where(valid_ray, numer / np.where(valid_ray, denom, 1.0), -1.0)
        hit_pos = rays * t_hit[..., None] - t_vec

        bx = hit_pos @ R[:, 0]
        by = hit_pos @ R[:, 1]
        in_board = (t_hit > 0.15) & (np.abs(bx) <= board_hw) & (np.abs(by) <= board_hh)
        in_inner = in_board & (np.abs(bx) <= inner_hw) & (np.abs(by) <= inner_hh)

        ix = np.floor((bx + inner_hw) / sq).astype(np.int32)
        iy = np.floor((by + inner_hh) / sq).astype(np.int32)
        is_white_sq = ((ix + iy) % 2) == 1

        img = np.full((OUT_H, OUT_W, 3), (118, 122, 128), dtype=np.uint8)
        img[in_board] = (242, 242, 242)
        black_mask = in_inner & (~is_white_sq)
        img[black_mask] = (18, 18, 18)

        # 轻微抗锯齿平滑以提升亚像素角点精度
        img = cv2.GaussianBlur(img, (3, 3), 0.55)
        cv2.imwrite(str(chess_dir / f"frame_{f+1:04d}.png"), img)

    chess_mp4 = OUT_DIR / "chess.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", "15",
            "-i", str(chess_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-movflags", "+faststart", "-an",
            str(chess_mp4),
        ],
        check=True,
    )
    print(f"[DirtySuite] 已生成广角畸变棋盘格视频: {chess_mp4}")


def warp_and_grade_cameras():
    meta = json.loads((TMP_DIR / "dirty_meta.json").read_text(encoding="utf-8"))
    wide_fx = meta["wideFx"]
    wide_cx = meta["wideW"] * 0.5
    wide_cy = meta["wideH"] * 0.5

    # 预计算 OpenCV Brown-Conrady 畸变重映射表 (1280x720 输出像素 -> 1600x900 宽幅底片坐标)
    grid_u, grid_v = np.meshgrid(np.arange(OUT_W, dtype=np.float32), np.arange(OUT_H, dtype=np.float32))
    pts = np.stack([grid_u, grid_v], axis=-1).reshape(-1, 1, 2)
    undist = cv2.undistortPoints(pts, K_TRUE, DIST_TRUE)
    map_x = (undist[:, 0, 0] * wide_fx + wide_cx).reshape(OUT_H, OUT_W).astype(np.float32)
    map_y = (undist[:, 0, 1] * wide_fx + wide_cy).reshape(OUT_H, OUT_W).astype(np.float32)

    rng = np.random.default_rng(777)
    isp_profiles = {
        "cam_dirty_01": {
            "name": "现场冷色欠曝+歪斜广角 1 号枪机",
            "bgr_gain": np.array([1.12, 1.02, 0.86], dtype=np.float32),
            "contrast": 0.90,
            "brightness": -18.0,
            "colorAdjust": {"brightness": 0.08, "contrast": 1.12, "saturation": 1.05},
        },
        "cam_dirty_02": {
            "name": "现场暖色过曝+歪斜广角 2 号枪机",
            "bgr_gain": np.array([0.86, 1.01, 1.15], dtype=np.float32),
            "contrast": 1.08,
            "brightness": 15.0,
            "colorAdjust": {"brightness": -0.07, "contrast": 0.94, "saturation": 0.92},
        },
    }

    fov_y_true = round(2.0 * math.degrees(math.atan(OUT_H / (2.0 * K_TRUE[1, 1]))), 4)
    intr_true_dict = {
        "width": OUT_W,
        "height": OUT_H,
        "fx": round(float(K_TRUE[0, 0]), 4),
        "fy": round(float(K_TRUE[1, 1]), 4),
        "cx": round(float(K_TRUE[0, 2]), 4),
        "cy": round(float(K_TRUE[1, 2]), 4),
        "fovY": fov_y_true,
        "depthFovScale": 1.18,
        "near": 0.5,
        "far": 65.0,
    }
    dist_true_dict = {
        "k1": float(DIST_TRUE[0]),
        "k2": float(DIST_TRUE[1]),
        "p1": float(DIST_TRUE[2]),
        "p2": float(DIST_TRUE[3]),
    }

    cameras_json = []
    for cid, cam_meta in meta["cameras"].items():
        src_dir = TMP_DIR / cid
        dst_dir = TMP_DIR / f"{cid}_warped"
        dst_dir.mkdir(parents=True, exist_ok=True)
        prof = isp_profiles[cid]

        frames = sorted(src_dir.glob("frame_*.png"))
        for idx, fpath in enumerate(frames):
            raw = cv2.imread(str(fpath))
            warped = cv2.remap(raw, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

            # 模拟真实传感器色差、曝光偏移与轻度噪点
            graded = warped.astype(np.float32) * prof["bgr_gain"]
            graded = (graded - 127.5) * prof["contrast"] + 127.5 + prof["brightness"]
            graded += rng.normal(0.0, 2.2, graded.shape).astype(np.float32)
            out_img = np.clip(graded, 0, 255).astype(np.uint8)
            cv2.imwrite(str(dst_dir / fpath.name), out_img)
            if idx == 0:
                cv2.imwrite(str(OUT_DIR / f"{cid}_ref.png"), out_img)

        mp4_path = OUT_DIR / f"{cid}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(FPS),
                "-i", str(dst_dir / "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "21", "-movflags", "+faststart", "-an",
                str(mp4_path),
            ],
            check=True,
        )
        print(f"[DirtySuite] 已生成广角畸变与色差监控视频: {mp4_path}")

        # 投影真实 3D 控制点至带畸变像素坐标 + 叠加人工采点手抖噪声 (±1.5px) + 第 9 点严重遮挡误点
        pts_2d_true = []
        pts_2d_noisy = []
        for i, (xn, yn) in enumerate(cam_meta["normRays"]):
            obj = np.array([[xn, yn, 1.0]], dtype=np.float64)
            proj, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K_TRUE, DIST_TRUE)
            u_gt, v_gt = float(proj[0, 0, 0]), float(proj[0, 0, 1])
            pts_2d_true.append([round(u_gt, 3), round(v_gt, 3)])

            if i < 8:
                ju = float(rng.uniform(-1.5, 1.5))
                jv = float(rng.uniform(-1.5, 1.5))
                pts_2d_noisy.append([round(u_gt + ju, 2), round(v_gt + jv, 2)])
            else:
                # 第 9 个点模拟因遮挡/设备箱移位导致的严重误点击外点
                pts_2d_noisy.append([round(u_gt + 68.0, 2), round(v_gt - 45.0, 2)])

        calib_payload = {
            "camId": cid,
            "imageSize": [OUT_W, OUT_H],
            "intrinsics": intr_true_dict,
            "distortion": dist_true_dict,
            "groundTruthIntrinsics": intr_true_dict,
            "groundTruthDistortion": dist_true_dict,
            "groundTruthPosition": cam_meta["groundTruthPosition"],
            "groundTruthRollDeg": cam_meta["rollDeg"],
            "pts_3d": cam_meta["pts_3d"],
            "pts_2d": pts_2d_noisy,
            "pts_2d_clean": pts_2d_true,
        }
        (OUT_DIR / f"calib_points_{cid}.json").write_text(
            json.dumps(calib_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        cameras_json.append({
            "id": cid,
            "name": prof["name"],
            "type": "fixed",
            "enabled": True,
            "stream": {"type": "mp4", "url": f"/demo-blender-dirty/{cid}.mp4", "autoSuspend": True},
            "intrinsics": intr_true_dict,
            "distortion": dist_true_dict,
            "colorAdjust": prof["colorAdjust"],
        })

    park_cfg = {
        "schemaVersion": "3.0.0",
        "scene": {
            "coordinateSystem": "threejs-y-up",
            "unit": "meter",
            "glbUrl": "/demo-blender/blender_park.glb",
        },
        "cameras": cameras_json,
    }
    (OUT_DIR / "park_cameras_dirty.json").write_text(
        json.dumps(park_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[DirtySuite] 已输出脏数据标定配置: {OUT_DIR / 'park_cameras_dirty.json'}")


def main():
    if not BLENDER_EXE.exists():
        raise FileNotFoundError(f"未找到 Blender 5.3: {BLENDER_EXE}")
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_chessboard_video()
    subprocess.run(
        [
            str(BLENDER_EXE),
            "--background",
            "--factory-startup",
            "--python",
            str(ROOT_DIR / "blender-tools" / "blender_dirty_render.py"),
        ],
        check=True,
    )
    warp_and_grade_cameras()

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    print("[DirtySuite] 全部实景脏数据压测资产生成完毕！")


if __name__ == "__main__":
    main()
