#!/usr/bin/env python3
"""
从 .tmp_nvidia_w16/ground_truth.json 提取与当前 20 秒视频片段精确对齐的 3D 追踪真值，
区分普通工人 (person)、人形机器人 (robot) 与 AGV 搬运车 (agv)，
导出轻量前端轨迹文件 public/demo-nvidia-w16/tracks_w16.json。
"""

import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = ROOT / ".tmp_nvidia_w16"
OUT_DIR = ROOT / "public" / "demo-nvidia-w16"

SELECTED_CAMS = ["Camera", "Camera_01", "Camera_02", "Camera_03", "Camera_05", "Camera_06"]
SRC_W, SRC_H = 1920.0, 1080.0
START_FRAME = 148
FRAME_STEP = 2       # 30fps -> 15fps 关键帧，前端实时线性插值到 60fps
NUM_KEYFRAMES = 300  # 20.0 秒


def classify_type(raw_type: str) -> str:
    t = (raw_type or "").lower()
    if "person" in t:
        return "person"
    if any(k in t for k in ("fourier", "digit", "humanoid", "robot")):
        return "robot"
    return "agv"


def main():
    gt_path = TMP_DIR / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"未找到 {gt_path}")

    print("[GT] 读取 ground_truth.json ...")
    gt = json.loads(gt_path.read_text(encoding="utf-8"))

    frames_out = []
    class_counts = {"person": set(), "robot": set(), "agv": set()}

    for k in range(NUM_KEYFRAMES):
        fid = str(START_FRAME + k * FRAME_STEP)
        objs = gt.get(fid, [])
        frame_items = []
        for o in objs:
            raw_type = o.get("object type", "Unknown")
            cls = classify_type(raw_type)
            oid = int(o.get("object id", 0))
            loc = o.get("3d location")
            scale = o.get("3d bounding box scale")
            rot = o.get("3d bounding box rotation", [0, 0, 0])
            if not loc or not scale:
                continue

            # NVIDIA Z-Up (center=(0,-10,0)) -> Three.js Y-Up 脚底坐标
            xw, yw, zw_center = float(loc[0]), float(loc[1]), float(loc[2])
            sx, sy, sz_h = float(scale[0]), float(scale[1]), float(scale[2])
            xt = xw
            yt = max(0.0, zw_center - sz_h * 0.5)
            zt = -(yw + 10.0)
            w_m = max(sx, sy)
            h_m = sz_h

            # 仅保留仓库内有效目标
            if abs(xt) > 10.5 or abs(zt) > 10.5:
                continue

            # 收集 6 路目标相机中的精确 2D 裁切框 (转为 OpenGL UV: [u1, vMinGL, u2, vMaxGL])
            vis_2d = o.get("2d bounding box visible", {})
            boxes_gl = {}
            for cid in SELECTED_CAMS:
                b = vis_2d.get(cid)
                if not b or len(b) != 4:
                    continue
                x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                # 稍微外扩 4px 防止切到头顶安全帽或鞋尖
                u1 = max(0.0, (x1 - 4.0) / SRC_W)
                u2 = min(1.0, (x2 + 4.0) / SRC_W)
                v1_top = max(0.0, (y1 - 4.0) / SRC_H)
                v2_bot = min(1.0, (y2 + 4.0) / SRC_H)
                if (u2 - u1) < 0.008 or (v2_bot - v1_top) < 0.015:
                    continue
                v_min_gl = 1.0 - v2_bot
                v_max_gl = 1.0 - v1_top
                boxes_gl[cid.lower()] = [
                    round(u1, 4),
                    round(v_min_gl, 4),
                    round(u2, 4),
                    round(v_max_gl, 4),
                ]

            if not boxes_gl:
                continue

            class_counts[cls].add(oid)
            frame_items.append({
                "id": oid,
                "cls": cls,
                "rawType": raw_type,
                "p": [round(xt, 3), round(yt, 3), round(zt, 3)],
                "s": [round(w_m, 2), round(h_m, 2)],
                "yaw": round(float(rot[2]), 3),
                "b": boxes_gl,
            })
        frames_out.append(frame_items)

    out_path = OUT_DIR / "tracks_w16.json"
    out_path.write_text(
        json.dumps(
            {
                "fps": 30 / FRAME_STEP,
                "duration": round(NUM_KEYFRAMES * FRAME_STEP / 30.0, 2),
                "startFrame": START_FRAME,
                "frames": frames_out,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    # 同步把 tracksUrl 写入 park_cameras_w16.json
    cfg_path = OUT_DIR / "park_cameras_w16.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg.setdefault("scene", {})["tracksUrl"] = "/demo-nvidia-w16/tracks_w16.json"
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[GT] 导出完成: {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB), "
        f"共 {len(frames_out)} 关键帧 | 独立实体: 工人={len(class_counts['person'])}, "
        f"机器人={len(class_counts['robot'])}, AGV={len(class_counts['agv'])}"
    )


if __name__ == "__main__":
    main()
