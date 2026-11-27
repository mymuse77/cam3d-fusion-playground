"""
从已生成的监控视频中提取逐像素时域中值静态背景底图 (*_bg.png)，
并同步更新 park_cameras_blender.json / park_cameras_dirty.json 中的 stream.bgUrl 字段。
"""
import json
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent


def extract_median_bg(video_path: Path, out_bg_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"无法读取视频帧: {video_path}")

    median_frame = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    cv2.imwrite(str(out_bg_path), median_frame)
    print(f"[ExtractBG] 已提取时域中值背景 ({len(frames)} 帧): {out_bg_path}")


def patch_suite(suite_dir: Path, url_prefix: str, json_name: str):
    cfg_path = suite_dir / json_name
    if not cfg_path.exists():
        print(f"[ExtractBG] 跳过不存在的配置: {cfg_path}")
        return

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for cam in cfg.get("cameras", []):
        cid = cam["id"]
        mp4_path = suite_dir / f"{cid}.mp4"
        bg_path = suite_dir / f"{cid}_bg.png"
        if mp4_path.exists():
            extract_median_bg(mp4_path, bg_path)
            cam.setdefault("stream", {})["bgUrl"] = f"{url_prefix}/{cid}_bg.png"

    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ExtractBG] 已更新 stream.bgUrl 至: {cfg_path}")


if __name__ == "__main__":
    patch_suite(ROOT_DIR / "public" / "demo-blender", "/demo-blender", "park_cameras_blender.json")
    patch_suite(ROOT_DIR / "public" / "demo-blender-dirty", "/demo-blender-dirty", "park_cameras_dirty.json")
