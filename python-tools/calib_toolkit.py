"""
数字孪生视频融合 v3.0 - Python OpenCV 离线标定工具箱
用法示例：
  1. 棋盘格内参标定:   python calib_toolkit.py intrinsics --video chess.mp4
  2. 去畸变效果验证:   python calib_toolkit.py verify --video cam01.mp4
  3. 2D 画面采点:      python calib_toolkit.py pick2d --source cam01.mp4
  4. PnP 求解并写入:   python calib_toolkit.py pnp --input calib_points.json --cam-id cam_fixed_01
"""
import argparse
import json
import math
from pathlib import Path
import cv2
import numpy as np


def run_intrinsics(video_path, pattern=(9, 6), interval=15, max_frames=25):
    cap = cv2.VideoCapture(video_path)
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * 0.03
    obj_pts, img_pts, img_size, idx = [], [], None, 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while len(obj_pts) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % interval != 0:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_FAST_CHECK)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_pts.append(objp)
            img_pts.append(corners)
            print(f"有效帧 {len(obj_pts)}/{max_frames}")
    cap.release()

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, img_size, None, None, flags=cv2.CALIB_FIX_K3)
    print(f"\nRMS 重投影误差: {rms:.4f} px\nK = {K.tolist()}\ndist = {dist.ravel()[:4].tolist()}")


def run_verify(video_path, K, dist):
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K_np, dist_np = np.array(K, np.float64), np.array(dist, np.float64)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K_np, dist_np, (w, h), 1.0, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K_np, dist_np, None, new_K, (w, h), cv2.CV_16SC2)

    cv2.namedWindow("Original vs Undistorted (Q退出)", cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        cv2.imshow("Original vs Undistorted (Q退出)", np.hstack((frame, undist)))
        if (cv2.waitKey(20) & 0xFF) in (27, ord('q'), ord('Q')):
            break
    cap.release()
    cv2.destroyAllWindows()


def run_pick_2d(source_path):
    img = cv2.imread(source_path)
    if img is None:
        cap = cv2.VideoCapture(source_path)
        _, img = cap.read()
        cap.release()

    pts = []

    def redraw():
        vis = img.copy()
        for i, (u, v) in enumerate(pts):
            cv2.circle(vis, (u, v), 6, (0, 0, 255), -1)
            cv2.putText(vis, f"#{i+1}", (u + 10, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("2D Picker (Enter输出 / 右键撤销)", vis)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([x, y])
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()
            redraw()

    cv2.namedWindow("2D Picker (Enter输出 / 右键撤销)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("2D Picker (Enter输出 / 右键撤销)", on_mouse)
    redraw()
    while (cv2.waitKey(20) & 0xFF) not in (13, 27):
        pass
    cv2.destroyAllWindows()
    print(f"W, H = {img.shape[1]}, {img.shape[0]}\npts_2d = {pts}")


def run_pnp_and_export(cam_id, json_path, image_size, pts_3d, pts_2d, K, dist):
    w, h = image_size
    obj_pts = np.array(pts_3d, dtype=np.float64)
    img_pts = np.array(pts_2d, dtype=np.float64)
    K_np = np.array(K, dtype=np.float64)
    dist_np = np.array(dist, dtype=np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts,
        img_pts,
        K_np,
        dist_np,
        iterationsCount=300,
        reprojectionError=4.0,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnPRansac 求解失败，请检查 2D/3D 点是否共面或顺序错位")

    inlier_idx = inliers.ravel() if inliers is not None and len(inliers) >= 4 else np.arange(len(obj_pts))
    inlier_obj = obj_pts[inlier_idx]
    inlier_img = img_pts[inlier_idx]

    proj_before, _ = cv2.projectPoints(inlier_obj, rvec, tvec, K_np, dist_np)
    err_before = float(np.mean(np.linalg.norm(inlier_img - proj_before.reshape(-1, 2), axis=1)))

    if hasattr(cv2, "solvePnPRefinementLM"):
        rvec, tvec = cv2.solvePnPRefinementLM(inlier_obj, inlier_img, K_np, dist_np, rvec, tvec)

    proj_after, _ = cv2.projectPoints(inlier_obj, rvec, tvec, K_np, dist_np)
    err_after = float(np.mean(np.linalg.norm(inlier_img - proj_after.reshape(-1, 2), axis=1)))

    R, _ = cv2.Rodrigues(rvec)
    M = np.eye(4)
    M[:3, :3] = R.T @ np.diag([1.0, -1.0, -1.0])
    M[:3, 3] = (-R.T @ tvec).flatten()

    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    updated = False
    for cam in data["cameras"]:
        if cam["id"] == cam_id:
            cam["intrinsics"].update({
                "width": w,
                "height": h,
                "fx": float(K[0][0]),
                "fy": float(K[1][1]),
                "cx": float(K[0][2]),
                "cy": float(K[1][2]),
                "fovY": round(2.0 * math.degrees(math.atan(h / (2.0 * K[1][1]))), 3),
            })
            cam["distortion"] = {"k1": float(dist[0]), "k2": float(dist[1]), "p1": float(dist[2]), "p2": float(dist[3])}
            cam["extrinsics"] = {
                "matrixWorldRowMajor": [round(float(x), 6) for x in M.flatten()],
                "reprojectionErrorPx": round(err_after, 3),
            }
            updated = True
            break

    if not updated:
        raise KeyError(f"配置文件 {path} 中未找到相机 ID: {cam_id}")

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"已更新 [{cam_id}] 外参至 {path} | 内点: {len(inlier_idx)}/{len(obj_pts)} | "
        f"RANSAC 误差: {err_before:.3f} px -> LM 精化后: {err_after:.3f} px"
    )


def _load_calib_input(input_path):
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    intr = payload["intrinsics"]
    dist_obj = payload.get("distortion", {})
    K = [
        [intr["fx"], 0.0, intr["cx"]],
        [0.0, intr["fy"], intr["cy"]],
        [0.0, 0.0, 1.0],
    ]
    dist = [
        dist_obj.get("k1", 0.0),
        dist_obj.get("k2", 0.0),
        dist_obj.get("p1", 0.0),
        dist_obj.get("p2", 0.0),
    ]
    image_size = tuple(payload.get("imageSize", [intr["width"], intr["height"]]))
    return payload.get("camId"), image_size, payload["pts_3d"], payload["pts_2d"], K, dist


if __name__ == "__main__":
    default_config = Path(__file__).resolve().parent.parent / "public" / "config" / "park_cameras.json"

    parser = argparse.ArgumentParser(description="数字孪生视频融合 Python OpenCV 标定工具箱")
    parser.add_argument("cmd", choices=["intrinsics", "verify", "pick2d", "pnp"])
    parser.add_argument("--video", default="chessboard.mp4")
    parser.add_argument("--source", default="cam01.mp4")
    parser.add_argument("--input", help="Web 标定台导出的 calib_points_*.json 路径")
    parser.add_argument("--cam-id", default=None, help="目标相机 ID（覆盖 JSON 内置 camId）")
    parser.add_argument("--config", default=str(default_config), help="park_cameras.json 路径")
    args = parser.parse_args()

    K_demo = [[1300.0, 0.0, 960.0], [0.0, 1300.0, 540.0], [0.0, 0.0, 1.0]]
    dist_demo = [-0.28, 0.09, 0.0, 0.0]

    if args.cmd == "intrinsics":
        run_intrinsics(args.video)
    elif args.cmd == "verify":
        run_verify(args.video, K_demo, dist_demo)
    elif args.cmd == "pick2d":
        run_pick_2d(args.source)
    elif args.cmd == "pnp":
        if args.input:
            in_cam_id, image_size, pts_3d, pts_2d, K, dist = _load_calib_input(args.input)
            target_cam = args.cam_id or in_cam_id or "cam_fixed_01"
            run_pnp_and_export(target_cam, args.config, image_size, pts_3d, pts_2d, K, dist)
        else:
            # 默认无外部输入时使用与 K_demo / dist_demo 严格自洽的立体示例控制点自检
            pts_3d_demo = [
                [-9.0, 8.0, -4.0],
                [9.0, 8.0, -4.0],
                [-9.0, 0.0, -4.0],
                [9.0, 0.0, -4.0],
                [-3.0, 5.5, 1.0],
                [3.5, 2.4, 2.0],
                [0.0, 0.0, 0.0],
            ]
            cam_pos = np.array([-10.0, 8.5, 11.0])
            target_pt = np.array([-1.0, 2.5, -6.0])
            fwd = target_pt - cam_pos
            fwd /= np.linalg.norm(fwd)
            right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
            right /= np.linalg.norm(right)
            down = np.cross(fwd, right)
            R_cv = np.vstack([right, down, fwd])
            t_cv = -R_cv @ cam_pos
            rvec_demo, _ = cv2.Rodrigues(R_cv)
            proj_pts, _ = cv2.projectPoints(
                np.array(pts_3d_demo, dtype=np.float64),
                rvec_demo,
                t_cv,
                np.array(K_demo, dtype=np.float64),
                np.array(dist_demo, dtype=np.float64),
            )
            pts_2d_demo = proj_pts.reshape(-1, 2).tolist()
            run_pnp_and_export(args.cam_id or "cam_fixed_01", args.config, (1920, 1080), pts_3d_demo, pts_2d_demo, K_demo, dist_demo)
