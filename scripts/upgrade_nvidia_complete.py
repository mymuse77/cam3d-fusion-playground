#!/usr/bin/env python3
"""
NVIDIA Warehouse_016 场景贴图与人物完整性全量升级脚本：
1. 利用 ground_truth.json 逐帧剔除所有人物/机器人/AGV 2D 框后再做时域聚合 + 空洞修复，
   彻底消除静止机器人 (#627/#354) 残留在 *_bg.png 中导致的“半透明镂空”与“地上躺尸灰影”；
2. 补齐其余 6 路相机 (Camera_04/07/08/09/10/11) 的干净静态帧与静态深度图，实现 12 路全仓无死角覆盖；
3. 在深度融合阶段同步用 ground_truth 掩膜剔除静止人员/机器人点云，杜绝地面鼓包；
4. 为地板与 3D 仓库网格烘焙 12 路多视角遮挡感知高清色彩贴图 (2048x2048)，彻底消除未激活视锥区的灰白无贴图死角；
5. 调用 Blender 5.3 导出带完整 UV 贴图的 warehouse_016.glb。
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = ROOT / ".tmp_nvidia_w16"
DEPTH_CACHE = TMP_DIR / "depth_static"
EXTRA_BG_DIR = TMP_DIR / "extra_bg"
OUT_DIR = ROOT / "public" / "demo-nvidia-w16"
BLENDER_EXE = Path(r"D:\Env\Blender 5.3\blender.exe")

sys.path.insert(0, str(ROOT / "scripts"))
from reconstruct_nvidia_mesh import HTTPRangeFile, HF_BASE

LIVE_CAMS = ["Camera", "Camera_01", "Camera_02", "Camera_03", "Camera_05", "Camera_06"]
EXTRA_CAMS = ["Camera_04", "Camera_07", "Camera_08", "Camera_09", "Camera_10", "Camera_11"]
ALL_CAMS = LIVE_CAMS + EXTRA_CAMS

START_FRAME = 148
CENTER_OFFSET_W = np.array([0.0, -10.0, 0.0], dtype=np.float64)
T_Z2Y = np.array([
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)


def make_gt_mask(gt_objs: list, cam_id: str, w: int, h: int, pad_px: int = 16) -> np.ndarray:
    """生成该帧该相机下所有动态目标的二值遮罩 (True = 被人物/机器人/AGV 占据)"""
    mask = np.zeros((h, w), dtype=bool)
    sx, sy = w / 1920.0, h / 1080.0
    for o in gt_objs:
        b = o.get("2d bounding box visible", {}).get(cam_id)
        if not b or len(b) != 4:
            continue
        x1 = max(0, int(b[0] * sx) - pad_px)
        y1 = max(0, int(b[1] * sy) - pad_px)
        x2 = min(w, int(b[2] * sx) + pad_px)
        y2 = min(h, int(b[3] * sy) + pad_px + int(10 * sy))  # 底部多留一点剔除脚底投影暗影
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def build_clean_bg_for_live_cam(cam_id: str, gt: dict):
    """用 ground_truth 掩膜剔除所有动态目标后计算干净静态背景，对全程静止目标区域做平滑纹理修复"""
    mp4_path = OUT_DIR / f"{cam_id.lower()}.mp4"
    out_bg = OUT_DIR / f"{cam_id.lower()}_bg.png"

    cap = cv2.VideoCapture(str(mp4_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sample_indices = np.linspace(0, max(0, total - 1), 45, dtype=int)
    masked_frames = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gt_fid = str(START_FRAME + int(idx))
        fg_mask = make_gt_mask(gt.get(gt_fid, []), cam_id, w, h, pad_px=14)
        f_float = frame.astype(np.float32)
        f_float[fg_mask] = np.nan
        masked_frames.append(f_float)
    cap.release()

    stack = np.stack(masked_frames, axis=0)
    with np.errstate(all="ignore"):
        clean = np.nanmedian(stack, axis=0)

    # 若某像素在 20 秒内始终被静止机器人占据 (全 NaN)，使用 OpenCV Telea 修复为周围地面/货架纹理
    hole_mask = np.isnan(clean[:, :, 0]).astype(np.uint8) * 255
    if np.any(hole_mask):
        fallback = np.nan_to_num(clean, nan=120.0).astype(np.uint8)
        hole_dilated = cv2.dilate(hole_mask, np.ones((5, 5), np.uint8), iterations=1)
        clean_u8 = cv2.inpaint(fallback, hole_dilated, inpaintRadius=9, flags=cv2.INPAINT_TELEA)
    else:
        clean_u8 = clean.astype(np.uint8)

    cv2.imwrite(str(out_bg), clean_u8)
    hole_pct = (hole_mask > 0).mean() * 100.0
    print(f"[BG] 已生成无人物干净底图 {out_bg.name} (修补全程静止盲区 {hole_pct:.2f}%)", flush=True)


def fetch_extra_cam_bg_and_depth(cam_id: str, gt: dict):
    """为其余 6 路辅助相机拉取 3 个时间点单帧图片与深度（补全全仓 3D 几何与烘焙贴图）"""
    EXTRA_BG_DIR.mkdir(parents=True, exist_ok=True)
    DEPTH_CACHE.mkdir(parents=True, exist_ok=True)

    bg_path = EXTRA_BG_DIR / f"{cam_id.lower()}_bg.png"
    if not bg_path.exists():
        print(f"[EXTRA] HTTP Range 采样辅助相机画面 {cam_id} ...", flush=True)
        frames = []
        # 采样 0s, 5s, 10s 三帧并用 GT 掩膜合成干净背景
        for sec, fid in [(0, 0), (5, 150), (10, 300)]:
            tmp_jpg = EXTRA_BG_DIR / f"_tmp_{cam_id}_{fid}.png"
            if not tmp_jpg.exists():
                cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", f"00:00:{sec:02d}",
                    "-i", f"{HF_BASE}/videos/{cam_id}.mp4",
                    "-vf", "scale=1280:720",
                    "-frames:v", "1",
                    str(tmp_jpg),
                ]
                subprocess.run(cmd, check=True)
            img = cv2.imread(str(tmp_jpg))
            fg_mask = make_gt_mask(gt.get(str(fid), []), cam_id, 1280, 720, pad_px=14)
            f_fl = img.astype(np.float32)
            f_fl[fg_mask] = np.nan
            frames.append(f_fl)

        stack = np.stack(frames, axis=0)
        with np.errstate(all="ignore"):
            clean = np.nanmedian(stack, axis=0)
        hole_mask = np.isnan(clean[:, :, 0]).astype(np.uint8) * 255
        if np.any(hole_mask):
            fallback = np.nan_to_num(clean, nan=120.0).astype(np.uint8)
            hole_dilated = cv2.dilate(hole_mask, np.ones((5, 5), np.uint8), iterations=1)
            clean_u8 = cv2.inpaint(fallback, hole_dilated, inpaintRadius=9, flags=cv2.INPAINT_TELEA)
        else:
            clean_u8 = clean.astype(np.uint8)
        cv2.imwrite(str(bg_path), clean_u8)
        print(f"[EXTRA] 完成辅助底图 {bg_path.name}", flush=True)

    # 拉取并缓存 GT 掩膜过滤后的干净静态深度图
    clean_depth_path = DEPTH_CACHE / f"{cam_id}_clean_depth.npy"
    if not clean_depth_path.exists():
        raw_legacy = DEPTH_CACHE / f"{cam_id}_static_depth.npy"
        url = f"{HF_BASE}/depth_maps/{cam_id}.h5"
        print(f"[DEPTH] HTTP Range 采样 GT 掩膜干净深度 {cam_id}.h5 ...", flush=True)
        rf = HTTPRangeFile(url, block_size=524288)
        d_frames = []
        with h5py.File(rf, "r") as f:
            for fid in (0, 60, 120):
                key = f"distance_to_image_plane_{fid:05d}.png"
                if key in f:
                    arr = f[key][()].astype(np.float32) / 1000.0
                    fg_mask = make_gt_mask(gt.get(str(fid), []), cam_id, 1920, 1080, pad_px=18)
                    arr[fg_mask] = np.nan
                    arr[(arr < 0.3) | (arr > 35.0)] = np.nan
                    d_frames.append(arr)
        with np.errstate(all="ignore"):
            d_clean = np.nanmax(np.stack(d_frames, axis=0), axis=0).astype(np.float32)
        d_clean = np.nan_to_num(d_clean, nan=0.0)
        np.save(clean_depth_path, d_clean)
        print(f"[DEPTH] 完成 {cam_id} 干净深度 ({rf.bytes_fetched/1e6:.2f} MB)", flush=True)


def ensure_all_clean_depths(gt: dict):
    """确保全部 12 路相机均具备经过 GT 掩膜去人的干净静态深度图"""
    for cid in LIVE_CAMS:
        clean_depth_path = DEPTH_CACHE / f"{cid}_clean_depth.npy"
        if clean_depth_path.exists():
            continue
        url = f"{HF_BASE}/depth_maps/{cid}.h5"
        print(f"[DEPTH] 为主相机 {cid} 生成 GT 掩膜去人干净深度 ...", flush=True)
        rf = HTTPRangeFile(url, block_size=524288)
        d_frames = []
        with h5py.File(rf, "r") as f:
            for fid in (0, 60, 120):
                key = f"distance_to_image_plane_{fid:05d}.png"
                if key in f:
                    arr = f[key][()].astype(np.float32) / 1000.0
                    fg_mask = make_gt_mask(gt.get(str(fid), []), cid, 1920, 1080, pad_px=18)
                    arr[fg_mask] = np.nan
                    arr[(arr < 0.3) | (arr > 35.0)] = np.nan
                    d_frames.append(arr)
        with np.errstate(all="ignore"):
            d_clean = np.nanmax(np.stack(d_frames, axis=0), axis=0).astype(np.float32)
        d_clean = np.nan_to_num(d_clean, nan=0.0)
        np.save(clean_depth_path, d_clean)
        print(f"[DEPTH] 完成 {cid} 去人深度 ({rf.bytes_fetched/1e6:.2f} MB)", flush=True)


def fuse_12cam_and_bake_atlas(cal: dict, grid_res: float = 0.045, half_span: float = 9.82):
    """融合 12 路去人深度生成完整全仓高度场，并利用 12 路深度遮挡校验烘焙 2048x2048 俯视/立体共享贴图"""
    sensors = {s["id"]: s for s in cal["sensors"]}
    n = int(round((2.0 * half_span) / grid_res))
    h_max = np.full((n, n), -1.0, dtype=np.float32)
    counts = np.zeros((n, n), dtype=np.int32)
    stride = 2

    cam_data = []
    for cid in ALL_CAMS:
        depth = np.load(DEPTH_CACHE / f"{cid}_clean_depth.npy")
        bg_file = (OUT_DIR / f"{cid.lower()}_bg.png") if cid in LIVE_CAMS else (EXTRA_BG_DIR / f"{cid.lower()}_bg.png")
        bg_bgr = cv2.imread(str(bg_file))
        bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        s = sensors[cid]
        K = np.array(s["intrinsicMatrix"], dtype=np.float64)
        M = np.array(s["extrinsicMatrix"], dtype=np.float64)
        R_w2c, t_w2c = M[:, :3], M[:, 3]
        R_c2w = R_w2c.T
        cam_pos_w = -R_c2w @ t_w2c

        cam_data.append({
            "id": cid,
            "depth": depth,
            "bg_rgb": bg_rgb,
            "K": K,
            "M": M,
            "P": K @ M,
            "cam_pos_w": cam_pos_w,
        })

        H, W = depth.shape
        ys = np.arange(0, H, stride, dtype=np.float64)
        xs = np.arange(0, W, stride, dtype=np.float64)
        uu, vv = np.meshgrid(xs, ys)
        zz = depth[::stride, ::stride].astype(np.float64)
        mask = (zz > 0.5) & (zz < 28.0)
        u_v, v_v, z_v = uu[mask], vv[mask], zz[mask]

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        P_cam = np.stack([(u_v - cx) * z_v / fx, (v_v - cy) * z_v / fy, z_v], axis=0)
        P_w = R_c2w @ P_cam + cam_pos_w[:, None]
        P_t = T_Z2Y @ (P_w - CENTER_OFFSET_W[:, None])

        xt, yt, zt = P_t[0], P_t[1], P_t[2]
        in_box = (
            (xt >= -half_span) & (xt < half_span)
            & (zt >= -half_span) & (zt < half_span)
            & (yt >= -0.05) & (yt <= 3.55)
        )
        xt, yt, zt = xt[in_box], yt[in_box], zt[in_box]
        ix = np.clip(np.floor((xt + half_span) / grid_res).astype(np.int32), 0, n - 1)
        iz = np.clip(np.floor((zt + half_span) / grid_res).astype(np.int32), 0, n - 1)
        np.add.at(counts, (iz, ix), 1)
        np.maximum.at(h_max, (iz, ix), yt.astype(np.float32))

    hf = np.where(counts > 0, np.maximum(0.0, h_max), 0.0).astype(np.float32)
    hf_med = median_filter(hf, size=3)
    hf = np.where(np.abs(hf - hf_med) > 0.25, hf_med, hf)
    hf = median_filter(hf, size=3)
    hf[hf < 0.045] = 0.0

    # 烘焙 2048x2048 全仓 12 路真实色彩顶视/表面贴图（带真实深度遮挡校验！）
    tex_size = 2048
    xs_t = np.linspace(-half_span, half_span, tex_size, dtype=np.float32)
    zs_t = np.linspace(-half_span, half_span, tex_size, dtype=np.float32)
    gx_t, gz_t = np.meshgrid(xs_t, zs_t)
    # 采样每个像素对应的真实表面高度 Y_t
    gy_t = cv2.resize(hf, (tex_size, tex_size), interpolation=cv2.INTER_NEAREST)

    # 转回 NVIDIA Z-Up 世界坐标: X_w = X_t, Y_w = -Z_t - 10.0, Z_w = Y_t
    Xw = gx_t.reshape(-1)
    Yw = -gz_t.reshape(-1) - 10.0
    Zw = gy_t.reshape(-1)
    pts_w = np.stack([Xw, Yw, Zw, np.ones_like(Xw)], axis=0)  # (4, M)

    accum_rgb = np.zeros((Xw.size, 3), dtype=np.float32)
    accum_w = np.zeros(Xw.size, dtype=np.float32)

    print("[BAKE] 正在从 12 路相机烘焙 2048x2048 全仓无死角表面真实贴图 ...", flush=True)
    for cd in cam_data:
        P = cd["P"]
        M = cd["M"]
        depth = cd["depth"]
        bg_rgb = cd["bg_rgb"]  # (720, 1280, 3)

        # 相机系 Z 深度
        zc = (M[2, :3] @ pts_w[:3, :]) + M[2, 3]
        valid = zc > 0.8
        idx_v = np.where(valid)[0]
        if idx_v.size == 0:
            continue

        uvw = P @ pts_w[:, idx_v]
        u1920 = uvw[0] / uvw[2]
        v1080 = uvw[1] / uvw[2]
        in_img = (u1920 >= 8) & (u1920 < 1912) & (v1080 >= 8) & (v1080 < 1072)
        idx_v = idx_v[in_img]
        u1920 = u1920[in_img]
        v1080 = v1080[in_img]
        zc_v = zc[idx_v]

        # 真实深度图遮挡校验（消除背面拉伸投射！）
        du = np.clip(np.round(u1920).astype(np.int32), 0, 1919)
        dv = np.clip(np.round(v1080).astype(np.int32), 0, 1079)
        d_gt = depth[dv, du]
        vis = (d_gt <= 0.1) | (zc_v <= d_gt + 0.22)
        idx_v = idx_v[vis]
        u1920 = u1920[vis]
        v1080 = v1080[vis]
        zc_v = zc_v[vis]

        # 采样 1280x720 干净 RGB 底图
        bu = np.clip(np.round(u1920 * (1280.0 / 1920.0)).astype(np.int32), 0, 1279)
        bv = np.clip(np.round(v1080 * (720.0 / 1080.0)).astype(np.int32), 0, 719)
        cols = bg_rgb[bv, bu]

        # 距离越近、越靠画面中心权重越高
        edge_x = np.minimum(u1920, 1920.0 - u1920) / 192.0
        edge_y = np.minimum(v1080, 1080.0 - v1080) / 108.0
        edge_w = np.clip(np.minimum(edge_x, edge_y), 0.05, 1.0)
        w_cam = edge_w / np.maximum(zc_v, 1.5)

        accum_rgb[idx_v] += cols * w_cam[:, None]
        accum_w[idx_v] += w_cam

    # 未被任何相机直接拍到的极少数盲区用 map.png 对应位置真实俯视色彩兜底（100% 零灰白死角！）
    map_bgr = cv2.imread(str(TMP_DIR / "map.png"))
    map_rgb = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    s0 = cal["sensors"][0]
    sf = s0["scaleFactor"]
    tx = s0["translationToGlobalCoordinates"]["x"]
    ty = s0["translationToGlobalCoordinates"]["y"]
    m_px = np.clip(np.round((Xw + tx) * sf).astype(np.int32), 0, 1919)
    m_py = np.clip(np.round(1080.0 - (Yw + ty) * sf).astype(np.int32), 0, 1079)
    fallback_rgb = map_rgb[m_py, m_px]

    has_w = accum_w > 1e-4
    final_rgb = np.where(has_w[:, None], accum_rgb / np.maximum(accum_w[:, None], 1e-5), fallback_rgb)
    atlas_bgr = cv2.cvtColor(np.clip(final_rgb, 0, 255).astype(np.uint8).reshape(tex_size, tex_size, 3), cv2.COLOR_RGB2BGR)
    atlas_path = TMP_DIR / "warehouse_baked_atlas.jpg"
    cv2.imwrite(str(atlas_path), atlas_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(
        f"[BAKE] 贴图烘焙完成: 12 路实景覆盖率 {(has_w.mean()*100):.1f}% (+ map.png 无缝兜底 = 100%)",
        flush=True,
    )

    npz_path = TMP_DIR / "warehouse_12cam_hf.npz"
    np.savez_compressed(npz_path, hf=hf, grid_res=grid_res, half_span=half_span)
    return npz_path, atlas_path


def export_textured_glb(npz_path: Path, atlas_path: Path):
    """在 Blender 5.3 中构建带完整烘焙贴图 UV 的地面 + 12路真实深度 3D 网格 + 实景墙体并导出 GLB"""
    glb_path = OUT_DIR / "warehouse_016.glb"

    blender_code = f"""
import bpy
import numpy as np

bpy.ops.wm.read_factory_settings(use_empty=True)

data = np.load(r"{str(npz_path)}")
hf = data["hf"]
grid_res = float(data["grid_res"])
half_span = float(data["half_span"])
nz, nx = hf.shape

img = bpy.data.images.load(r"{str(atlas_path)}")

def make_atlas_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    tex_node = nt.nodes.new("ShaderNodeTexImage")
    tex_node.image = img
    nt.links.new(bsdf.inputs["Base Color"], tex_node.outputs["Color"])
    bsdf.inputs["Roughness"].default_value = 0.86
    bsdf.inputs["Metallic"].default_value = 0.04
    return m

def make_mat(name, rgba, rough=0.84):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = 0.04
    return m

def add_box(name, center_three, size_three, mat, walkable=False):
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

mat_atlas = make_atlas_mat("Warehouse_Baked_Mat")
mat_wall = make_mat("Warehouse_Wall_Mat", (0.76, 0.78, 0.80, 1.0), rough=0.85)
mat_base = make_mat("Warehouse_Baseboard_Mat", (0.24, 0.26, 0.29, 1.0), rough=0.75)
mat_dado = make_mat("Warehouse_Dado_Mat", (0.12, 0.34, 0.62, 1.0), rough=0.65)

# 1. 带完整 12 路烘焙纹理的可行走地面网格 (Y=0.0, isWalkable=True)
bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=2.0 * half_span, location=(0.0, 0.0, 0.0))
floor_obj = bpy.context.active_object
floor_obj.name = "Warehouse_Floor"
floor_obj["isWalkable"] = True
floor_obj.data.materials.append(mat_atlas)
# 校正平面 UV: Three X in [-half_span, half_span] -> u in [0,1]; Three Z in [-half_span, half_span] (Blender Y=-Z) -> v_img = 1 - (Z + half_span)/(2*half_span)
uv_layer = floor_obj.data.uv_layers.active.data
for poly in floor_obj.data.polygons:
    for li in poly.loop_indices:
        vi = floor_obj.data.loops[li].vertex_index
        co = floor_obj.data.vertices[vi].co
        xt = co.x
        zt = -co.y
        u = (xt + half_span) / (2.0 * half_span)
        v = 1.0 - (zt + half_span) / (2.0 * half_span)
        uv_layer[li].uv = (u, v)

# 2. 12 路去人深度重建的完整全仓静态物体 3D 网格 + 烘焙色彩贴图
mesh = bpy.data.meshes.new("Warehouse_DepthMesh")
obj = bpy.data.objects.new("Warehouse_DepthMesh", mesh)
bpy.context.collection.objects.link(obj)
obj.data.materials.append(mat_atlas)

xs = -half_span + (np.arange(nx, dtype=np.float32) + 0.5) * grid_res
zs = -half_span + (np.arange(nz, dtype=np.float32) + 0.5) * grid_res
gx, gz = np.meshgrid(xs, zs)

elevated = hf > 0.04
pad = np.pad(elevated, 1, mode='constant', constant_values=False)
keep_vert = (
    pad[0:-2, 0:-2] | pad[0:-2, 1:-1] | pad[0:-2, 2:] |
    pad[1:-1, 0:-2] | pad[1:-1, 1:-1] | pad[1:-1, 2:] |
    pad[2:,   0:-2] | pad[2:,   1:-1] | pad[2:,   2:]
)

vert_idx = np.full((nz, nx), -1, dtype=np.int32)
ky, kx = np.where(keep_vert)
vert_idx[ky, kx] = np.arange(len(ky), dtype=np.int32)

verts = np.stack([gx[ky, kx], -gz[ky, kx], hf[ky, kx]], axis=1).tolist()

v00 = vert_idx[:-1, :-1]
v10 = vert_idx[:-1, 1:]
v11 = vert_idx[1:, 1:]
v01 = vert_idx[1:, :-1]
has_elev = elevated[:-1, :-1] | elevated[:-1, 1:] | elevated[1:, 1:] | elevated[1:, :-1]
valid_q = has_elev & (v00 >= 0) & (v10 >= 0) & (v11 >= 0) & (v01 >= 0)
qy, qx = np.where(valid_q)
faces = np.stack([v00[qy, qx], v01[qy, qx], v11[qy, qx], v10[qy, qx]], axis=1).tolist()

mesh.from_pydata(verts, [], faces)
mesh.update()

# 在减面前先写入精确世界平面投影 UV，Decimate 会自动保留 UV！
uv_lay = mesh.uv_layers.new(name="UVMap")
for poly in mesh.polygons:
    for li in poly.loop_indices:
        vi = mesh.loops[li].vertex_index
        co = mesh.vertices[vi].co
        u = (co.x + half_span) / (2.0 * half_span)
        v = 1.0 - (-co.y + half_span) / (2.0 * half_span)
        uv_lay.data[li].uv = (u, v)

bpy.context.view_layer.objects.active = obj
mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
mod.decimate_type = 'COLLAPSE'
mod.ratio = 0.35
bpy.ops.object.modifier_apply(modifier="Decimate")

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

print(f"BAKED_MESH_VERTS={{len(obj.data.vertices)}} POLYS={{len(obj.data.polygons)}}")

# 3. 精确周界墙体（还原深灰护墙脚 0~1.20m + 蓝色腰线 1.20~1.62m + 浅白上墙）
wall_h = 6.6
wall_t = 0.45
inner = 9.82
wall_c = inner + wall_t * 0.5
span = inner * 2.0 + wall_t * 2.0

add_box("Wall_North", (0.0, wall_h * 0.5, -wall_c), (span, wall_h, wall_t), mat_wall)
add_box("Wall_South", (0.0, wall_h * 0.5,  wall_c), (span, wall_h, wall_t), mat_wall)
add_box("Wall_West",  (-wall_c, wall_h * 0.5, 0.0), (wall_t, wall_h, span), mat_wall)
add_box("Wall_East",  ( wall_c, wall_h * 0.5, 0.0), (wall_t, wall_h, span), mat_wall)

# 深灰护墙板 (0~1.20m)
add_box("Base_North", (0.0, 0.60, -inner + 0.012), (inner * 2.0, 1.20, 0.024), mat_base)
add_box("Base_South", (0.0, 0.60,  inner - 0.012), (inner * 2.0, 1.20, 0.024), mat_base)
add_box("Base_West",  (-inner + 0.012, 0.60, 0.0), (0.024, 1.20, inner * 2.0), mat_base)
add_box("Base_East",  ( inner - 0.012, 0.60, 0.0), (0.024, 1.20, inner * 2.0), mat_base)

# 蓝色墙腰带 (1.20~1.62m)
add_box("Dado_North", (0.0, 1.41, -inner + 0.015), (inner * 2.0, 0.42, 0.03), mat_dado)
add_box("Dado_South", (0.0, 1.41,  inner - 0.015), (inner * 2.0, 0.42, 0.03), mat_dado)
add_box("Dado_West",  (-inner + 0.015, 1.41, 0.0), (0.03, 0.42, inner * 2.0), mat_dado)
add_box("Dado_East",  ( inner - 0.015, 1.41, 0.0), (0.03, 0.42, inner * 2.0), mat_dado)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.export_scene.gltf(
    filepath=r"{str(glb_path)}",
    export_format='GLB',
    use_selection=True,
    export_yup=True,
    export_apply=True,
    export_extras=True,
)
print("EXPORT_BAKED_GLB_OK")
"""
    script_path = TMP_DIR / "build_baked_glb.py"
    script_path.write_text(blender_code, encoding="utf-8")

    print("[BLENDER] 正在导出带 2048x2048 实景烘焙贴图的 warehouse_016.glb ...", flush=True)
    res = subprocess.run(
        [str(BLENDER_EXE), "--background", "--factory-startup", "--python", str(script_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in res.stdout.splitlines():
        if "BAKED_MESH_VERTS" in line or "EXPORT_BAKED_GLB_OK" in line:
            print(" ", line, flush=True)
    if "EXPORT_BAKED_GLB_OK" not in res.stdout:
        print(res.stdout[-600:])
        print(res.stderr[-600:])
        raise RuntimeError("Blender 导出失败")
    print(f"[BLENDER] 成功导出 {glb_path.name} ({glb_path.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)


def update_camera_near_far():
    """优化 park_cameras_w16.json 的 near/far (1.5m~34m)，大幅提升深度缓冲精度"""
    cfg_path = OUT_DIR / "park_cameras_w16.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for c in cfg.get("cameras", []):
        intr = c.get("intrinsics", {})
        intr["near"] = 1.5
        intr["far"] = 34.0
        intr["depthFovScale"] = 1.12
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[CFG] 已更新 park_cameras_w16.json (near=1.5m, far=34.0m)", flush=True)


def main():
    gt = json.loads((TMP_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    cal = json.loads((TMP_DIR / "calibration.json").read_text(encoding="utf-8"))

    # 1. 为 6 路实时相机重新生成 100% 剔除人物/静止机器人的干净 *_bg.png
    for cid in LIVE_CAMS:
        build_clean_bg_for_live_cam(cid, gt)

    # 2. 确保主相机干净去人深度图就绪，并拉取其余 6 路辅助相机的干净画面与深度
    ensure_all_clean_depths(gt)
    for cid in EXTRA_CAMS:
        fetch_extra_cam_bg_and_depth(cid, gt)

    # 3. 12 路深度融合 + 烘焙 2048x2048 实景贴图
    npz_path, atlas_path = fuse_12cam_and_bake_atlas(cal, grid_res=0.045, half_span=9.82)

    # 4. 导出带完整 UV 贴图的 warehouse_016.glb 并优化相机深度范围
    export_textured_glb(npz_path, atlas_path)
    update_camera_near_far()


if __name__ == "__main__":
    main()
