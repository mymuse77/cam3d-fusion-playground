#!/usr/bin/env python3
"""
NVIDIA Warehouse_016 终极画质提升脚本：
1. 【消除场景立面拉条纹】Blender Smart UV Project 三向展开 + 12路法线加权 (N·V)^2 遮挡感知纹理烘焙 -> 重建 warehouse_016.glb；
2. 【消除人物撞色残缺】使用 Ultralytics YOLO11n-seg 实例分割 + GT 框内形态学轮廓实心闭合，
   离线生成 6 路合一 (2x3 拼合 1280x1080) 绿幕级 Alpha 掩膜流 public/demo-nvidia-w16/masks_w16.mp4，
   并写入 park_cameras_w16.json。
"""

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = ROOT / ".tmp_nvidia_w16"
DEPTH_CACHE = TMP_DIR / "depth_static"
EXTRA_BG_DIR = TMP_DIR / "extra_bg"
OUT_DIR = ROOT / "public" / "demo-nvidia-w16"
BLENDER_EXE = Path(r"D:\Env\Blender 5.3\blender.exe")

LIVE_CAMS = ["Camera", "Camera_01", "Camera_02", "Camera_03", "Camera_05", "Camera_06"]
EXTRA_CAMS = ["Camera_04", "Camera_07", "Camera_08", "Camera_09", "Camera_10", "Camera_11"]
ALL_CAMS = LIVE_CAMS + EXTRA_CAMS
START_FRAME = 148


# =========================================================
# Part 1: Blender Smart UV 展开 + 12路法线感知多视角烘焙
# =========================================================
def build_smart_uv_warehouse_glb(cal: dict):
    npz_hf = TMP_DIR / "warehouse_12cam_hf.npz"
    unwrapped_npz = TMP_DIR / "warehouse_smart_unwrapped.npz"
    smart_atlas_jpg = TMP_DIR / "warehouse_smart_atlas.jpg"
    floor_atlas_jpg = TMP_DIR / "warehouse_baked_atlas.jpg"
    glb_path = OUT_DIR / "warehouse_016.glb"

    # Step 1A: 在 Blender 中构建减面三角网格并执行 Smart UV Project，使所有竖直立面拥有独立不拉伸的 UV 岛
    step1_py = TMP_DIR / "blender_step1_smart_uv.py"
    step1_py.write_text(
        f"""
import bpy, math
import numpy as np

bpy.ops.wm.read_factory_settings(use_empty=True)
data = np.load(r"{str(npz_hf)}")
hf = data["hf"]
grid_res = float(data["grid_res"])
half_span = float(data["half_span"])
nz, nx = hf.shape

mesh = bpy.data.meshes.new("Warehouse_DepthMesh")
obj = bpy.data.objects.new("Warehouse_DepthMesh", mesh)
bpy.context.collection.objects.link(obj)

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

bpy.context.view_layer.objects.active = obj
obj.select_set(True)

mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
mod.decimate_type = 'COLLAPSE'
mod.ratio = 0.34
bpy.ops.object.modifier_apply(modifier="Decimate")

# 全部三角化并执行 Smart UV Project，保证每个竖直面与顶面都有独立等比 UV
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.quads_convert_to_tris()
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.0025, scale_to_bounds=True)
bpy.ops.object.mode_set(mode='OBJECT')

m = obj.data
n_verts = len(m.vertices)
n_polys = len(m.polygons)

v_co = np.empty(n_verts * 3, dtype=np.float32)
m.vertices.foreach_get("co", v_co)
v_co = v_co.reshape(n_verts, 3)

f_vi = np.empty(n_polys * 3, dtype=np.int32)
m.polygons.foreach_get("vertices", f_vi)
f_vi = f_vi.reshape(n_polys, 3)

f_no = np.empty(n_polys * 3, dtype=np.float32)
m.polygons.foreach_get("normal", f_no)
f_no = f_no.reshape(n_polys, 3)

uv_data = np.empty(n_polys * 3 * 2, dtype=np.float32)
m.uv_layers.active.data.foreach_get("uv", uv_data)
f_uv = uv_data.reshape(n_polys, 3, 2)

np.savez_compressed(r"{str(unwrapped_npz)}", v_co=v_co, f_vi=f_vi, f_no=f_no, f_uv=f_uv, half_span=half_span)
print(f"SMART_UV_OK VERTS={{n_verts}} TRIS={{n_polys}}")
""",
        encoding="utf-8",
    )

    print("[SMART-UV] 正在调用 Blender 5.3 执行 Smart UV 三向无拉伸展开 ...", flush=True)
    res = subprocess.run(
        [str(BLENDER_EXE), "--background", "--factory-startup", "--python", str(step1_py)],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in res.stdout.splitlines():
        if "SMART_UV_OK" in line:
            print(" ", line, flush=True)

    # Step 1B: 在 Python 中光栅化 UV 三角形并按法线 (N·V)^2.2 从 12 路相机烘焙高清纹理
    u_data = np.load(unwrapped_npz)
    v_co = u_data["v_co"]   # Blender (x, y_bl=-z_t, z_bl=y_t)
    f_vi = u_data["f_vi"]   # (T, 3)
    f_no = u_data["f_no"]   # Blender normal (nx, ny_bl, nz_bl)
    f_uv = u_data["f_uv"]   # (T, 3, 2) in [0, 1]
    half_span = float(u_data["half_span"])

    # 转回 NVIDIA Z-Up 世界坐标:
    # Blender co = (X_t, -Z_t, Y_t), 而 X_w = X_t, Y_w = -Z_t - 10 = co.y - 10, Z_w = Y_t = co.z!
    # 法线同理: N_w = (f_no.x, f_no.y, f_no.z) 直接就是 NVIDIA Z-Up 法线！(因为 Y_w = co.y - 10, Z_w = co.z, X_w = co.x!)
    v_world = v_co.copy()
    v_world[:, 1] -= 10.0  # (N, 3) exact NVIDIA Z-Up world coords!
    n_world = f_no.copy()  # (T, 3) exact NVIDIA Z-Up unit normals!

    tex_res = 2048
    pos_map = np.zeros((tex_res, tex_res, 3), dtype=np.float32)
    nrm_map = np.zeros((tex_res, tex_res, 3), dtype=np.float32)
    valid_map = np.zeros((tex_res, tex_res), dtype=bool)

    print(f"[SMART-UV] 正在光栅化 {len(f_vi)} 个 UV 三角面到 {tex_res}x{tex_res} 法线感知纹素网格 ...", flush=True)
    t0 = time.time()
    # 注意图像行 py 对应 v_img = 1.0 - uv.y (Blender glTF 导出图像纹理时 V 自底向上)
    uv_px = np.empty_like(f_uv)
    uv_px[:, :, 0] = f_uv[:, :, 0] * (tex_res - 1)
    uv_px[:, :, 1] = (1.0 - f_uv[:, :, 1]) * (tex_res - 1)

    tri_world = v_world[f_vi]  # (T, 3, 3)

    for t_idx in range(len(f_vi)):
        pts = uv_px[t_idx]
        x0 = max(0, int(np.floor(pts[:, 0].min())))
        x1 = min(tex_res - 1, int(np.ceil(pts[:, 0].max())))
        y0 = max(0, int(np.floor(pts[:, 1].min())))
        y1 = min(tex_res - 1, int(np.ceil(pts[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue

        ax, ay = pts[0, 0], pts[0, 1]
        bx, by = pts[1, 0], pts[1, 1]
        cx, cy = pts[2, 0], pts[2, 1]
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denom) < 1e-6:
            continue
        inv_d = 1.0 / denom

        xs = np.arange(x0, x1 + 1, dtype=np.float32)
        ys = np.arange(y0, y1 + 1, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)

        w0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) * inv_d
        w1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) * inv_d
        w2 = 1.0 - w0 - w1

        # 轻微外扩 -0.04 覆盖 UV 岛像素边缘，消除接缝黑线
        inside = (w0 >= -0.04) & (w1 >= -0.04) & (w2 >= -0.04)
        if not np.any(inside):
            continue

        w0c = np.clip(w0[inside], 0.0, 1.0)
        w1c = np.clip(w1[inside], 0.0, 1.0)
        w2c = np.clip(w2[inside], 0.0, 1.0)
        ws = np.maximum(w0c + w1c + w2c, 1e-6)
        w0c /= ws
        w1c /= ws
        w2c /= ws

        tw = tri_world[t_idx]
        pw = w0c[:, None] * tw[0] + w1c[:, None] * tw[1] + w2c[:, None] * tw[2]

        sub_pos = pos_map[y0 : y1 + 1, x0 : x1 + 1]
        sub_nrm = nrm_map[y0 : y1 + 1, x0 : x1 + 1]
        sub_val = valid_map[y0 : y1 + 1, x0 : x1 + 1]

        sub_pos[inside] = pw
        sub_nrm[inside] = n_world[t_idx]
        sub_val[inside] = True

    print(f"[SMART-UV] 光栅化完成 ({time.time() - t0:.1f}s)，有效纹素占比 {valid_map.mean()*100:.1f}%", flush=True)

    vy, vx = np.where(valid_map)
    pts_w3 = pos_map[vy, vx]   # (M, 3)
    nrms_w = nrm_map[vy, vx]   # (M, 3)
    pts_w4 = np.concatenate([pts_w3, np.ones((len(pts_w3), 1), dtype=np.float32)], axis=1).T  # (4, M)

    accum_rgb = np.zeros((len(pts_w3), 3), dtype=np.float32)
    accum_w = np.zeros(len(pts_w3), dtype=np.float32)

    sensors = {s["id"]: s for s in cal["sensors"]}
    for cid in ALL_CAMS:
        depth = np.load(DEPTH_CACHE / f"{cid}_clean_depth.npy")
        bg_file = (OUT_DIR / f"{cid.lower()}_bg.png") if cid in LIVE_CAMS else (EXTRA_BG_DIR / f"{cid.lower()}_bg.png")
        bg_rgb = cv2.cvtColor(cv2.imread(str(bg_file)), cv2.COLOR_BGR2RGB).astype(np.float32)

        s = sensors[cid]
        K = np.array(s["intrinsicMatrix"], dtype=np.float64)
        M = np.array(s["extrinsicMatrix"], dtype=np.float64)
        P = K @ M
        R_w2c, t_w2c = M[:, :3], M[:, 3]
        cam_pos_w = (-R_w2c.T @ t_w2c).astype(np.float32)

        # 法线朝向校验: dot(N_w, normalize(cam_pos_w - P_w)) > 0.08！
        vec_to_cam = cam_pos_w[None, :] - pts_w3
        dist = np.linalg.norm(vec_to_cam, axis=1)
        dir_to_cam = vec_to_cam / np.maximum(dist[:, None], 1e-4)
        ndotv = np.sum(nrms_w * dir_to_cam, axis=1)

        zc = (M[2, :3] @ pts_w3.T) + M[2, 3]
        cand = (zc > 0.8) & (ndotv > 0.08)
        idx_v = np.where(cand)[0]
        if idx_v.size == 0:
            continue

        uvw = P @ pts_w4[:, idx_v]
        u1920 = uvw[0] / uvw[2]
        v1080 = uvw[1] / uvw[2]
        in_img = (u1920 >= 8) & (u1920 < 1912) & (v1080 >= 8) & (v1080 < 1072)
        idx_v = idx_v[in_img]
        u1920 = u1920[in_img]
        v1080 = v1080[in_img]
        zc_v = zc[idx_v]
        ndotv_v = ndotv[idx_v]

        du = np.clip(np.round(u1920).astype(np.int32), 0, 1919)
        dv = np.clip(np.round(v1080).astype(np.int32), 0, 1079)
        d_gt = depth[dv, du]
        vis = (d_gt <= 0.1) | (zc_v <= d_gt + 0.25)
        idx_v = idx_v[vis]
        u1920 = u1920[vis]
        v1080 = v1080[vis]
        zc_v = zc_v[vis]
        ndotv_v = ndotv_v[vis]

        bu = np.clip(np.round(u1920 * (1280.0 / 1920.0)).astype(np.int32), 0, 1279)
        bv = np.clip(np.round(v1080 * (720.0 / 1080.0)).astype(np.int32), 0, 719)
        cols = bg_rgb[bv, bu]

        edge_x = np.minimum(u1920, 1920.0 - u1920) / 192.0
        edge_y = np.minimum(v1080, 1080.0 - v1080) / 108.0
        edge_w = np.clip(np.minimum(edge_x, edge_y), 0.05, 1.0)
        # 核心：按 (N·V)^2.2 强优先选用正对竖直立面的相机，彻底消灭掠射拉条纹！
        w_cam = (ndotv_v ** 2.2) * edge_w / np.maximum(zc_v, 1.5)

        accum_rgb[idx_v] += cols * w_cam[:, None]
        accum_w[idx_v] += w_cam

    # 极少数靠墙夹缝未被正拍到的纹素用 map.png 颜色温和兜底
    map_rgb = cv2.cvtColor(cv2.imread(str(TMP_DIR / "map.png")), cv2.COLOR_BGR2RGB).astype(np.float32)
    s0 = cal["sensors"][0]
    sf = s0["scaleFactor"]
    tx = s0["translationToGlobalCoordinates"]["x"]
    ty = s0["translationToGlobalCoordinates"]["y"]
    m_px = np.clip(np.round((pts_w3[:, 0] + tx) * sf).astype(np.int32), 0, 1919)
    m_py = np.clip(np.round(1080.0 - (pts_w3[:, 1] + ty) * sf).astype(np.int32), 0, 1079)
    fb_rgb = map_rgb[m_py, m_px]

    has_w = accum_w > 1e-5
    texel_rgb = np.where(has_w[:, None], accum_rgb / np.maximum(accum_w[:, None], 1e-6), fb_rgb)

    atlas_img = np.full((tex_res, tex_res, 3), 118, dtype=np.uint8)
    atlas_img[vy, vx] = np.clip(texel_rgb, 0, 255).astype(np.uint8)
    # UV 岛外扩 3px 防止双线性采样渗色 (bleeding)
    inpaint_mask = (~valid_map).astype(np.uint8) * 255
    border_band = cv2.dilate(valid_map.astype(np.uint8) * 255, np.ones((7, 7), np.uint8)) - (valid_map.astype(np.uint8) * 255)
    atlas_img = cv2.inpaint(atlas_img, border_band, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    cv2.imwrite(str(smart_atlas_jpg), cv2.cvtColor(atlas_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 93])
    print(f"[SMART-UV] 12 路法线感知贴图烘焙完成: 正向覆盖率 {has_w.mean()*100:.1f}%", flush=True)

    # Step 1C: Blender 组装双贴图 (地面 + Smart-UV 立面网格 + 周界墙体) 并导出 warehouse_016.glb
    step2_py = TMP_DIR / "blender_step2_export_smart.py"
    step2_py.write_text(
        f"""
import bpy
import numpy as np

bpy.ops.wm.read_factory_settings(use_empty=True)
u_data = np.load(r"{str(unwrapped_npz)}")
v_co = u_data["v_co"]
f_vi = u_data["f_vi"]
f_uv = u_data["f_uv"]
half_span = float(u_data["half_span"])

img_floor = bpy.data.images.load(r"{str(floor_atlas_jpg)}")
img_smart = bpy.data.images.load(r"{str(smart_atlas_jpg)}")

def make_tex_mat(name, image):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    tex_node = nt.nodes.new("ShaderNodeTexImage")
    tex_node.image = image
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

mat_floor = make_tex_mat("Warehouse_Floor_Mat", img_floor)
mat_smart = make_tex_mat("Warehouse_Smart_Mat", img_smart)
mat_wall = make_mat("Warehouse_Wall_Mat", (0.76, 0.78, 0.80, 1.0), rough=0.85)
mat_base = make_mat("Warehouse_Baseboard_Mat", (0.24, 0.26, 0.29, 1.0), rough=0.75)
mat_dado = make_mat("Warehouse_Dado_Mat", (0.12, 0.34, 0.62, 1.0), rough=0.65)

# 1. 地面
bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=2.0 * half_span, location=(0.0, 0.0, 0.0))
floor_obj = bpy.context.active_object
floor_obj.name = "Warehouse_Floor"
floor_obj["isWalkable"] = True
floor_obj.data.materials.append(mat_floor)
uv_layer = floor_obj.data.uv_layers.active.data
for poly in floor_obj.data.polygons:
    for li in poly.loop_indices:
        co = floor_obj.data.vertices[floor_obj.data.loops[li].vertex_index].co
        uv_layer[li].uv = ((co.x + half_span) / (2.0 * half_span), 1.0 - (-co.y + half_span) / (2.0 * half_span))

# 2. Smart-UV 立面无拉伸 3D 仓库网格
mesh = bpy.data.meshes.new("Warehouse_DepthMesh")
mesh.from_pydata(v_co.tolist(), [], f_vi.tolist())
mesh.update()
uv_lay = mesh.uv_layers.new(name="UVMap")
uv_lay.data.foreach_set("uv", f_uv.reshape(-1))
obj = bpy.data.objects.new("Warehouse_DepthMesh", mesh)
bpy.context.collection.objects.link(obj)
obj.data.materials.append(mat_smart)

# 3. 周界墙体
wall_h, wall_t, inner = 6.6, 0.45, 9.82
wall_c = inner + wall_t * 0.5
span = inner * 2.0 + wall_t * 2.0
add_box("Wall_North", (0.0, wall_h * 0.5, -wall_c), (span, wall_h, wall_t), mat_wall)
add_box("Wall_South", (0.0, wall_h * 0.5,  wall_c), (span, wall_h, wall_t), mat_wall)
add_box("Wall_West",  (-wall_c, wall_h * 0.5, 0.0), (wall_t, wall_h, span), mat_wall)
add_box("Wall_East",  ( wall_c, wall_h * 0.5, 0.0), (wall_t, wall_h, span), mat_wall)

add_box("Base_North", (0.0, 0.60, -inner + 0.012), (inner * 2.0, 1.20, 0.024), mat_base)
add_box("Base_South", (0.0, 0.60,  inner - 0.012), (inner * 2.0, 1.20, 0.024), mat_base)
add_box("Base_West",  (-inner + 0.012, 0.60, 0.0), (0.024, 1.20, inner * 2.0), mat_base)
add_box("Base_East",  ( inner - 0.012, 0.60, 0.0), (0.024, 1.20, inner * 2.0), mat_base)

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
print("EXPORT_SMART_GLB_OK")
""",
        encoding="utf-8",
    )

    res2 = subprocess.run(
        [str(BLENDER_EXE), "--background", "--factory-startup", "--python", str(step2_py)],
        capture_output=True,
        text=True,
        check=True,
    )
    if "EXPORT_SMART_GLB_OK" not in res2.stdout:
        print(res2.stdout[-500:])
        print(res2.stderr[-500:])
        raise RuntimeError("Smart-UV GLB 导出失败")
    print(f"[SMART-UV] 成功导出无拉条纹 {glb_path.name} ({glb_path.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)


# =========================================================
# Part 2: YOLOv11-seg + GT 轮廓实心闭合 -> 2x3 AI 掩膜流
# =========================================================
def build_ai_mask_atlas_video(gt: dict):
    torch.set_num_threads(8)
    print("[AI-SEG] 正在加载 YOLO11n-seg 实例分割模型 ...", flush=True)
    model = YOLO(str(TMP_DIR / "yolo11n-seg.pt"))

    tile_w, tile_h = 640, 360
    atlas_w, atlas_h = tile_w * 2, tile_h * 3  # 1280 x 1080
    sx_src, sy_src = tile_w / 1920.0, tile_h / 1080.0

    caps = []
    bgs_small = []
    for cid in LIVE_CAMS:
        caps.append(cv2.VideoCapture(str(OUT_DIR / f"{cid.lower()}.mp4")))
        bg = cv2.imread(str(OUT_DIR / f"{cid.lower()}_bg.png"))
        bgs_small.append(cv2.resize(bg, (tile_w, tile_h), interpolation=cv2.INTER_AREA).astype(np.int16))

    out_mp4 = OUT_DIR / "masks_w16.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-s", f"{atlas_w}x{atlas_h}",
        "-r", "30",
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    total_frames = 600
    last_ai_masks = [np.zeros((tile_h, tile_w), dtype=np.uint8) for _ in range(6)]
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    t0 = time.time()
    for f_idx in range(total_frames):
        imgs_small = []
        ok_all = True
        for cap in caps:
            ret, frm = cap.read()
            if not ret:
                ok_all = False
                break
            imgs_small.append(cv2.resize(frm, (tile_w, tile_h), interpolation=cv2.INTER_AREA))
        if not ok_all:
            break

        # 每 2 帧 (15Hz) 跑一次 6 路 batch YOLO11n-seg 推理，速度翻倍且零视觉差异
        if f_idx % 2 == 0:
            results = model.predict(
                imgs_small,
                imgsz=512,
                classes=[0],  # person (含大部分人形机器人)
                conf=0.12,
                verbose=False,
                retina_masks=False,
            )
            for c_i, res in enumerate(results):
                m_union = np.zeros((tile_h, tile_w), dtype=np.uint8)
                if res.masks is not None and len(res.masks.data) > 0:
                    m_np = (res.masks.data.cpu().numpy().max(axis=0) > 0.35).astype(np.uint8) * 255
                    if m_np.shape != (tile_h, tile_w):
                        m_np = cv2.resize(m_np, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
                    m_union = m_np
                last_ai_masks[c_i] = m_union

        gt_objs = gt.get(str(START_FRAME + f_idx), [])
        atlas_frame = np.zeros((atlas_h, atlas_w), dtype=np.uint8)

        for c_i, cid in enumerate(LIVE_CAMS):
            img_s = imgs_small[c_i].astype(np.int16)
            bg_s = bgs_small[c_i]

            # 统计本帧该相机所有 GT 目标框范围（防止背景静态噪声误入掩膜，同时为机器人/AGV 补充形态学实心轮廓）
            gt_union_roi = np.zeros((tile_h, tile_w), dtype=np.uint8)
            contour_fill_mask = np.zeros((tile_h, tile_w), dtype=np.uint8)

            # 预计算无暗影前景差分
            diff_max = np.max(np.abs(img_s - bg_s), axis=2)
            lum_f = (29 * img_s[:, :, 2] + 59 * img_s[:, :, 1] + 12 * img_s[:, :, 0]) // 100
            lum_b = (29 * bg_s[:, :, 2] + 59 * bg_s[:, :, 1] + 12 * bg_s[:, :, 0]) // 100
            chroma_d = np.abs((img_s[:, :, 2] - lum_f) - (bg_s[:, :, 2] - lum_b)) + np.abs(
                (img_s[:, :, 0] - lum_f) - (bg_s[:, :, 0] - lum_b)
            )
            cast_sh = (lum_f < (lum_b * 93) // 100) & (lum_f > (lum_b * 52) // 100) & (chroma_d < 12)
            fg_raw = ((diff_max > 20) & (~cast_sh)).astype(np.uint8) * 255

            for o in gt_objs:
                b = o.get("2d bounding box visible", {}).get(cid)
                if not b or len(b) != 4:
                    continue
                x1 = max(0, int(b[0] * sx_src) - 5)
                y1 = max(0, int(b[1] * sy_src) - 5)
                x2 = min(tile_w, int(b[2] * sx_src) + 5)
                y2 = min(tile_h, int(b[3] * sy_src) + 5)
                if x2 - x1 < 4 or y2 - y1 < 6:
                    continue
                gt_union_roi[y1:y2, x1:x2] = 255

                # 在单个目标框内做形态学闭运算 + 外轮廓实心填充，彻底补实黑色机器人与灰衣胸口空洞！
                sub = fg_raw[y1:y2, x1:x2]
                closed = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, kernel_close)
                cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filled = np.zeros_like(closed)
                for cnt in cnts:
                    if cv2.contourArea(cnt) >= 12:
                        cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
                contour_fill_mask[y1:y2, x1:x2] = np.maximum(contour_fill_mask[y1:y2, x1:x2], filled)

            # 融合 YOLO11-seg 语义掩膜 与 GT 框内轮廓实心掩膜，并做 5x5 羽化抗锯齿
            combined = np.bitwise_and(np.maximum(last_ai_masks[c_i], contour_fill_mask), gt_union_roi)
            combined = cv2.GaussianBlur(combined, (5, 5), 0)

            col = c_i % 2
            row = c_i // 2
            atlas_frame[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = combined

        proc.stdin.write(atlas_frame.tobytes())
        if (f_idx + 1) % 120 == 0:
            print(f"  [AI-SEG] 已完成 {f_idx + 1}/{total_frames} 帧 ({time.time() - t0:.1f}s)", flush=True)

    proc.stdin.close()
    proc.wait()
    for cap in caps:
        cap.release()

    if proc.returncode != 0:
        raise RuntimeError("FFmpeg 编码 masks_w16.mp4 失败")

    # 写入 park_cameras_w16.json
    cfg_path = OUT_DIR / "park_cameras_w16.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    sc = cfg.setdefault("scene", {})
    sc["maskAtlasUrl"] = "/demo-nvidia-w16/masks_w16.mp4"
    sc["maskCamOrder"] = [c.lower() for c in LIVE_CAMS]
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[AI-SEG] 成功生成 6路合一 AI 分割掩膜流 {out_mp4.name} ({out_mp4.stat().st_size / 1024 / 1024:.2f} MB)！",
        flush=True,
    )


def main():
    cal = json.loads((TMP_DIR / "calibration.json").read_text(encoding="utf-8"))
    gt = json.loads((TMP_DIR / "ground_truth.json").read_text(encoding="utf-8"))

    build_smart_uv_warehouse_glb(cal)
    build_ai_mask_atlas_video(gt)


if __name__ == "__main__":
    main()
