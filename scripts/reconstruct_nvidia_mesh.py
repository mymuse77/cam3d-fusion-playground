#!/usr/bin/env python3
"""
利用 NVIDIA PhysicalAI-SmartSpaces (Warehouse_016) 官方 depth_maps/*.h5
通过 HTTP Range 零全量下载（仅按需读取每路 5 帧静态深度图，共 ~50MB 替代 40GB），
融合 6 路相机真实毫米级深度反投影，重建像素级对齐的 warehouse_016.glb。
"""

import http.client
import io
import json
import subprocess
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = ROOT / ".tmp_nvidia_w16"
DEPTH_CACHE = TMP_DIR / "depth_static"
OUT_DIR = ROOT / "public" / "demo-nvidia-w16"
BLENDER_EXE = Path(r"D:\Env\Blender 5.3\blender.exe")

HF_BASE = (
    "https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/resolve/main/"
    "MTMC_Tracking_2025/val/Warehouse_016"
)

SELECTED_CAMS = ["Camera", "Camera_01", "Camera_02", "Camera_03", "Camera_05", "Camera_06"]
# 选取 3 个间隔 2 秒的早期帧（紧邻文件头 B-Tree 浅层，单路约 10 秒），取逐像素最大值剔除移动人物/机器人/AGV
SAMPLE_FRAMES = [0, 60, 120]

CENTER_OFFSET_W = np.array([0.0, -10.0, 0.0], dtype=np.float64)
T_Z2Y = np.array([
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)


class HTTPRangeFile(io.RawIOBase):
    """持久 HTTPS Keep-Alive + 块缓存的 HTTP Range 随机读取器，支持 h5py 远程按需切片"""

    def __init__(self, url: str, block_size: int = 524288):
        self.url = url
        self.block_size = block_size
        self.pos = 0
        self.bytes_fetched = 0
        self.cache = {}

        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            self.final_url = resp.geturl()
            self.size = int(resp.headers["Content-Length"])

        parsed = urllib.parse.urlparse(self.final_url)
        self.host = parsed.netloc
        self.path = parsed.path + ("?" + parsed.query if parsed.query else "")
        self.conn = http.client.HTTPSConnection(self.host, timeout=15)

    def _reconnect(self):
        try:
            self.conn.close()
        except Exception:
            pass
        self.conn = http.client.HTTPSConnection(self.host, timeout=15)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        self.pos = max(0, min(self.pos, self.size))
        return self.pos

    def _fetch_range(self, start: int, end: int) -> bytes:
        for attempt in range(4):
            try:
                self.conn.request(
                    "GET",
                    self.path,
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "User-Agent": "Mozilla/5.0",
                        "Connection": "keep-alive",
                    },
                )
                resp = self.conn.getresponse()
                data = resp.read()
                if resp.status not in (200, 206):
                    raise RuntimeError(f"HTTP {resp.status}")
                self.bytes_fetched += len(data)
                return data
            except Exception:
                self._reconnect()
                if attempt == 3:
                    raise

    def readinto(self, b):
        n = len(b)
        if self.pos >= self.size or n == 0:
            return 0
        n = min(n, self.size - self.pos)
        out = bytearray()
        cur = self.pos
        remaining = n
        while remaining > 0:
            if remaining > self.block_size * 2:
                chunk = self._fetch_range(cur, cur + remaining - 1)
                out.extend(chunk)
                cur += len(chunk)
                remaining -= len(chunk)
                break
            blk = cur // self.block_size
            if blk not in self.cache:
                b_start = blk * self.block_size
                b_end = min(self.size - 1, b_start + self.block_size - 1)
                self.cache[blk] = self._fetch_range(b_start, b_end)
                if len(self.cache) > 64:
                    self.cache.pop(next(iter(self.cache)))
            blk_data = self.cache[blk]
            off = cur - blk * self.block_size
            take = min(remaining, len(blk_data) - off)
            out.extend(blk_data[off : off + take])
            cur += take
            remaining -= take
        b[: len(out)] = out
        self.pos += len(out)
        return len(out)


def fetch_static_depth(cam_id: str) -> Path:
    """远程按需读取 5 帧 uint16 深度图，取第 2 大有效深度（去除动态人物/机器人），缓存为 .npy"""
    DEPTH_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = DEPTH_CACHE / f"{cam_id}_static_depth.npy"
    if cache_path.exists() and cache_path.stat().st_size > 100_000:
        print(f"[DEPTH] 复用缓存静态深度图: {cache_path.name}", flush=True)
        return cache_path

    url = f"{HF_BASE}/depth_maps/{cam_id}.h5"
    print(f"[DEPTH] 远程 HTTP Range 采样 {cam_id}.h5 ({len(SAMPLE_FRAMES)} 帧) ...", flush=True)
    rf = HTTPRangeFile(url, block_size=524288)
    frames = []
    with h5py.File(rf, "r") as f:
        for fid in SAMPLE_FRAMES:
            key = f"distance_to_image_plane_{fid:05d}.png"
            if key in f:
                arr = f[key][()].astype(np.float32) / 1000.0  # mm -> meters
                frames.append(arr)
                print(f"    [{cam_id}] 已读取 {key} (累计 {rf.bytes_fetched/1e6:.2f} MB)", flush=True)
    if not frames:
        raise RuntimeError(f"{cam_id}.h5 中未找到指定深度帧")

    stack = np.stack(frames, axis=0)
    # 剔除无效值 (0 或 > 35m)
    valid = (stack > 0.3) & (stack < 35.0)
    stack_clean = np.where(valid, stack, np.nan)
    # 动态物体比背景更近，取逐像素最大有效深度即可抹除移动中的行人/机器人/AGV
    static_depth = np.nanmax(stack_clean, axis=0).astype(np.float32)
    static_depth = np.nan_to_num(static_depth, nan=0.0)
    np.save(cache_path, static_depth)
    print(
        f"[DEPTH] 完成 {cam_id}: 仅下载 {rf.bytes_fetched / 1e6:.2f} MB / {rf.size / 1e9:.2f} GB",
        flush=True,
    )
    return cache_path


def fuse_depth_to_heightfield(cal: dict, grid_res: float = 0.045, half_span: float = 9.82):
    """将 6 路静态深度图严格反投影到 Three.js Y-Up 坐标系，融合为高精 2.5D 表面高度场"""
    sensors_by_id = {s["id"]: s for s in cal["sensors"]}
    n = int(round((2.0 * half_span) / grid_res))
    # 记录每个网格单元的高程样本：用 max-bucket（忽略 >3.6m 吊灯/天花板）并统计计数以抗单点噪声
    h_max = np.full((n, n), -1.0, dtype=np.float32)
    h_2nd = np.full((n, n), -1.0, dtype=np.float32)
    counts = np.zeros((n, n), dtype=np.int32)

    # 每隔 2 像素采样一次 (960x540 = 518,400 点/相机 × 6路 ≈ 310万点)，既密集又秒算
    stride = 2
    floor_errs = []

    for cid in SELECTED_CAMS:
        depth = np.load(DEPTH_CACHE / f"{cid}_static_depth.npy")
        H, W = depth.shape
        sensor = sensors_by_id[cid]
        K = np.array(sensor["intrinsicMatrix"], dtype=np.float64)
        M = np.array(sensor["extrinsicMatrix"], dtype=np.float64)
        R_w2c = M[:, :3]
        t_w2c = M[:, 3]
        R_c2w = R_w2c.T
        cam_pos_w = -R_c2w @ t_w2c

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        ys = np.arange(0, H, stride, dtype=np.float64)
        xs = np.arange(0, W, stride, dtype=np.float64)
        uu, vv = np.meshgrid(xs, ys)
        zz = depth[::stride, ::stride].astype(np.float64)

        mask = (zz > 0.5) & (zz < 28.0)
        u_v = uu[mask]
        v_v = vv[mask]
        z_v = zz[mask]

        # OpenCV 相机系 3D 点
        xc = (u_v - cx) * z_v / fx
        yc = (v_v - cy) * z_v / fy
        P_cam = np.stack([xc, yc, z_v], axis=0)  # (3, N)

        # 转回 NVIDIA Z-Up 世界系，再转 Three.js Y-Up（仓库中心归零）
        P_w = R_c2w @ P_cam + cam_pos_w[:, None]
        P_t = T_Z2Y @ (P_w - CENTER_OFFSET_W[:, None])  # (3, N): [X_t, Y_t(up), Z_t]

        xt, yt, zt = P_t[0], P_t[1], P_t[2]

        # 校验地面精度
        floor_mask = (yt > -0.08) & (yt < 0.08) & (np.abs(xt) < 9.5) & (np.abs(zt) < 9.5)
        if np.any(floor_mask):
            floor_errs.append(float(np.median(yt[floor_mask])))

        # 仅保留室内有效高度（过滤吊灯、天花板、墙外点）
        in_box = (
            (xt >= -half_span) & (xt < half_span)
            & (zt >= -half_span) & (zt < half_span)
            & (yt >= -0.05) & (yt <= 3.45)
        )
        xt, yt, zt = xt[in_box], yt[in_box], zt[in_box]

        ix = np.floor((xt + half_span) / grid_res).astype(np.int32)
        iz = np.floor((zt + half_span) / grid_res).astype(np.int32)
        ix = np.clip(ix, 0, n - 1)
        iz = np.clip(iz, 0, n - 1)
        yt_f = yt.astype(np.float32)

        # 用两档最大值 (h_max, h_2nd) 防止单像素飞点，同时保留箱子顶面锐利高度
        # 先按高度升序，利用 np.maximum.at 更新
        np.add.at(counts, (iz, ix), 1)
        # 为抗单点噪声：把略低于当前 h_max 的记录进 h_2nd，简化为两次遍历或直接 maximum + median_filter
        np.maximum.at(h_max, (iz, ix), yt_f)

    print(f"[DEPTH] 6 路深度地面中值高度校准检查: {[round(e, 4) for e in floor_errs]} m (理论 0.0000 m)", flush=True)

    # 未观测到的角落填 0.0
    valid_cells = counts > 0
    hf = np.where(valid_cells, np.maximum(0.0, h_max), 0.0).astype(np.float32)

    # 对微小未观测缝隙（如被箱子遮挡的 1-2 格）做轻量空洞修复，不破坏已有高度
    # 先做 3x3 中值滤波消除孤立飞点（如残余人物轮廓噪点），再将地面 < 4.5cm 归零保证平整
    hf_med = median_filter(hf, size=3)
    # 保留锐利箱体边缘：仅在局部与中值相差 >0.25m 的孤立飞点处替换
    spike = np.abs(hf - hf_med) > 0.25
    hf = np.where(spike, hf_med, hf)
    # 再做一次温和的 3x3 中值平滑表面微锯齿
    hf = median_filter(hf, size=3)
    hf[hf < 0.045] = 0.0

    out_npz = TMP_DIR / "warehouse_heightfield.npz"
    np.savez_compressed(out_npz, hf=hf, grid_res=grid_res, half_span=half_span)
    print(
        f"[DEPTH] 高度场融合完成: {hf.shape} (分辨率 {grid_res*100:.1f} cm), "
        f"非地面单元占比 {(hf > 0.05).mean()*100:.1f}%, 最大高度 {hf.max():.2f} m",
        flush=True,
    )
    return out_npz


def build_mesh_glb_in_blender(npz_path: Path):
    """在 Blender 5.3 中将真实深度高度场构造成高质量减面 3D 网格 + 精准周界墙体并导出 GLB"""
    glb_path = OUT_DIR / "warehouse_016.glb"

    blender_code = f"""
import bpy, bmesh
import numpy as np

bpy.ops.wm.read_factory_settings(use_empty=True)

data = np.load(r"{str(npz_path)}")
hf = data["hf"]
grid_res = float(data["grid_res"])
half_span = float(data["half_span"])
nz, nx = hf.shape

def make_mat(name, rgba, rough=0.84, metal=0.04):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = rough
        bsdf.inputs["Metallic"].default_value = metal
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

mat_floor = make_mat("Warehouse_Floor_Mat", (0.46, 0.47, 0.49, 1.0), rough=0.88)
mat_mesh = make_mat("Warehouse_DepthMesh_Mat", (0.56, 0.54, 0.50, 1.0), rough=0.82)
mat_wall = make_mat("Warehouse_Wall_Mat", (0.72, 0.74, 0.76, 1.0), rough=0.85)
mat_dado = make_mat("Warehouse_Dado_Mat", (0.14, 0.32, 0.56, 1.0), rough=0.65)

# 1. 独立平整可行走地面 (Y=0.0, isWalkable=True)，确保 3D 视频立牌与反投影射线稳定落脚
add_box("Warehouse_Floor", (0.0, -0.15, 0.0), (20.2, 0.30, 20.2), mat_floor, walkable=True)

# 2. 基于 6 路深度图反推的真实 3D 室内物体表面网格（仅保留有高度起伏区域，地面由平滑地板承接避免 Z-fighting）
mesh = bpy.data.meshes.new("Warehouse_DepthMesh")
obj = bpy.data.objects.new("Warehouse_DepthMesh", mesh)
bpy.context.collection.objects.link(obj)
obj.data.materials.append(mat_mesh)

# 构造顶点: Three(x, y, z) -> Blender(x, -z, y)
xs = -half_span + (np.arange(nx, dtype=np.float32) + 0.5) * grid_res
zs = -half_span + (np.arange(nz, dtype=np.float32) + 0.5) * grid_res
gx, gz = np.meshgrid(xs, zs)

# 为了让所有高出地面的箱子/托盘/货架底座紧密贴地，保留其外扩 1 圈地面邻接点，剔除纯平地大面片
elevated = hf > 0.04
# 3x3 膨胀标记保留边界裙边
pad = np.pad(elevated, 1, mode='constant', constant_values=False)
keep_vert = (
    pad[0:-2, 0:-2] | pad[0:-2, 1:-1] | pad[0:-2, 2:] |
    pad[1:-1, 0:-2] | pad[1:-1, 1:-1] | pad[1:-1, 2:] |
    pad[2:,   0:-2] | pad[2:,   1:-1] | pad[2:,   2:]
)

# 建立压缩顶点索引表
vert_idx = np.full((nz, nx), -1, dtype=np.int32)
ky, kx = np.where(keep_vert)
vert_idx[ky, kx] = np.arange(len(ky), dtype=np.int32)

# Blender coords: (x, -z, y)
verts = np.stack([gx[ky, kx], -gz[ky, kx], hf[ky, kx]], axis=1).tolist()

# 仅生成至少有一个顶点高出地面的 quad 面片（避免与 Warehouse_Floor 重叠闪烁）
v00 = vert_idx[:-1, :-1]
v10 = vert_idx[:-1, 1:]
v11 = vert_idx[1:, 1:]
v01 = vert_idx[1:, :-1]
has_elev = elevated[:-1, :-1] | elevated[:-1, 1:] | elevated[1:, 1:] | elevated[1:, :-1]
valid_q = has_elev & (v00 >= 0) & (v10 >= 0) & (v11 >= 0) & (v01 >= 0)
qy, qx = np.where(valid_q)

# 注意 Blender 坐标 (x, -z, y) 下：x 增向右，-z 增向前（gz 增向后），保持法线朝上 (+Z_blender)
faces = np.stack([v00[qy, qx], v01[qy, qx], v11[qy, qx], v10[qy, qx]], axis=1).tolist()

mesh.from_pydata(verts, [], faces)
mesh.update()

# 轻量减面（保留箱体与货架锐利折角，同时将面数压缩到浏览器 60FPS 流畅级别）
bpy.context.view_layer.objects.active = obj
mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
mod.decimate_type = 'COLLAPSE'
mod.ratio = 0.28
bpy.ops.object.modifier_apply(modifier="Decimate")

# 重新计算一致朝外法线，采用平面着色保持纸箱/货架棱角清晰
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

print(f"DEPTH_MESH_VERTS={{len(obj.data.vertices)}} POLYS={{len(obj.data.polygons)}}")

# 3. 精确内壁周界墙体 (±9.82m) + 蓝色墙裙
wall_h = 6.6
wall_t = 0.45
inner = 9.82
wall_c = inner + wall_t * 0.5
span = inner * 2.0 + wall_t * 2.0

add_box("Wall_North", (0.0, wall_h * 0.5, -wall_c), (span, wall_h, wall_t), mat_wall)
add_box("Wall_South", (0.0, wall_h * 0.5,  wall_c), (span, wall_h, wall_t), mat_wall)
add_box("Wall_West",  (-wall_c, wall_h * 0.5, 0.0), (wall_t, wall_h, span), mat_wall)
add_box("Wall_East",  ( wall_c, wall_h * 0.5, 0.0), (wall_t, wall_h, span), mat_wall)

add_box("Dado_North", (0.0, 1.42, -inner + 0.015), (inner * 2.0, 0.45, 0.03), mat_dado)
add_box("Dado_South", (0.0, 1.42,  inner - 0.015), (inner * 2.0, 0.45, 0.03), mat_dado)
add_box("Dado_West",  (-inner + 0.015, 1.42, 0.0), (0.03, 0.45, inner * 2.0), mat_dado)
add_box("Dado_East",  ( inner - 0.015, 1.42, 0.0), (0.03, 0.45, inner * 2.0), mat_dado)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.export_scene.gltf(
    filepath=r"{str(glb_path)}",
    export_format='GLB',
    use_selection=True,
    export_yup=True,
    export_apply=True,
    export_extras=True,
)
print("EXPORT_DEPTH_GLB_OK")
"""
    script_path = TMP_DIR / "build_depth_glb.py"
    script_path.write_text(blender_code, encoding="utf-8")

    print("[DEPTH] 正在调用 Blender 5.3 生成真实深度重建 3D 网格 warehouse_016.glb ...", flush=True)
    res = subprocess.run(
        [str(BLENDER_EXE), "--background", "--factory-startup", "--python", str(script_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in res.stdout.splitlines():
        if "DEPTH_MESH_VERTS" in line or "EXPORT_DEPTH_GLB_OK" in line:
            print(" ", line, flush=True)
    if "EXPORT_DEPTH_GLB_OK" not in res.stdout:
        print(res.stdout[-600:])
        print(res.stderr[-600:])
        raise RuntimeError("Blender 深度网格导出未确认成功")
    print(f"[DEPTH] 成功导出 {glb_path.name} ({glb_path.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)


def main():
    cal = json.loads((TMP_DIR / "calibration.json").read_text(encoding="utf-8"))

    # 1. 顺序 HTTP Range 拉取 6 路静态深度图（避免 h5py C 底层全局锁在多线程下阻塞）
    for cid in SELECTED_CAMS:
        fetch_static_depth(cid)

    # 2. 6 路毫米级深度反投影融合为高度场
    npz_path = fuse_depth_to_heightfield(cal, grid_res=0.045, half_span=9.82)

    # 3. Blender 构建减面 3D 网格并导出 warehouse_016.glb
    build_mesh_glb_in_blender(npz_path)


if __name__ == "__main__":
    main()
