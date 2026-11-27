"""
使用 Blender 5.3 无头模式生成成套数字孪生视频融合实操资产：
1. 导出精确对齐 Three.js Y-Up 坐标系的静态园区模型: public/demo-blender/blender_park.glb
2. 渲染含动态巡检员与巡检小车的两路 720P H.264 监控视频: cam_fixed_01.mp4 / cam_fixed_02.mp4
3. 导出第 1 帧标定参考图及 8 组不共面 2D/3D 真值控制点 JSON 与初始相机配置 JSON
"""
import json
import math
import shutil
import subprocess
from pathlib import Path

import bpy
import bmesh
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "public" / "demo-blender"
TMP_DIR = ROOT_DIR / ".tmp_blender_frames"

RES_W = 1280
RES_H = 720
FPS = 24
NUM_FRAMES = 48
SENSOR_WIDTH_MM = 36.0
FX = 866.6667
LENS_MM = FX * SENSOR_WIDTH_MM / RES_W  # 24.375mm (对应约 45° 垂直视场角)


def to_blender(x_t, y_t, z_t):
    """Three.js Y-Up (Xt, Yt, Zt) -> Blender Z-Up (Xb, Yb, Zb)"""
    return (x_t, -z_t, y_t)


def make_pbr_mat(name, rgba, roughness=0.75, metallic=0.1, emission_strength=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
        if emission_strength > 0 and "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = rgba
            bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat


def add_box(name, center_three, size_three, mat, collection, walkable=False):
    """按 Three.js 中心坐标与宽高深创建长方体"""
    cx, cy, cz = to_blender(*center_three)
    sx, sy, sz = size_three[0], size_three[2], size_three[1]
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(cx, cy, cz))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (sx, sy, sz)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    if walkable:
        obj["isWalkable"] = True
    if obj.name not in collection.objects:
        collection.objects.link(obj)
    for col in list(obj.users_collection):
        if col != collection:
            col.objects.unlink(obj)
    return obj


def add_ramp(name, mat, collection):
    """创建东侧斜坡引道 (Three.js: x ∈ [5.5, 8.5], z 从 3.5 爬升至 0.5, 高度 0 -> 0.8m)"""
    mesh = bpy.data.meshes.new(name + "_Mesh")
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)

    bm = bmesh.new()
    v0 = bm.verts.new(to_blender(5.5, 0.0, 3.5))
    v1 = bm.verts.new(to_blender(8.5, 0.0, 3.5))
    v2 = bm.verts.new(to_blender(8.5, 0.0, 0.5))
    v3 = bm.verts.new(to_blender(5.5, 0.0, 0.5))
    v4 = bm.verts.new(to_blender(8.5, 0.8, 0.5))
    v5 = bm.verts.new(to_blender(5.5, 0.8, 0.5))

    bm.faces.new((v0, v1, v4, v5))  # 顶面斜坡
    bm.faces.new((v3, v2, v4, v5))  # 背面立面
    bm.faces.new((v0, v5, v3))      # 左侧三角面
    bm.faces.new((v1, v2, v4))      # 右侧三角面
    bm.faces.new((v0, v3, v2, v1))  # 底面
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    obj.data.materials.append(mat)
    obj["isWalkable"] = True
    return obj


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def build_scene():
    clear_scene()
    scene = bpy.context.scene
    static_col = bpy.data.collections.new("StaticPark")
    dyn_col = bpy.data.collections.new("DynamicActors")
    cam_col = bpy.data.collections.new("Cameras")
    scene.collection.children.link(static_col)
    scene.collection.children.link(dyn_col)
    scene.collection.children.link(cam_col)

    # 1. 环境天光与主太阳光
    world = bpy.data.worlds.new("ParkWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.12, 0.16, 0.22, 1.0)
        bg.inputs["Strength"].default_value = 0.8

    sun_data = bpy.data.lights.new(name="SunLight", type="SUN")
    sun_data.energy = 3.2
    sun_obj = bpy.data.objects.new(name="SunLight", object_data=sun_data)
    sun_obj.location = to_blender(18, 30, 22)
    sun_obj.rotation_euler = (math.radians(42), math.radians(15), math.radians(-30))
    cam_col.objects.link(sun_obj)

    # 2. 材质表
    mat_ground = make_pbr_mat("MatGround", (0.18, 0.20, 0.23, 1.0), roughness=0.9)
    mat_lane = make_pbr_mat("MatLane", (0.58, 0.62, 0.68, 1.0), roughness=0.7)
    mat_main = make_pbr_mat("MatMainBuilding", (0.45, 0.50, 0.56, 1.0), roughness=0.75)
    mat_trim = make_pbr_mat("MatTrim", (0.15, 0.35, 0.58, 1.0), roughness=0.5)
    mat_door = make_pbr_mat("MatDoor", (0.16, 0.18, 0.21, 1.0), roughness=0.6, metallic=0.3)
    mat_side = make_pbr_mat("MatSideBuilding", (0.36, 0.42, 0.48, 1.0), roughness=0.8)
    mat_pillar = make_pbr_mat("MatPillar", (0.68, 0.46, 0.30, 1.0), roughness=0.75)
    mat_equip = make_pbr_mat("MatEquip", (0.24, 0.48, 0.42, 1.0), roughness=0.55, metallic=0.25)
    mat_plat = make_pbr_mat("MatPlatform", (0.32, 0.38, 0.44, 1.0), roughness=0.75)
    mat_ramp = make_pbr_mat("MatRamp", (0.42, 0.49, 0.58, 1.0), roughness=0.75)

    # 3. 静态园区模型（对称 60x60 地面保证 center.x=0, center.z=0, min.y=0）
    static_meshes = []
    bpy.ops.mesh.primitive_plane_add(size=60.0, location=(0.0, 0.0, 0.0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.data.materials.append(mat_ground)
    ground["isWalkable"] = True
    static_col.objects.link(ground)
    for col in list(ground.users_collection):
        if col != static_col:
            col.objects.unlink(ground)
    static_meshes.append(ground)

    # 地面斑马线 / 道路标线，增强视觉锚点与真实监控感
    for lx in (-5.0, 0.0, 5.0):
        static_meshes.append(add_box(f"LaneMark_{lx}", (lx, 0.015, 1.0), (0.35, 0.03, 9.0), mat_lane, static_col))

    # 主厂房 + 屋顶女儿墙 + 工业大门
    static_meshes.append(add_box("MainBuilding", (0.0, 4.0, -8.0), (18.0, 8.0, 8.0), mat_main, static_col))
    static_meshes.append(add_box("MainRoofTrim", (0.0, 8.15, -8.0), (18.3, 0.3, 8.3), mat_trim, static_col))
    static_meshes.append(add_box("MainGate", (0.0, 2.0, -3.95), (5.0, 4.0, 0.15), mat_door, static_col))

    # 西侧配电室、前排遮挡立柱、设备箱、东侧 0.8m 巡检平台与斜坡
    static_meshes.append(add_box("SideBuilding", (-12.0, 2.5, -2.0), (6.0, 5.0, 10.0), mat_side, static_col))
    static_meshes.append(add_box("FrontPillar", (-3.0, 2.75, 1.0), (2.0, 5.5, 2.0), mat_pillar, static_col))
    static_meshes.append(add_box("EquipmentBox", (3.5, 1.2, 2.0), (3.2, 2.4, 2.0), mat_equip, static_col))
    static_meshes.append(add_box("InspectionPlatform", (7.0, 0.4, -2.5), (5.0, 0.8, 6.0), mat_plat, static_col, walkable=True))
    static_meshes.append(add_ramp("InspectionRamp", mat_ramp, static_col))

    # 仅选中静态园区对象导出 GLB
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    glb_path = OUT_DIR / "blender_park.glb"
    bpy.ops.object.select_all(action="DESELECT")
    for obj in static_meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = ground
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_apply=True,
        export_extras=True,
    )
    print(f"[Blender] 已导出静态园区模型: {glb_path}")

    # 4. 添加动态巡检员与巡检小车（仅出现在监控渲染视频中，不进入静态 GLB）
    mat_worker = make_pbr_mat("MatWorker", (0.98, 0.38, 0.06, 1.0), roughness=0.45, emission_strength=0.4)
    mat_helmet = make_pbr_mat("MatHelmet", (1.0, 0.85, 0.05, 1.0), roughness=0.35, emission_strength=0.5)
    mat_agv = make_pbr_mat("MatAGV", (0.04, 0.85, 0.95, 1.0), roughness=0.35, emission_strength=0.4)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.32, depth=1.55, location=(0, 0, 0.775))
    worker = bpy.context.active_object
    worker.name = "InspectorA"
    worker.data.materials.append(mat_worker)
    dyn_col.objects.link(worker)
    for col in list(worker.users_collection):
        if col != dyn_col:
            col.objects.unlink(worker)

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(0, 0, 0.96))
    helmet = bpy.context.active_object
    helmet.name = "InspectorA_Helmet"
    helmet.data.materials.append(mat_helmet)
    helmet.parent = worker
    dyn_col.objects.link(helmet)
    for col in list(helmet.users_collection):
        if col != dyn_col:
            col.objects.unlink(helmet)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.32))
    agv = bpy.context.active_object
    agv.name = "PatrolAGV_B"
    agv.scale = (0.9, 1.3, 0.64)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    agv.data.materials.append(mat_agv)
    dyn_col.objects.link(agv)
    for col in list(agv.users_collection):
        if col != dyn_col:
            col.objects.unlink(agv)

    scene.frame_start = 1
    scene.frame_end = NUM_FRAMES
    for f in range(1, NUM_FRAMES + 1):
        t = (f - 1) / NUM_FRAMES
        theta = t * 2.0 * math.pi
        wx = 5.5 * math.sin(theta)
        wz = -1.0 + 2.2 * math.cos(theta)
        worker.location = to_blender(wx, 0.775, wz)
        worker.keyframe_insert(data_path="location", frame=f)

        # AGV 沿斜坡 (z: 3.2 -> 0.5) 上下 0.8m 巡检平台 (z: 0.5 -> -3.0)
        pingpong = 0.5 * (1.0 - math.cos(theta))
        az = 3.2 - pingpong * 6.2
        if az <= 0.5:
            ay = 0.8
        elif az < 3.5:
            ay = 0.8 * ((3.5 - az) / 3.0)
        else:
            ay = 0.0
        agv.location = to_blender(7.0, ay + 0.32, az)
        agv.keyframe_insert(data_path="location", frame=f)

    # 5. 设置两路标准监控相机并渲染视频 + 计算精确 2D/3D 控制点
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 8
    scene.render.resolution_x = RES_W
    scene.render.resolution_y = RES_H
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.render.image_settings.file_format = "PNG"

    cam_specs = [
        {
            "id": "cam_fixed_01",
            "name": "Blender 左侧固定枪机",
            "pos_three": (-10.0, 8.5, 11.0),
            "target_three": (-1.0, 2.5, -6.0),
        },
        {
            "id": "cam_fixed_02",
            "name": "Blender 右侧固定枪机",
            "pos_three": (10.0, 8.5, 11.0),
            "target_three": (1.0, 2.5, -6.0),
        },
    ]

    # 8 个分布于不同深度的非共面静态园区角点（Three.js Y-Up 坐标）
    ctrl_pts_3d = [
        [-9.0, 8.0, -4.0],     # #1 主厂房前顶左角
        [9.0, 8.0, -4.0],      # #2 主厂房前顶右角
        [-9.0, 0.0, -4.0],     # #3 主厂房前底左角
        [9.0, 0.0, -4.0],      # #4 主厂房前底右角
        [-2.0, 5.5, 2.0],      # #5 前排立柱前顶右角
        [1.9, 2.4, 3.0],       # #6 设备箱前顶左角
        [4.5, 0.8, 0.5],       # #7 东侧巡检平台前顶左角
        [0.0, 0.0, 0.0],       # #8 园区地面原点
    ]

    fov_y = round(2.0 * math.degrees(math.atan(RES_H / (2.0 * FX))), 4)
    intrinsics = {
        "width": RES_W,
        "height": RES_H,
        "fx": FX,
        "fy": FX,
        "cx": RES_W * 0.5,
        "cy": RES_H * 0.5,
        "fovY": fov_y,
        "depthFovScale": 1.15,
        "near": 0.5,
        "far": 60.0,
    }
    distortion = {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0}

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)

    cameras_json = []
    for spec in cam_specs:
        cam_id = spec["id"]
        cam_data = bpy.data.cameras.new(cam_id)
        cam_data.sensor_fit = "HORIZONTAL"
        cam_data.sensor_width = SENSOR_WIDTH_MM
        cam_data.lens = LENS_MM
        cam_data.clip_start = 0.5
        cam_data.clip_end = 60.0

        cam_obj = bpy.data.objects.new(cam_id, cam_data)
        cam_obj.location = to_blender(*spec["pos_three"])
        cam_col.objects.link(cam_obj)

        tgt = bpy.data.objects.new(f"{cam_id}_Target", None)
        tgt.location = to_blender(*spec["target_three"])
        cam_col.objects.link(tgt)

        track = cam_obj.constraints.new(type="TRACK_TO")
        track.target = tgt
        track.track_axis = "TRACK_NEGATIVE_Z"
        track.up_axis = "UP_Y"

        scene.camera = cam_obj
        scene.frame_set(1)
        bpy.context.view_layer.update()

        # 计算精确 2D 像素真值
        pts_2d = []
        for p3 in ctrl_pts_3d:
            uvz = world_to_camera_view(scene, cam_obj, Vector(to_blender(*p3)))
            u = round(float(uvz.x * RES_W), 3)
            v = round(float((1.0 - uvz.y) * RES_H), 3)
            pts_2d.append([u, v])

        calib_payload = {
            "camId": cam_id,
            "imageSize": [RES_W, RES_H],
            "intrinsics": intrinsics,
            "distortion": distortion,
            "groundTruthPosition": list(spec["pos_three"]),
            "pts_3d": ctrl_pts_3d,
            "pts_2d": pts_2d,
        }
        calib_file = OUT_DIR / f"calib_points_{cam_id}.json"
        calib_file.write_text(json.dumps(calib_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        # 逐帧渲染 PNG 并封装 H.264 MP4
        frame_dir = TMP_DIR / cam_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        for f in range(1, NUM_FRAMES + 1):
            scene.frame_set(f)
            scene.render.filepath = str(frame_dir / f"frame_{f:04d}.png")
            bpy.ops.render.render(write_still=True)

        shutil.copyfile(frame_dir / "frame_0001.png", OUT_DIR / f"{cam_id}_ref.png")
        mp4_path = OUT_DIR / f"{cam_id}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(FPS),
                "-i", str(frame_dir / "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "21", "-movflags", "+faststart", "-an",
                str(mp4_path),
            ],
            check=True,
        )
        print(f"[Blender] 已生成监控视频与参考帧: {mp4_path}")

        cameras_json.append({
            "id": cam_id,
            "name": spec["name"],
            "type": "fixed",
            "enabled": True,
            "stream": {
                "type": "mp4",
                "url": f"/demo-blender/{cam_id}.mp4",
                "bgUrl": f"/demo-blender/{cam_id}_bg.png",
                "autoSuspend": True,
            },
            "intrinsics": intrinsics,
            "distortion": distortion,
            "colorAdjust": {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0},
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
    cfg_path = OUT_DIR / "park_cameras_blender.json"
    cfg_path.write_text(json.dumps(park_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Blender] 已写入初始相机配置: {cfg_path}")

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)


if __name__ == "__main__":
    build_scene()
