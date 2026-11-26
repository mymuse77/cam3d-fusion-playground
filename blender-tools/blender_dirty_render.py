"""
Blender 5.3 实景脏数据渲染子脚本（由 generate_dirty_suite.py 调用）：
在园区白模基础上加入施工尺寸偏差、违建雨棚与遮挡树木，并添加支架滚转歪斜，
以更宽视场角 (1600x900) 渲染无畸变底片并导出相机内点对应射线，供 OpenCV 施加广角桶形畸变与 ISP 色差。
"""
import json
import math
import shutil
from pathlib import Path

import bpy
import bmesh
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

ROOT_DIR = Path(__file__).resolve().parent.parent
TMP_DIR = ROOT_DIR / ".tmp_dirty_frames"

WIDE_W = 1600
WIDE_H = 900
WIDE_FX = 560.0
SENSOR_WIDTH_MM = 36.0
WIDE_LENS_MM = WIDE_FX * SENSOR_WIDTH_MM / WIDE_W  # 12.6mm 广角底片
FPS = 24
NUM_FRAMES = 48


def to_blender(x_t, y_t, z_t):
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


def add_box(name, center_three, size_three, mat, collection):
    cx, cy, cz = to_blender(*center_three)
    sx, sy, sz = size_three[0], size_three[2], size_three[1]
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(cx, cy, cz))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (sx, sy, sz)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    collection.objects.link(obj)
    for col in list(obj.users_collection):
        if col != collection:
            col.objects.unlink(obj)
    return obj


def add_ramp(name, mat, collection):
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
    bm.faces.new((v0, v1, v4, v5))
    bm.faces.new((v3, v2, v4, v5))
    bm.faces.new((v0, v5, v3))
    bm.faces.new((v1, v2, v4))
    bm.faces.new((v0, v3, v2, v1))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    obj.data.materials.append(mat)
    return obj


def run_render():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    col = bpy.data.collections.new("DirtyFieldScene")
    scene.collection.children.link(col)

    world = bpy.data.worlds.new("DirtyWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.14, 0.17, 0.22, 1.0)
        bg.inputs["Strength"].default_value = 0.85

    sun_data = bpy.data.lights.new(name="SunLight", type="SUN")
    sun_data.energy = 3.2
    sun_obj = bpy.data.objects.new(name="SunLight", object_data=sun_data)
    sun_obj.location = to_blender(18, 30, 22)
    sun_obj.rotation_euler = (math.radians(42), math.radians(15), math.radians(-30))
    col.objects.link(sun_obj)

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

    # 现场额外违建与遮挡物材质
    mat_trunk = make_pbr_mat("MatTrunk", (0.32, 0.21, 0.14, 1.0), roughness=0.9)
    mat_leaf = make_pbr_mat("MatLeaf", (0.14, 0.45, 0.19, 1.0), roughness=0.8)
    mat_canopy = make_pbr_mat("MatCanopy", (0.12, 0.42, 0.78, 1.0), roughness=0.4)
    mat_cargo = make_pbr_mat("MatCargo", (0.72, 0.55, 0.28, 1.0), roughness=0.7)

    bpy.ops.mesh.primitive_plane_add(size=60.0, location=(0.0, 0.0, 0.0))
    ground = bpy.context.active_object
    ground.data.materials.append(mat_ground)
    col.objects.link(ground)
    for c in list(ground.users_collection):
        if c != col:
            c.objects.unlink(ground)

    for lx in (-5.0, 0.0, 5.0):
        add_box(f"LaneMark_{lx}", (lx, 0.015, 1.0), (0.35, 0.03, 9.0), mat_lane, col)

    add_box("MainBuilding", (0.0, 4.0, -8.0), (18.0, 8.0, 8.0), mat_main, col)
    add_box("MainRoofTrim", (0.0, 8.15, -8.0), (18.3, 0.3, 8.3), mat_trim, col)
    add_box("MainGate", (0.0, 2.0, -3.95), (5.0, 4.0, 0.15), mat_door, col)
    add_box("SideBuilding", (-12.0, 2.5, -2.0), (6.0, 5.0, 10.0), mat_side, col)
    add_box("FrontPillar", (-3.0, 2.75, 1.0), (2.0, 5.5, 2.0), mat_pillar, col)

    # 模实偏差 1：绿色设备箱向东偏移 +0.38m、增高 +0.22m（设计白模仍在 (3.5, 1.2, 2.0)）
    add_box("EquipmentBox_FieldShifted", (3.88, 1.31, 2.0), (3.2, 2.62, 2.0), mat_equip, col)
    add_box("InspectionPlatform", (7.0, 0.4, -2.5), (5.0, 0.8, 6.0), mat_plat, col)
    add_ramp("InspectionRamp", mat_ramp, col)

    # 模实偏差 2：现场新增遮挡乔木 + 临时蓝色违建雨棚与堆放货箱
    bpy.ops.mesh.primitive_cylinder_add(radius=0.28, depth=3.2, location=to_blender(3.6, 1.6, -2.6))
    trunk = bpy.context.active_object
    trunk.data.materials.append(mat_trunk)
    col.objects.link(trunk)
    for c in list(trunk.users_collection):
        if c != col:
            c.objects.unlink(trunk)

    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.55, location=to_blender(3.6, 3.6, -2.6))
    canopy_tree = bpy.context.active_object
    canopy_tree.data.materials.append(mat_leaf)
    col.objects.link(canopy_tree)
    for c in list(canopy_tree.users_collection):
        if c != col:
            c.objects.unlink(canopy_tree)

    add_box("TempBlueShed", (-8.5, 2.1, 2.5), (3.4, 0.18, 2.8), mat_canopy, col)
    add_box("TempCargoA", (-8.2, 0.55, 2.4), (1.3, 1.1, 1.2), mat_cargo, col)

    # 动态目标
    mat_worker = make_pbr_mat("MatWorker", (0.98, 0.38, 0.06, 1.0), roughness=0.45, emission_strength=0.4)
    mat_helmet = make_pbr_mat("MatHelmet", (1.0, 0.85, 0.05, 1.0), roughness=0.35, emission_strength=0.5)
    mat_agv = make_pbr_mat("MatAGV", (0.04, 0.85, 0.95, 1.0), roughness=0.35, emission_strength=0.4)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.32, depth=1.55, location=(0, 0, 0.775))
    worker = bpy.context.active_object
    worker.data.materials.append(mat_worker)
    col.objects.link(worker)
    for c in list(worker.users_collection):
        if c != col:
            c.objects.unlink(worker)

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(0, 0, 0.96))
    helmet = bpy.context.active_object
    helmet.data.materials.append(mat_helmet)
    helmet.parent = worker
    col.objects.link(helmet)
    for c in list(helmet.users_collection):
        if c != col:
            c.objects.unlink(helmet)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.32))
    agv = bpy.context.active_object
    agv.scale = (0.9, 1.3, 0.64)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    agv.data.materials.append(mat_agv)
    col.objects.link(agv)
    for c in list(agv.users_collection):
        if c != col:
            c.objects.unlink(agv)

    scene.frame_start = 1
    scene.frame_end = NUM_FRAMES
    for f in range(1, NUM_FRAMES + 1):
        t = (f - 1) / NUM_FRAMES
        theta = t * 2.0 * math.pi
        worker.location = to_blender(5.5 * math.sin(theta), 0.775, -1.0 + 2.2 * math.cos(theta))
        worker.keyframe_insert(data_path="location", frame=f)

        pingpong = 0.5 * (1.0 - math.cos(theta))
        az = 3.2 - pingpong * 6.2
        ay = 0.8 if az <= 0.5 else 0.8 * ((3.5 - az) / 3.0)
        agv.location = to_blender(7.0, ay + 0.32, az)
        agv.keyframe_insert(data_path="location", frame=f)

    # 渲染设置
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 8
    scene.render.resolution_x = WIDE_W
    scene.render.resolution_y = WIDE_H
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    # 前 8 个清晰无偏差建筑/立柱/高台角点 + 第 9 个移位设备箱遮挡误点
    ctrl_pts_3d = [
        [-9.0, 8.0, -4.0],
        [9.0, 8.0, -4.0],
        [-9.0, 0.0, -4.0],
        [9.0, 0.0, -4.0],
        [-2.0, 5.5, 2.0],
        [4.5, 0.8, 0.5],
        [-4.0, 5.5, 2.0],
        [0.0, 0.0, 0.0],
        [5.1, 2.4, 3.0],  # #9 对应移位后的设备箱，同时后续将叠加外点偏差
    ]

    cam_specs = [
        {
            "id": "cam_dirty_01",
            "pos_three": (-10.0, 8.5, 11.0),
            "target_three": (-1.0, 2.5, -6.0),
            "roll_deg": 2.6,
        },
        {
            "id": "cam_dirty_02",
            "pos_three": (10.0, 8.5, 11.0),
            "target_three": (1.0, 2.5, -6.0),
            "roll_deg": -2.2,
        },
    ]

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    meta = {"wideW": WIDE_W, "wideH": WIDE_H, "wideFx": WIDE_FX, "cameras": {}}
    for spec in cam_specs:
        cid = spec["id"]
        cam_data = bpy.data.cameras.new(cid)
        cam_data.sensor_fit = "HORIZONTAL"
        cam_data.sensor_width = SENSOR_WIDTH_MM
        cam_data.lens = WIDE_LENS_MM
        cam_data.clip_start = 0.5
        cam_data.clip_end = 70.0

        cam_obj = bpy.data.objects.new(cid, cam_data)
        cam_obj.location = to_blender(*spec["pos_three"])
        col.objects.link(cam_obj)

        tgt = bpy.data.objects.new(f"{cid}_Target", None)
        tgt.location = to_blender(*spec["target_three"])
        col.objects.link(tgt)

        track = cam_obj.constraints.new(type="TRACK_TO")
        track.target = tgt
        track.track_axis = "TRACK_NEGATIVE_Z"
        track.up_axis = "UP_Y"

        scene.camera = cam_obj
        scene.frame_set(1)
        bpy.context.view_layer.update()

        # 固化 TRACK_TO 后叠加真实的安装支架滚转偏差 Roll
        mw = cam_obj.matrix_world.copy()
        cam_obj.constraints.clear()
        cam_obj.matrix_world = mw @ Matrix.Rotation(math.radians(spec["roll_deg"]), 4, "Z")
        bpy.context.view_layer.update()

        norm_rays = []
        for p3 in ctrl_pts_3d:
            uvz = world_to_camera_view(scene, cam_obj, Vector(to_blender(*p3)))
            u_wide = float(uvz.x * WIDE_W)
            v_wide = float((1.0 - uvz.y) * WIDE_H)
            xn = (u_wide - WIDE_W * 0.5) / WIDE_FX
            yn = (v_wide - WIDE_H * 0.5) / WIDE_FX
            norm_rays.append([xn, yn])

        meta["cameras"][cid] = {
            "groundTruthPosition": list(spec["pos_three"]),
            "rollDeg": spec["roll_deg"],
            "pts_3d": ctrl_pts_3d,
            "normRays": norm_rays,
        }

        frame_dir = TMP_DIR / cid
        frame_dir.mkdir(parents=True, exist_ok=True)
        for f in range(1, NUM_FRAMES + 1):
            scene.frame_set(f)
            scene.render.filepath = str(frame_dir / f"frame_{f:04d}.png")
            bpy.ops.render.render(write_still=True)

    (TMP_DIR / "dirty_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[BlenderDirty] 广角底片渲染完成")


if __name__ == "__main__":
    run_render()
