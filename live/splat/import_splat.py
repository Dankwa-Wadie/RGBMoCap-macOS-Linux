"""Blender: load a 3D Gaussian Splat (.ply) as a room around the avatar.

    Blender AVATAR.blend -b --python live/splat/import_splat.py -- SPLAT.ply ALIGN.json OUT.blend [MAX]

Blender has no native splat renderer, so each Gaussian becomes a small ellipsoid
instance (Geometry Nodes) with its own position, rotation, size, base colour (SH
degree 0) and opacity: an approximation that works in the viewport and in EEVEE
renders. The ALIGN.json matrix (live/splat/align.py) makes the room metric and
Z-up with the floor at Z = 0. Keeps the MAX most visible Gaussians (default 400k).
"""
import json, sys
from pathlib import Path
import bpy
import numpy as np

args = sys.argv[sys.argv.index('--') + 1:]
ply, align, out = Path(args[0]), Path(args[1]), Path(args[2])
limit = int(args[3]) if len(args) > 3 else 400_000

# --- binary little-endian PLY reader (standard 3DGS layout) ---------------------------------
raw = ply.read_bytes(); end = raw.index(b'end_header\n') + len(b'end_header\n')
header = raw[:end].decode().splitlines()
count = next(int(l.split()[2]) for l in header if l.startswith('element vertex'))
types = {'float': 'f4', 'double': 'f8', 'uchar': 'u1', 'int': 'i4', 'uint': 'u4', 'short': 'i2'}
fields = [(l.split()[2], '<' + types[l.split()[1]]) for l in header if l.startswith('property ')]
v = np.frombuffer(raw, dtype=np.dtype(fields), count=count, offset=end)

xyz = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)
opacity = 1 / (1 + np.exp(-v['opacity'].astype(np.float64)))
scales = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1).astype(np.float64))
quat = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], 1).astype(np.float64)   # w, x, y, z
quat /= np.linalg.norm(quat, axis=1, keepdims=True)
C0 = 0.28209479177387814
color = np.clip(0.5 + C0 * np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], 1), 0, 1)

# --- COLMAP -> Blender (metric, Z up) first, so filters below are in metres ------------
M = np.array(json.loads(align.read_text())['matrix'])
s = np.cbrt(np.linalg.det(M[:3, :3])); R = M[:3, :3] / s
xyz = xyz @ M[:3, :3].T + M[:3, 3]
scales = scales * s
# Keep visible, small Gaussians inside the room (drop floaters and huge background blobs),
# then the most opaque ones.
lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0)
keep = ((opacity > 0.2) & (scales.max(1) < 0.12) & np.all((xyz > lo) & (xyz < hi), 1))
order = np.argsort(-opacity[keep])[:limit]
idx = np.flatnonzero(keep)[order]
xyz, opacity, scales, quat, color = xyz[idx], opacity[idx], scales[idx], quat[idx], color[idx]

def mat_to_quat(m):
    w = np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    return np.array([w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)])


def qmul(a, b):
    aw, ax, ay, az = a.T if a.ndim > 1 else a; bw, bx, by, bz = b.T
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], 1)


quat = qmul(mat_to_quat(R), quat)

# --- point mesh with per-Gaussian attributes ------------------------------------------------
mesh = bpy.data.meshes.new('RoomSplat')
mesh.vertices.add(len(xyz)); mesh.vertices.foreach_set('co', xyz.astype(np.float32).ravel())
for name, kind, data in (('splat_color', 'FLOAT_COLOR', np.c_[color, opacity]),
                         ('splat_scale', 'FLOAT_VECTOR', scales * 1.5),   # ellipsoid out to ~1.5 sigma
                         ('splat_rot', 'QUATERNION', quat),
                         ('splat_alpha', 'FLOAT', opacity)):
    attr = mesh.attributes.new(name, kind, 'POINT')
    key = {'FLOAT_COLOR': 'color', 'FLOAT_VECTOR': 'vector'}.get(kind, 'value')
    attr.data.foreach_set(key, data.astype(np.float32).ravel())
obj = bpy.data.objects.new('RoomSplat', mesh); bpy.context.scene.collection.objects.link(obj)

# --- Geometry Nodes: an ellipsoid per point -----------------------------------------------------
tree = bpy.data.node_groups.new('SplatInstances', 'GeometryNodeTree')
tree.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
tree.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
n, link = tree.nodes, tree.links.new
gin = n.new('NodeGroupInput'); gout = n.new('NodeGroupOutput')
to_points = n.new('GeometryNodeMeshToPoints')
ico = n.new('GeometryNodeMeshIcoSphere'); ico.inputs['Subdivisions'].default_value = 1; ico.inputs['Radius'].default_value = 1.0
inst = n.new('GeometryNodeInstanceOnPoints')
rot = n.new('GeometryNodeInputNamedAttribute'); rot.data_type = 'QUATERNION'; rot.inputs['Name'].default_value = 'splat_rot'
scl = n.new('GeometryNodeInputNamedAttribute'); scl.data_type = 'FLOAT_VECTOR'; scl.inputs['Name'].default_value = 'splat_scale'
mat_node = n.new('GeometryNodeSetMaterial')
link(gin.outputs[0], to_points.inputs['Mesh'])
link(to_points.outputs['Points'], inst.inputs['Points'])
link(ico.outputs['Mesh'], inst.inputs['Instance'])
link(rot.outputs['Attribute'], inst.inputs['Rotation'])
link(scl.outputs['Attribute'], inst.inputs['Scale'])
link(inst.outputs['Instances'], mat_node.inputs['Geometry'])
link(mat_node.outputs['Geometry'], gout.inputs[0])
mod = obj.modifiers.new('Splat', 'NODES'); mod.node_group = tree

# Material: unlit splat colour, transparent by opacity (attributes read from the instancer).
mat = bpy.data.materials.new('RoomSplat'); mat.use_nodes = True
nt = mat.node_tree; nt.nodes.clear()
col = nt.nodes.new('ShaderNodeAttribute'); col.attribute_type = 'INSTANCER'; col.attribute_name = 'splat_color'
alpha = nt.nodes.new('ShaderNodeAttribute'); alpha.attribute_type = 'INSTANCER'; alpha.attribute_name = 'splat_alpha'
emit = nt.nodes.new('ShaderNodeEmission'); clear = nt.nodes.new('ShaderNodeBsdfTransparent')
mix = nt.nodes.new('ShaderNodeMixShader'); outn = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(col.outputs['Color'], emit.inputs['Color'])
# Gaussian-like falloff: full opacity facing the viewer, fading towards the silhouette.
facing = nt.nodes.new('ShaderNodeLayerWeight'); falloff = nt.nodes.new('ShaderNodeMath'); falloff.operation = 'POWER'
fade = nt.nodes.new('ShaderNodeMath'); fade.operation = 'SUBTRACT'; fade.inputs[0].default_value = 1.0
weight = nt.nodes.new('ShaderNodeMath'); weight.operation = 'MULTIPLY'
nt.links.new(facing.outputs['Facing'], fade.inputs[1])            # 1 - facing: 1 at centre, 0 at edge
nt.links.new(fade.outputs[0], falloff.inputs[0]); falloff.inputs[1].default_value = 2.0
nt.links.new(falloff.outputs[0], weight.inputs[0]); nt.links.new(alpha.outputs['Fac'], weight.inputs[1])
nt.links.new(weight.outputs[0], mix.inputs['Fac'])
nt.links.new(clear.outputs[0], mix.inputs[1]); nt.links.new(emit.outputs[0], mix.inputs[2])
nt.links.new(mix.outputs[0], outn.inputs['Surface'])
for attr, value in (('surface_render_method', 'BLENDED'), ('blend_method', 'BLEND')):
    try:
        setattr(mat, attr, value); break
    except (AttributeError, TypeError):
        pass
mat_node.inputs['Material'].default_value = mat

# Room view camera + avatar placement from live/splat/render_room.py, when available.
view_path = align.parent / 'view.json'
if view_path.exists():
    from mathutils import Matrix
    view = json.loads(view_path.read_text())
    cam = bpy.data.objects.new('RoomView', bpy.data.cameras.new('RoomView')); bpy.context.scene.collection.objects.link(cam)
    cam.matrix_world = Matrix(view['matrix_world'])
    cam.data.sensor_fit = 'HORIZONTAL'; cam.data.sensor_width = view['sensor_width']; cam.data.lens = view['lens']
    cam.data.shift_x, cam.data.shift_y = view['shift_x'], view['shift_y']
    scene = bpy.context.scene; scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = view['width'], view['height']
    arm = next((o for o in bpy.data.objects if o.type == 'ARMATURE' and 'rgbmocap_humanoid' in o), None)
    if arm is not None:
        arm.location = view['avatar_location']; arm.rotation_mode = 'XYZ'; arm.rotation_euler = (0, 0, view['avatar_yaw'])
bpy.ops.wm.save_as_mainfile(filepath=str(out))
lo, hi = xyz.min(0), xyz.max(0)
print('SPLAT IMPORTED', len(xyz), 'gaussians of', count, '| room bounds', np.round(lo, 1).tolist(), np.round(hi, 1).tolist(), '->', out)
