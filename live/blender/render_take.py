"""Blender (headless): render a recorded live take on the avatar as numbered PNG frames.

    Blender -b AVATAR.blend --python live/blender/render_take.py -- \
        TAKE.jsonl FIRST LAST FPS OUT_DIR WIDTH HEIGHT ENGINE

Output frame k shows the take at time t(FIRST) + k / FPS, so the video runs in
real time whatever rate the camera delivered. ENGINE is "fast" (Workbench) or
"quality" (EEVEE). Prints "RENDER k/N" progress lines.
"""
import bisect, importlib.util, json, sys
from pathlib import Path
import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index('--') + 1:]
take, first, last, fps, out_dir, width, height, engine = (
    Path(args[0]), int(args[1]), int(args[2]), float(args[3]), Path(args[4]), int(args[5]), int(args[6]), args[7])
here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rgbmocap_live', here / 'rgbmocap_live.py')
live = importlib.util.module_from_spec(spec); spec.loader.exec_module(live)

rows = [json.loads(line) for line in take.open(encoding='utf-8')][first:last + 1]
times = [r['t'] - rows[0]['t'] for r in rows]
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE' and 'rgbmocap_humanoid' in o)
avatar = live.Avatar(arm)
scene = bpy.context.scene

# Engine: read the available identifiers instead of hard-coding them.
engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items]
wanted = [e for e in engines if 'EEVEE' in e] if engine == 'quality' else ['BLENDER_WORKBENCH']
try:
    scene.render.engine = (wanted or ['BLENDER_WORKBENCH'])[0]
except TypeError:
    scene.render.engine = 'BLENDER_WORKBENCH'
if scene.render.engine == 'BLENDER_WORKBENCH':
    scene.display.shading.light = 'STUDIO'
    scene.display.shading.color_type = 'TEXTURE'
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = width, height, 100
scene.render.image_settings.file_format = 'PNG'
scene.render.film_transparent = False

# Lighting and background
world = scene.world or bpy.data.worlds.new('World'); scene.world = world
world.use_nodes = True
background = next((n for n in world.node_tree.nodes if n.type == 'BACKGROUND'), None)
if background:
    background.inputs[0].default_value = (0.16, 0.17, 0.19, 1); background.inputs[1].default_value = 1.0
sun = bpy.data.objects.new('RenderSun', bpy.data.lights.new('RenderSun', 'SUN'))
sun.data.energy = 3.0; sun.rotation_euler = (0.9, 0.0, 0.5); scene.collection.objects.link(sun)

# Camera framing the avatar's rest-pose bounds from the front (avatar faces -Y).
meshes = [o for o in scene.objects if o.type == 'MESH' and any(m.type == 'ARMATURE' and m.object == arm for m in o.modifiers)]
corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
low = Vector(map(min, *[(c.x, c.y, c.z) for c in corners])); high = Vector(map(max, *[(c.x, c.y, c.z) for c in corners]))
height_m = high.z - low.z; center = (low + high) / 2
cam = bpy.data.objects.new('RenderCam', bpy.data.cameras.new('RenderCam')); scene.collection.objects.link(cam)
cam.data.lens = 50; scene.camera = cam
distance = height_m * 1.15 * cam.data.lens / cam.data.sensor_width * max(1.0, height / width) + 0.5
cam.location = (center.x, low.y - distance, center.z)
cam.rotation_euler = (1.5708, 0, 0)

# Warm up easing on the first pose, then render one frame per output time step.
for _ in range(12):
    avatar.apply(rows[0])
count = int(times[-1] * fps) + 1
out_dir.mkdir(parents=True, exist_ok=True)
for k in range(count):
    i = max(0, bisect.bisect_right(times, k / fps) - 1)
    avatar.apply(rows[i])
    bpy.context.view_layer.update()
    scene.render.filepath = str(out_dir / f'{k:05d}.png')
    bpy.ops.render.render(write_still=True)
    print(f'RENDER {k + 1}/{count}', flush=True)
print('RENDER DONE', count, scene.render.engine, flush=True)
