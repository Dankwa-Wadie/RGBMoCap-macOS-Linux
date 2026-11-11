"""Blender headless: apply captured packets to the avatar and render front views (test)."""
import json, sys, importlib.util
from pathlib import Path
import bpy
from mathutils import Vector
args = sys.argv[sys.argv.index('--') + 1:]
dump, out_dir = Path(args[0]), Path(args[1]); frames = [int(f) for f in args[2].split(',')]
spec = importlib.util.spec_from_file_location('rgbmocap_live', Path(__file__).resolve().parents[1] / 'live/blender/rgbmocap_live.py')
live = importlib.util.module_from_spec(spec); spec.loader.exec_module(live)
arm = bpy.data.objects['RGBMoCap_Avatar']; avatar = live.Avatar(arm)
rows = {r['frame']: r for r in map(json.loads, open(dump))}
scene = bpy.context.scene
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); scene.collection.objects.link(cam); scene.camera = cam
cam.location = (0, -4.2, 1.0); cam.rotation_euler = (1.5708, 0, 0); cam.data.lens = 45
light = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN')); light.rotation_euler = (0.6, 0, 0.3); scene.collection.objects.link(light)
scene.render.engine = 'BLENDER_WORKBENCH'; scene.render.resolution_x, scene.render.resolution_y = 480, 640
scene.display.shading.color_type = 'TEXTURE'
face_test = {'jawOpen': 0.8, 'eyeBlinkLeft': 1.0, 'mouthSmileLeft': 0.9, 'mouthSmileRight': 0.9}
for f in frames:
    packet = dict(rows[f]); packet['face'] = face_test
    avatar.apply(packet)
    bpy.context.view_layer.update()
    scene.render.filepath = str(out_dir / f'live_{f:04d}.png'); bpy.ops.render.render(write_still=True)
keys = {kb.name: round(kb.value, 2) for o in avatar.shape_targets for kb in o.data.shape_keys.key_blocks if kb.value > 0.05}
print('ACTIVE SHAPE KEYS', keys)
print('RENDERED', [f for f in frames])
