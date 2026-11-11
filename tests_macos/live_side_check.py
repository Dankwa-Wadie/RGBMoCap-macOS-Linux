"""Blender headless: the avatar's left/right hands must land on the person's left/right side."""
import json, sys, importlib.util
from pathlib import Path
import bpy
from mathutils import Vector
folder = Path(sys.argv[sys.argv.index('--') + 1])
spec = importlib.util.spec_from_file_location('rgbmocap_live', Path(__file__).resolve().parents[1] / 'live/blender/rgbmocap_live.py')
live = importlib.util.module_from_spec(spec); spec.loader.exec_module(live)
arm = bpy.data.objects['RGBMoCap_Avatar']; av = live.Avatar(arm)
agree = total = 0
for f in sorted(folder.glob('camera_*.json')):
    p = json.loads(f.read_text())
    if 'pose' not in p: continue
    for _ in range(15): av.apply(p)       # let easing settle
    bpy.context.view_layer.update()
    hand = lambda h: arm.matrix_world @ av.bones[h].head
    avatar_lr = hand('leftHand') - hand('rightHand')
    person_lr = Vector(p['pose'][15]) - Vector(p['pose'][16])
    total += 1; agree += avatar_lr.normalized().dot(person_lr.normalized()) > 0.3
    if total <= 4:
        print(f.name, 'avatar L-R hand', [round(x, 2) for x in avatar_lr], 'person L-R wrist', [round(x, 2) for x in person_lr])
print(f'SIDE CHECK: {agree}/{total} frames agree')
