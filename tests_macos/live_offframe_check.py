"""Blender headless: off-frame legs must ease to rest while visible arms stay tracked."""
import json, sys, importlib.util
from pathlib import Path
import bpy
dump = Path(sys.argv[sys.argv.index('--') + 1])
spec = importlib.util.spec_from_file_location('rgbmocap_live', Path(__file__).resolve().parents[1] / 'live/blender/rgbmocap_live.py')
live = importlib.util.module_from_spec(spec); spec.loader.exec_module(live)
arm = bpy.data.objects['RGBMoCap_Avatar']; avatar = live.Avatar(arm)
packet = next(r for r in map(json.loads, open(dump)) if r['frame'] == 150)
angle = lambda h: round(avatar.bones[h].rotation_quaternion.angle * 57.3, 1)
legs, arms = ('leftUpperLeg', 'rightLowerLeg'), ('leftUpperArm', 'rightLowerArm')
for _ in range(10):
    avatar.apply(dict(packet, pose_valid=[True] * 33))
print('legs visible   :', {h: angle(h) for h in legs + arms})
hidden = [i < 23 for i in range(33)]  # hips, knees, ankles, feet out of frame
for _ in range(30):
    avatar.apply(dict(packet, pose_valid=hidden))
print('legs off-frame :', {h: angle(h) for h in legs + arms})
assert all(angle(h) < 1 for h in legs), 'legs did not return to rest'
assert all(angle(h) > 10 for h in arms), 'arms stopped tracking'
print('OFFFRAME OK')
