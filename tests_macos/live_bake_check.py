"""Blender headless: the app's bake_take command keys a trimmed take onto the avatar."""
import json, sys, importlib.util
from pathlib import Path
import bpy
take = Path(sys.argv[sys.argv.index('--') + 1])
spec = importlib.util.spec_from_file_location('rgbmocap_live', Path(__file__).resolve().parents[1] / 'live/blender/rgbmocap_live.py')
live = importlib.util.module_from_spec(spec); spec.loader.exec_module(live)
arm = bpy.data.objects['RGBMoCap_Avatar']
live.RECEIVER.avatar = live.Avatar(arm)
meta = json.loads((take.parent / 'take.json').read_text())
live.RECEIVER.command({'cmd': 'bake_take', 'path': str(take), 'in': meta.get('in', 0), 'out': meta.get('out')})
act = arm.animation_data.action
curves = [fc for layer in act.layers for strip in layer.strips for bag in strip.channelbags for fc in bag.fcurves] if hasattr(act, 'layers') else list(act.fcurves)
keys = max(len(fc.keyframe_points) for fc in curves)
print('BAKE OK', act.name, 'fcurves', len(curves), 'keys per curve', keys, 'frame range', tuple(act.frame_range), 'trim', meta.get('in'), meta.get('out'))
