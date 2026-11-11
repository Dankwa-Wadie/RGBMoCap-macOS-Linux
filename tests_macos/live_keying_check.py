"""Blender headless: live_record_start/stop keys received packets in real time order."""
import json, sys, importlib.util
from pathlib import Path
import bpy
take = Path(sys.argv[sys.argv.index('--') + 1])
spec = importlib.util.spec_from_file_location('rgbmocap_live', Path(__file__).resolve().parents[1] / 'live/blender/rgbmocap_live.py')
live = importlib.util.module_from_spec(spec); spec.loader.exec_module(live)
r = live.RECEIVER; r.avatar = live.Avatar(bpy.data.objects['RGBMoCap_Avatar'])
r.command({'cmd': 'live_record_start'})
for line in take.open():
    p = json.loads(line); r.avatar.apply(p); r.key_live(p)   # what tick() does per packet
r.command({'cmd': 'live_record_stop'})
act = r.avatar.arm.animation_data.action
print('LIVE KEYS OK', act.name, 'frame range', tuple(act.frame_range), 'scene end', bpy.context.scene.frame_end,
      'shape keys keyed', len(r.driven_shape_keys()))
