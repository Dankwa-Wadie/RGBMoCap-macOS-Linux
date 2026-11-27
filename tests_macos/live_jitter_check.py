# Headless Blender: replay a packet dump (capture.py --dump REC/packets.jsonl); report bones turning > 60 deg in one frame.
#   Blender -b avatar.blend --python tests_macos/live_jitter_check.py -- REC_DIR
import json, sys, importlib.util, collections
from pathlib import Path
import bpy
argv = sys.argv[sys.argv.index('--') + 1:]; rec = Path(argv[0])
root = Path.cwd()
spec = importlib.util.spec_from_file_location('l', root / 'live/blender/rgbmocap_live.py'); l = importlib.util.module_from_spec(spec); spec.loader.exec_module(l)
sys.path.insert(0, str(root / 'live')); from track_modes import apply_mode
MODE = argv[1] if len(argv) > 1 else 'auto'
arm = next(o for o in bpy.data.objects if 'rgbmocap_humanoid' in o); av = l.Avatar(arm)
prev = {}; jumps = collections.defaultdict(list); rows = []
for n, line in enumerate((rec / 'packets.jsonl').open()):
    p = apply_mode(json.loads(line), MODE); av.apply(p)
    cur = {h: pb.rotation_quaternion.copy() for h, pb in av.bones.items()}  # armature-space orientation
    for h, q in cur.items():
        if h in prev:
            d = prev[h].rotation_difference(q).angle * 57.3; d = min(d, 360 - d)
            if d > 60: jumps[h].append((n, round(d)))
    prev = cur
for h, js in sorted(jumps.items(), key=lambda kv: -len(kv[1])):
    print('JUMP', h, len(js), js[:8])
print('FRAMES', n + 1)
