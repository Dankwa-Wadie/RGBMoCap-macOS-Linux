"""Blender headless: replay recorded live packets through the receiver and render the avatar
at every camera snapshot, for side-by-side comparison with what the camera saw.

    Blender -b avatar.blend --python tests_macos/live_replay_render.py -- REC_DIR OUT_DIR

Validity is recomputed with live/capture.py's current rules (so old recordings test new rules).
"""
import json, sys, importlib.util
from pathlib import Path
import bpy
from mathutils import Vector
argv = sys.argv[sys.argv.index('--') + 1:]
rec, out = Path(argv[0]), Path(argv[1]); mode = argv[2] if len(argv) > 2 else 'full'; out.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rgbmocap_live', root / 'live/blender/rgbmocap_live.py')
live = importlib.util.module_from_spec(spec); spec.loader.exec_module(live)
src = (root / 'live/capture.py').read_text()
ns = {}; exec(src[src.index('LEGS = '):src.index('class OneEuro')], ns)   # landmark_valid + LEGS
STABLE = int(src.split('STABLE_FRAMES = ')[1].split()[0])
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE' and 'rgbmocap_humanoid' in o)
av = live.Avatar(arm); scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'; scene.display.shading.color_type = 'TEXTURE'
scene.render.resolution_x, scene.render.resolution_y = 360, 540
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); scene.collection.objects.link(cam); scene.camera = cam
cam.location = (0, -5.5, 0.9); cam.rotation_euler = (1.5708, 0, 0); cam.data.lens = 40
snap_frames = {json.loads(f.read_text()).get('frame'): f.stem.split('_')[1] for f in rec.glob('camera_*.json')}
streak = None; rendered = 0; log = []
for line in (rec / 'packets.jsonl').open():
    p = json.loads(line)
    if 'pose' in p:
        img = p.get('pose_image') or [[0.5, 0.5]] * len(p['pose'])
        raw = [ns['landmark_valid'](i, v, x, y) for i, (v, (x, y)) in enumerate(zip(p['pose_visibility'], img))]
        streak = [s + 1 if r else 0 for s, r in zip(streak or [0] * len(raw), raw)]
        p['pose_valid'] = [s >= STABLE for s in streak]
    sys.path.insert(0, str(root / 'live')); from track_modes import apply_mode
    p = apply_mode(p, mode)
    av.apply(p)
    if p['frame'] in snap_frames:
        bpy.context.view_layer.update()
        stamp = snap_frames[p['frame']]
        scene.render.filepath = str(out / f'avatar_{stamp}.png'); bpy.ops.render.render(write_still=True); rendered += 1
        log.append(dict(stamp=stamp, lying=av.lying, valid=[i for i, v in enumerate(p.get('pose_valid') or []) if v],
                        hands=sorted(p.get('hands') or {}), bones={h: round(pb.rotation_quaternion.angle * 57.3)
                        for h, pb in av.bones.items() if pb.rotation_quaternion.angle * 57.3 > 15}))
(out / 'log.json').write_text(json.dumps(log, indent=0))
print('REPLAY RENDERED', rendered)
