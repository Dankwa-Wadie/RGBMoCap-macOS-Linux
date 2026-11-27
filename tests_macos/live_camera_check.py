"""Blender headless: replay a recording with "Match webcam camera" and lay the avatar, rendered through the
solved camera, over each camera snapshot (capture.py --dump REC/packets.jsonl --snapshots REC).

    Blender -b avatar.blend --python tests_macos/live_camera_check.py -- REC_DIR OUT_DIR [MODE] [HFOV]

Writes OUT_DIR/overlay_<stamp>.png (camera image with the avatar on top) and prints the mean fit error
("CAMERA fit ...", fraction of the picture height). If the match works, the avatar sits on the performer.
Validity is recomputed with live/capture.py's current rules; old recordings get image size from the snapshots.
"""
import json, sys, importlib.util, statistics
from pathlib import Path
import bpy
argv = sys.argv[sys.argv.index('--') + 1:]
rec, out = Path(argv[0]), Path(argv[1]); mode = argv[2] if len(argv) > 2 else 'auto'
hfov = float(argv[3]) if len(argv) > 3 else 70.0
out.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root / 'live'))
spec = importlib.util.spec_from_file_location('l', root / 'live/blender/rgbmocap_live.py'); l = importlib.util.module_from_spec(spec); spec.loader.exec_module(l)
src = (root / 'live/capture.py').read_text(); ns = {}
exec(src[src.index('LEGS = '):src.index('class OneEuro')], ns)
STABLE = int(src.split('STABLE_FRAMES = ')[1].split()[0])
from track_modes import apply_mode, leg_chain
arm = next(o for o in bpy.data.objects if 'rgbmocap_humanoid' in o); av = l.Avatar(arm); match = l.CameraMatch()
scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'; scene.display.shading.color_type = 'MATERIAL'
scene.render.film_transparent = True; scene.render.resolution_x = 640; scene.render.resolution_y = 360
snaps = {}
for f in rec.glob('camera_*.json'):
    snaps[json.loads(f.read_text()).get('frame')] = f.with_suffix('.jpg')
size = None
streak, errors, n = None, [], 0
for line in (rec / 'packets.jsonl').open():
    p = json.loads(line)
    if 'pose' in p and p.get('pose_image'):
        raw = [ns['landmark_valid'](i, v, x, y) for i, (v, (x, y)) in enumerate(zip(p['pose_visibility'], p['pose_image']))]
        streak = [s + 1 if r else 0 for s, r in zip(streak or [0] * len(raw), raw)]
        p['pose_valid'] = leg_chain([s >= STABLE for s in streak])
    if 'image' not in p:
        if size is None:
            import struct
            jpg = next(iter(snaps.values())).read_bytes(); i = 2
            while i < len(jpg):  # JPEG SOF marker -> width, height
                marker, length = jpg[i + 1], struct.unpack('>H', jpg[i + 2:i + 4])[0]
                if marker in (0xC0, 0xC2):
                    size = struct.unpack('>HH', jpg[i + 5:i + 9])[::-1]; break
                i += 2 + length
        p['image'] = [size[0], size[1], hfov]
    p = apply_mode(p, mode)
    av.apply(p)
    if match.update(av, p) is not None and match.error is not None:
        errors.append(match.error)
    if p.get('frame') in snaps and match.shown is not None:
        match.apply(scene, p); bpy.context.view_layer.update()
        stamp = snaps[p['frame']].stem.split('_')[1]
        avatar_png = out / f'avatar_{stamp}.png'
        scene.render.filepath = str(avatar_png); bpy.ops.render.render(write_still=True)
        cam = bpy.data.images.load(str(snaps[p['frame']])); av_img = bpy.data.images.load(str(avatar_png))
        W, H = av_img.size
        cam.scale(W, H)
        import numpy as np
        a = np.array(av_img.pixels[:]).reshape(H, W, 4); c = np.array(cam.pixels[:]).reshape(H, W, 4)
        alpha = a[..., 3:4] * 0.75
        c[..., :3] = c[..., :3] * (1 - alpha) + a[..., :3] * alpha
        cam.pixels[:] = c.ravel(); cam.filepath_raw = str(out / f'overlay_{stamp}.png'); cam.file_format = 'PNG'; cam.save()
        bpy.data.images.remove(cam); bpy.data.images.remove(av_img); n += 1
print('CAMERA fit mean %.3f median %.3f frames %d overlays %d' % (
    statistics.mean(errors) if errors else -1, statistics.median(errors) if errors else -1, len(errors), n))
