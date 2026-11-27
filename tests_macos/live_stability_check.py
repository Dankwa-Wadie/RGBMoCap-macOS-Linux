"""Blender headless: replay a packet dump through the receiver and score stability per body group.

    Blender -b avatar.blend --python tests_macos/live_stability_check.py -- PACKETS.jsonl [MODE]

jitter  = mean deg a bone sits off the midpoint of its previous and next frame (0 for smooth motion)
flicker = tracked<->untracked switches per 100 frames (each one eases the bone in or out)
lag     = frames the applied rotation trails the raw tracked target (cost of smoothing)
Validity is recomputed from raw visibility with live/capture.py's current rules.
"""
import json, sys, importlib.util, statistics
from pathlib import Path
import bpy
argv = sys.argv[sys.argv.index('--') + 1:]
packets, mode = Path(argv[0]), (argv[1] if len(argv) > 1 else 'auto')
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root / 'live'))
import os; spec = importlib.util.spec_from_file_location('l', os.environ.get('LIVE_RECEIVER') or root / 'live/blender/rgbmocap_live.py'); l = importlib.util.module_from_spec(spec); spec.loader.exec_module(l)
src = (root / 'live/capture.py').read_text(); ns = {}
exec(src[src.index('LEGS = '):src.index('class OneEuro')], ns)
STABLE = int(src.split('STABLE_FRAMES = ')[1].split()[0])
from track_modes import apply_mode, leg_chain
arm = next(o for o in bpy.data.objects if 'rgbmocap_humanoid' in o); av = l.Avatar(arm)
GROUPS = {'torso': ('hips', 'spine', 'chest', 'upperChest', 'neck'), 'head': ('head',),
          'arms': ('UpperArm', 'LowerArm', 'Shoulder'), 'hands': ('Hand',),
          'fingers': ('Thumb', 'Index', 'Middle', 'Ring', 'Little'), 'legs': ('UpperLeg', 'LowerLeg', 'Foot', 'Toes')}
group = lambda h: next((g for g, keys in GROUPS.items() if any(k in h for k in keys)), None)
track, tracked, raw_t, streak = {h: [] for h in av.bones}, {h: [] for h in av.bones}, {h: [] for h in av.bones}, None
validity = l.Avatar.__dict__.get('stable_validity')
for line in packets.open():
    p = json.loads(line)
    if 'pose' in p and p.get('pose_image'):
        raw = [ns['landmark_valid'](i, v, x, y) for i, (v, (x, y)) in enumerate(zip(p['pose_visibility'], p['pose_image']))]
        streak = [s + 1 if r else 0 for s, r in zip(streak or [0] * len(raw), raw)]
        p['pose_valid'] = leg_chain([s >= STABLE for s in streak])
    av.apply(apply_mode(p, mode))
    for h, pb in av.bones.items():
        track[h].append(pb.rotation_quaternion.copy()); tracked[h].append(av.weight.get(pb.name, 0) > 0.5)
        raw_t[h].append(av.last_basis.get(pb.name, pb.rotation_quaternion).copy())  # unfiltered tracked target
def lag(h):
    # frames the applied rotation trails the raw target: best-matching shift of the angle-speed signals
    a = [x.rotation_difference(y).angle for x, y in zip(raw_t[h], raw_t[h][1:])]
    b = [x.rotation_difference(y).angle for x, y in zip(track[h], track[h][1:])]
    if sum(a) < 1e-3: return None
    score = lambda k: sum(x * y for x, y in zip(a, b[k:]))
    return max(range(0, 6), key=score)
jit, flick, lags = {}, {}, {}
for h, qs in track.items():
    g = group(h)
    if not g: continue
    d = [qs[i - 1].slerp(qs[i + 1], 0.5).rotation_difference(qs[i]).angle * 57.3 for i in range(1, len(qs) - 1)]
    d = [min(x, 360 - x) for x in d]
    t = tracked[h]; sw = sum(1 for a, b in zip(t, t[1:]) if a != b)
    jit.setdefault(g, []).append(statistics.mean(d)); flick.setdefault(g, []).append(100 * sw / len(t))
    k = lag(h)
    if k is not None: lags.setdefault(g, []).append(k)
for g in GROUPS:
    if g in jit: print(f'STABILITY {g:8s} jitter {statistics.mean(jit[g]):5.2f} deg   flicker {statistics.mean(flick[g]):5.2f} /100f'
                       f'   lag {statistics.mean(lags.get(g, [0])):.1f} f')
