"""Photoreal room background from a trained splat, plus the matching Blender camera.

    .venv-splat/bin/python live/splat/render_room.py ROOM_DIR [--view INDEX] [--distance 1.8]

ROOM_DIR contains undistorted/ (COLMAP), room.ckpt (live/splat/train_room.py) and
align.json (live/splat/align.py). The view is one of the video's own camera poses
(default: the most open one), so it is always inside the room. Writes:

* ROOM_DIR/background.png  the splat rendered by msplat from that camera
* ROOM_DIR/view.json       Blender camera (matrix_world, lens, resolution) and where the
                           avatar stands: on the floor DISTANCE metres ahead, facing the camera
"""
import argparse, json, struct
from pathlib import Path
import numpy as np
import cv2
import msplat

ap = argparse.ArgumentParser()
ap.add_argument('room', type=Path); ap.add_argument('--view', type=int)
ap.add_argument('--distance', type=float, default=1.8)
args = ap.parse_args()
align = json.loads((args.room / 'align.json').read_text())
M = np.array(align['matrix']); s = np.cbrt(np.linalg.det(M[:3, :3])); Rm = M[:3, :3] / s

# Intrinsics of the undistorted PINHOLE camera (cameras.bin).
raw = (args.room / 'undistorted/sparse/0/cameras.bin').read_bytes()
_, model, width, height = struct.unpack_from('<iiQQ', raw, 8)
fx, fy, cx, cy = struct.unpack_from('<4d', raw, 8 + 24)
assert model == 1, 'expected a PINHOLE camera'

dataset = msplat.load_dataset(str(args.room / 'undistorted'))
config = msplat.TrainingConfig(); config.keep_crs = True
trainer = msplat.GaussianTrainer(dataset, config)
trainer.load_checkpoint(str(args.room / 'room.ckpt'))

cams = align['cameras']
# msplat re-centres/rescales its scene internally; recover that similarity transform from the
# camera centres (same image order as COLMAP's name-sorted cameras) and fold it into the poses.
raw_poses = [np.asarray(dataset.camera_pose(i), np.float64).reshape(4, 4) for i in range(dataset.num_train)]
src = np.array([p[:3, 3] for p in raw_poses]); dst = np.array([c['position'] for c in cams])
assert len(src) == len(dst), f'{len(src)} msplat cameras vs {len(dst)} COLMAP cameras'
mu_s, mu_d = src.mean(0), dst.mean(0); A, B = src - mu_s, dst - mu_d
U, S, Vt = np.linalg.svd(B.T @ A); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
Rsim = U @ D @ Vt; ssim = (S * np.diag(D)).sum() / (A ** 2).sum(); tsim = mu_d - ssim * Rsim @ mu_s
fit = np.linalg.norm(src @ (ssim * Rsim).T + tsim - dst, axis=1)
print('msplat->room fit: scale %.3f, residual median %.3f m, max %.3f m' % (ssim, np.median(fit), fit.max()))


def to_blender(pose):
    out = np.eye(4); out[:3, :3] = Rsim @ pose[:3, :3]; out[:3, 3] = ssim * Rsim @ pose[:3, 3] + tsim
    return out


poses = [to_blender(p) for p in raw_poses]

# Dense obstacle map from the splat itself (COLMAP's sparse points miss smooth surfaces such as
# cabinet fronts): solid Gaussians above the floor, in the Blender room frame.
ply = (args.room / 'room.ply').read_bytes(); end = ply.index(b'end_header\n') + 11
header = ply[:end].decode().splitlines()
count = next(int(l.split()[2]) for l in header if l.startswith('element vertex'))
fields = [(l.split()[2], '<f4') for l in header if l.startswith('property float')]
g = np.frombuffer(ply, dtype=np.dtype(fields), count=count, offset=end)
solid = 1 / (1 + np.exp(-g['opacity'])) > 0.5
pts = np.stack([g['x'], g['y'], g['z']], 1)[solid].astype(np.float64)
pts = pts @ (ssim * Rsim).T + tsim
pts = pts[pts[:, 2] > 0.15]


def free_distance(pose, cone_deg=12, near=0.3, far=3.5):
    pos = pose[:3, 3]; fwd = -pose[:3, 2].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd)
    rel = pts - pos; dist = rel @ fwd; lateral = np.linalg.norm(rel[:, :2] - np.outer(dist, fwd[:2]), axis=1)
    hit = dist[(dist > near) & (dist < far) & (lateral < np.tan(np.radians(cone_deg)) * dist + 0.15)]
    return float(np.percentile(hit, 2)) if len(hit) > 20 else far


free = [free_distance(p) for p in poses]
if args.view is None:
    best = int(np.argmax(free))
    args.view = int(np.argmin([np.linalg.norm(np.array(c['position']) - poses[best][:3, 3]) for c in cams]))
want = args.view
target = np.array(cams[want]['position'])
index = int(np.argmin([np.linalg.norm(p[:3, 3] - target) for p in poses]))
mismatch = float(np.linalg.norm(poses[index][:3, 3] - target))
assert mismatch < 0.05, f'camera order mismatch ({mismatch:.3f} m)'

image = np.clip(np.asarray(trainer.render(index)), 0, 1)
cv2.imwrite(str(args.room / 'background.png'), cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
cam = poses[index]
forward = -cam[:3, 2].copy(); forward[2] = 0; forward /= np.linalg.norm(forward)
open_m = free[index]
spot = cam[:3, 3] + forward * max(0.8, min(args.distance, open_m - 0.5)); spot[2] = 0.0
# The avatar's front is -Y; rotate it about Z so it faces back along -forward (towards the camera).
yaw = float(np.arctan2(-forward[0], forward[1]))
sensor = 36.0
view = {'name': cams[want]['name'], 'matrix_world': cam.tolist(), 'width': int(width), 'height': int(height),
        'sensor_width': sensor, 'lens': fx * sensor / width,
        'shift_x': (width / 2 - cx) / width, 'shift_y': (cy - height / 2) / width,
        'avatar_location': spot.tolist(), 'avatar_yaw': yaw, 'render_shape': list(image.shape)}
(args.room / 'view.json').write_text(json.dumps(view, indent=1))
print('ROOM VIEW', view['name'], f'{width}x{height}', 'lens %.1fmm' % view['lens'], 'open %.2f m' % open_m,
      'avatar at', np.round(spot, 2).tolist(), 'distance %.2f m' % np.linalg.norm((spot - cam[:3, 3])[:2]))
