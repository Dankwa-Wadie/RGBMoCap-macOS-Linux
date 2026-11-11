"""Metric, Z-up alignment for a COLMAP room reconstruction (for placing the avatar in it).

    python live/splat/align.py SPARSE_DIR OUT.json [--eye-height 1.45]

* Up: the average image "up" of the phone cameras (frames were upright portrait).
* Floor: a low percentile of the sparse points' heights along that up axis.
* Scale: the median camera height above the floor becomes --eye-height metres
  (COLMAP's own scale is arbitrary).
* Origin: the floor point under the middle of the camera path. +Y points the way
  the first frame looked, so the room the phone saw lies behind an avatar facing -Y.
Writes a 4x4 row-major matrix taking COLMAP world coordinates to Blender.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pycolmap

ap = argparse.ArgumentParser()
ap.add_argument('sparse', type=Path); ap.add_argument('out', type=Path)
ap.add_argument('--eye-height', type=float, default=1.45)
args = ap.parse_args()
rec = pycolmap.Reconstruction(str(args.sparse))
images = sorted(rec.images.values(), key=lambda im: im.name)
centers, ups, forwards = [], [], []
for im in images:
    pose = im.cam_from_world() if callable(getattr(im, 'cam_from_world', None)) else im.cam_from_world
    R = pose.rotation.matrix(); t = pose.translation
    centers.append(-R.T @ t); ups.append(-R.T @ np.array([0, 1, 0])); forwards.append(R.T @ np.array([0, 0, 1]))
centers, up = np.array(centers), np.mean(ups, 0)
up /= np.linalg.norm(up)
points = np.array([p.xyz for p in rec.points3D.values()])
heights = points @ up
floor = np.percentile(heights, 3)
cam_heights = centers @ up - floor
scale = args.eye_height / np.median(cam_heights)
forward = forwards[0] - (forwards[0] @ up) * up; forward /= np.linalg.norm(forward)
x_axis = np.cross(forward, up)
R = np.stack([x_axis, forward, up])          # rows: Blender X, Y, Z in COLMAP coordinates
middle = centers.mean(0); origin = middle - ((middle @ up) - floor) * up
M = np.eye(4); M[:3, :3] = scale * R; M[:3, 3] = -scale * R @ origin
room = (points - origin) @ R.T * scale
# Per camera (Blender frame): position, level forward direction, and how far the view is open
# (no sparse points within a 10 degree cone, 0.3-3 m ahead) - a good spot to stand an avatar.
cams = []
for im, c, f in zip(images, centers, forwards):
    pos = (c - origin) @ R.T * scale; fwd = R @ f; fwd[2] = 0; fwd /= np.linalg.norm(fwd)
    rel = room - pos; dist = rel @ fwd; lateral = np.linalg.norm(rel - np.outer(dist, fwd), axis=1)
    blocked = dist[(dist > 0.3) & (dist < 3.0) & (lateral < np.tan(np.radians(10)) * dist) & (room[:, 2] > 0.2)]
    cams.append({'name': im.name, 'position': pos.tolist(), 'forward': fwd.tolist(),
                 'free_m': float(blocked.min()) if len(blocked) else 3.0})
best = max(range(len(cams)), key=lambda i: cams[i]['free_m'])
args.out.write_text(json.dumps({'matrix': M.tolist(), 'scale': scale, 'images': len(images), 'points': len(points),
                                'camera_height_median_colmap': float(np.median(cam_heights)),
                                'best_view': best, 'cameras': cams}, indent=1))
print('best view', cams[best]['name'], 'open for %.1f m' % cams[best]['free_m'])
print('ALIGNED | scale %.3f | room extent (m) X %.1f Y %.1f Z %.1f | cameras %d' % (
    scale, *(np.percentile(room, 95, 0) - np.percentile(room, 5, 0)), len(images)))
