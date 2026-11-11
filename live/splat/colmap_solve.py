"""Solve camera poses for a room video's frames with COLMAP (pycolmap, CPU).

    python live/splat/colmap_solve.py WORK_DIR      # WORK_DIR/images/*.jpg

Writes WORK_DIR/sparse/0 (the largest model) and an undistorted PINHOLE copy in
WORK_DIR/undistorted (images/ + sparse/), the layout Gaussian-splat trainers read.
"""
import sys, time
from pathlib import Path
import pycolmap

work = Path(sys.argv[1]); images = work / 'images'; db = work / 'database.db'
db.unlink(missing_ok=True)
t = time.monotonic()
reader = pycolmap.ImageReaderOptions(); reader.camera_model = 'OPENCV'
pycolmap.extract_features(db, images, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
print(f'features done {time.monotonic() - t:.0f}s', flush=True)
matching = pycolmap.SequentialMatchingOptions(); matching.overlap = 12; matching.loop_detection = False
pycolmap.match_sequential(db, matching_options=matching)
print(f'matching done {time.monotonic() - t:.0f}s', flush=True)
maps = pycolmap.incremental_mapping(db, images, work / 'sparse')
best = max(maps.values(), key=lambda r: r.num_reg_images())
print(f'mapping done {time.monotonic() - t:.0f}s | models {len(maps)} | registered {best.num_reg_images()}/{len(list(images.glob("*.jpg")))}'
      f' | points {best.num_points3D()} | reproj error {best.compute_mean_reprojection_error():.2f}px', flush=True)
best_dir = work / 'sparse' / 'best'; best_dir.mkdir(parents=True, exist_ok=True); best.write(best_dir)
pycolmap.undistort_images(work / 'undistorted', best_dir, images)
print('UNDISTORTED', work / 'undistorted', flush=True)
