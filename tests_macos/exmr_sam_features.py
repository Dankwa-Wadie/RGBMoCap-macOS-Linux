"""SAM 3D Body features on MPS for the first N tracked frames, through ExMR's own wrapper."""
import sys, time, tempfile, os
from pathlib import Path
from mocap.torch_mps import install_on_import
install_on_import()
import numpy as np, torch
from mocap.exmr_backend import activate

project, n = Path(sys.argv[1]), int(sys.argv[2])
root, weights = activate()
from exmr.preprocessing.body_features import SamBodyFeatures
with np.load(project / 'tracking.npz') as t:
    boxes = torch.from_numpy(t['boxes'][:n].copy()).float(); k = torch.from_numpy(t['K'].copy()).float()
k = k[None].repeat(n, 1, 1) if k.ndim == 2 else k[:n]
frames = sorted((project / 'exmr_frames').glob('*.jpg'))[:n]
work = Path(tempfile.mkdtemp())
for f in frames: os.symlink(f.resolve(), work / f.name)
t0 = time.monotonic()
model = SamBodyFeatures(weights / 'sam-3d-body-dinov3', batch_size=8, num_workers=0)
print('loaded in', round(time.monotonic() - t0, 1), 's', flush=True)
t1 = time.monotonic(); out = model(work, boxes, k)
print('features', tuple(out.shape), out.dtype, 'finite', bool(torch.isfinite(out).all()),
      'std', float(out.std()), f'{n/(time.monotonic()-t1):.2f} frames/s', flush=True)
torch.save(out, project / 'mps_sam_features_test.pt')
