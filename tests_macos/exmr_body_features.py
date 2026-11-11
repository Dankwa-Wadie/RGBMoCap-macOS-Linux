"""Run ExMR's SAM 3D Body feature phase on MPS via the real backend (test harness).

The body phase does not use MANO/SMPL-X; the stage's upfront model check is
narrowed to the files this phase loads.
"""
import json, sys, time
from pathlib import Path
from mocap.torch_mps import install_on_import
install_on_import()
from mocap import models, exmr_backend

project = Path(sys.argv[1])
original = models.missing
models.missing = lambda stage, tool=None: [m for m in original(stage, tool) if m.key not in ('mano_right', 'smplx')]
def emit(kind, **value): print(kind, json.dumps(value)[:200], flush=True)
t = time.monotonic()
exmr_backend.run_stage(project, 'exmr_features', 'auto', emit, feature_phase='body')
print('BODY-FEATURES-DONE', round(time.monotonic() - t, 1), 's', flush=True)
