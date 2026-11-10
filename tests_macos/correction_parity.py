"""Run the native correction exactly as mocap.correction does, on fixed synthetic motion.

usage: python correction_parity.py MODULE.so OUT.npz
Run once with the arm64 (NEON via sse2neon) build and once with an x86_64 (SSE)
build under Rosetta, then compare the two outputs.
"""
import importlib.util, sys
import numpy as np

PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
spec = importlib.util.spec_from_file_location('_motion_correction', sys.argv[1])
native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)

rng = np.random.default_rng(1234)
# SMPL-like 22 joint rest positions (metres, Y up).
rest = np.array([[0,.95,0],[.06,.86,0],[-.06,.86,0],[0,1.05,0],[.1,.48,0],[-.1,.48,0],[0,1.18,0],
                 [.1,.07,-.03],[-.1,.07,-.03],[0,1.24,0],[.1,.02,.1],[-.1,.02,.1],[0,1.45,0],[.08,1.36,0],
                 [-.08,1.36,0],[0,1.56,0],[.18,1.4,0],[-.18,1.4,0],[.44,1.4,0],[-.44,1.4,0],[.68,1.4,0],[-.68,1.4,0]],float)
offsets = rest.copy()
for j in range(1, 22): offsets[j] -= rest[PARENTS[j]]
n = 120
t = np.arange(n) / 30
root = np.stack([.3 * np.sin(t), .95 + .02 * np.sin(4 * t), .8 * t], 1).astype(np.float32)
angles = rng.normal(scale=.15, size=(n, 22, 3)).cumsum(0) * .05
half = np.linalg.norm(angles, axis=-1, keepdims=True) / 2
axis = angles / np.maximum(2 * half, 1e-9)
quat = np.concatenate([np.cos(half), axis * np.sin(half)], -1).astype(np.float32)  # w,x,y,z
contacts = np.zeros((n, 4), np.float32)
contacts[10:50, 0] = 1; contacts[60:100, 2] = 1
targets = np.zeros((n, 2, 3), np.float32)
targets[:, 0] = [.1, .07, .4]; targets[:, 1] = [-.1, .07, 1.2]
axes = np.ascontiguousarray([[1., 0, 0], [1., 0, 0]], np.float64).reshape(-1)
r = root.copy(); q = quat.copy(); zero = np.zeros(n, np.float32)
report = native.correct_motion(r.reshape(-1), q.reshape(-1), r.copy().reshape(-1), q.copy().reshape(-1),
    zero, zero, zero, zero, zero, zero, contacts.reshape(-1), PARENTS, offsets.tolist(),
    [[0., 0., 0., 1.] for _ in PARENTS], 20, 21, 7, 8, .5, .04, False, targets.reshape(-1), axes)
np.savez(sys.argv[2], root=r, quat=q, root_in=root, quat_in=quat)
print('changed root', float(np.abs(r - root).max()), 'changed quat', float(np.abs(q - quat).max()))
print('report keys', sorted(report)[:8])
