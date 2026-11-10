"""CPU skinning must equal a literal transliteration of each GLSL compute kernel."""
import numpy as np
from mocap import cpu_skin

rng = np.random.default_rng(7)
V = 37


def row(pose, o, v):  # dot(vec4(pose[o..o+3]), v)
    return sum(pose[o + k] * v[k] for k in range(4))


def palette(count):
    m = rng.normal(size=(count, 4, 4)).astype(np.float32); m[:, 3] = [0, 0, 0, 1]
    return m.reshape(-1)


def smplx_packed():
    n = 3 * V
    w = rng.random((V, 55)).astype(np.float32); w[w < .7] = 0; w /= np.maximum(w.sum(1, keepdims=True), 1e-6)
    return np.concatenate([rng.normal(size=507 * n).astype(np.float32) * .01, w.ravel()])


def smplx_pose():
    return np.concatenate([palette(55), rng.normal(size=486).astype(np.float32), rng.normal(size=20).astype(np.float32), rng.normal(size=3).astype(np.float32)])


def glsl_mesh(model, pose):
    n = V * 3; out = np.zeros((V, 3))
    for i in range(V):
        p = np.array(model[3*i:3*i+3], float)
        for k in range(20):
            o = n + k*n + 3*i; p += pose[1366+k] * model[o:o+3]
        for k in range(486):
            o = n*21 + k*n + 3*i; p += pose[880+k] * model[o:o+3]
        v = [*p, 1]; r = np.zeros(3)
        for j in range(55):
            w = model[n*507 + i*55 + j]
            if w == 0: continue
            o = j*16; r += w * np.array([row(pose, o, v), row(pose, o+4, v), row(pose, o+8, v)])
        out[i] = r + pose[1386:1389]
    return out


def test_smplx():
    model, pose = smplx_packed(), smplx_pose()
    got = cpu_skin.skin_smplx(cpu_skin.SmplxModel(model, V), pose)
    np.testing.assert_allclose(got, glsl_mesh(model, pose), rtol=2e-4, atol=2e-4)


def test_bound():
    model, pose = smplx_packed(), smplx_pose(); n = 3 * V
    binding = rng.normal(size=n + 55 * 9).astype(np.float32)
    ref = np.zeros((V, 3))
    for i in range(V):
        delta = np.zeros(3)
        for k in range(486):
            o = n*21 + k*n + 3*i; delta += pose[880+k] * model[o:o+3]
        t = np.zeros(3)
        for j in range(55):
            w = model[n*507 + i*55 + j]
            if w == 0: continue
            o = n + j*9; t += w * np.array([binding[o:o+3] @ delta, binding[o+3:o+6] @ delta, binding[o+6:o+9] @ delta])
        p = [*(binding[3*i:3*i+3] + t), 1]; r = np.zeros(3)
        for j in range(55):
            w = model[n*507 + i*55 + j]
            if w == 0: continue
            o = j*16; r += w * np.array([row(pose, o, p), row(pose, o+4, p), row(pose, o+8, p)])
        ref[i] = r
    got = cpu_skin.skin_bound(cpu_skin.SmplxModel(model, V), pose, binding)
    np.testing.assert_allclose(got, ref, rtol=2e-4, atol=2e-4)


def test_rigid():
    bones = 5
    packed = np.concatenate([rng.normal(size=(V, 6)), rng.integers(0, bones, (V, 1))], 1).astype(np.float32).ravel()
    frame = palette(bones)
    pos, nor = cpu_skin.skin_rigid(packed, V, frame, bones)
    for i in range(V):
        p = i*7; b = int(packed[p+6])*16
        v = [*packed[p:p+3], 1]; nn = [*packed[p+3:p+6], 0]
        np.testing.assert_allclose(pos[i], [row(frame, b, v), row(frame, b+4, v), row(frame, b+8, v)], rtol=1e-4, atol=1e-4)
        e = np.array([row(frame, b, nn), row(frame, b+4, nn), row(frame, b+8, nn)])
        np.testing.assert_allclose(nor[i], e / np.linalg.norm(e), rtol=1e-4, atol=1e-4)


def test_sparse():
    bones = 6; counts = rng.integers(1, 4, V); starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pairs = np.stack([rng.integers(0, bones, counts.sum()), rng.random(counts.sum())], 1)
    model = np.concatenate([rng.normal(size=3*V), np.stack([starts, counts], 1).ravel(), pairs.ravel()]).astype(np.float32)
    pose = palette(bones)
    got = cpu_skin.skin_sparse(model, V, pose, bones)
    for i in range(V):
        v = [*model[3*i:3*i+3], 1]; r = 3*V + 2*i; a = int(model[r]); c = int(model[r+1]); p = np.zeros(3)
        for k in range(c):
            o = 5*V + 2*(a+k); b = 16*int(model[o]); w = model[o+1]
            p += w * np.array([row(pose, b, v), row(pose, b+4, v), row(pose, b+8, v)])
        np.testing.assert_allclose(got[i], p, rtol=1e-4, atol=1e-4)


def test_normals():
    pos = rng.normal(size=(V, 3)).astype(np.float32)
    counts = rng.integers(0, 4, V); offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
    neighbors = rng.integers(0, V, 2 * counts.sum()).astype(np.uint32)
    got = cpu_skin.NormalTopology(offsets, neighbors, V)(pos)
    for i in range(V):
        n = np.zeros(3)
        for a in range(offsets[i], offsets[i+1]):
            n += np.cross(pos[neighbors[2*a]] - pos[i], pos[neighbors[2*a+1]] - pos[i])
        e = n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else n
        np.testing.assert_allclose(got[i], e, rtol=1e-4, atol=1e-4)
