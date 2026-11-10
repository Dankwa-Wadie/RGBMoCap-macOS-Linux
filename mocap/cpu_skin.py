"""CPU equivalents of the GLSL 430 compute skinning shaders.

macOS stops at OpenGL 4.1, which has no compute shaders or shader storage
buffers. These functions reproduce each compute kernel exactly, on the same
packed float32 layouts the renderers upload, and return the interleaved
``{vec4 position; vec4 normal;}`` vertex records the draw shaders consume.
Comments name the kernel each function mirrors.
"""
from __future__ import annotations

import numpy as np

SMPLX_JOINTS = 55
SHAPE_TERMS = 20
POSE_TERMS = 486


def _affine(pose, count):
    """Row-major 4x4 palette -> (count,3,4); GLSL dotted rows 0..2."""
    return np.asarray(pose[:count * 16], np.float32).reshape(count, 4, 4)[:, :3, :]


def _blend(weights, palette):
    """Per-vertex weighted affine (V,3,4) from dense weights (V,J)."""
    return (weights @ palette.reshape(len(palette), 12)).reshape(-1, 3, 4)


def _apply(transform, points):
    return np.einsum('vij,vj->vi', transform[:, :, :3], points) + transform[:, :, 3]


class SmplxModel:
    """Views over mesh_gl's packed model: template, shapes, pose correctives, weights."""

    def __init__(self, packed, vertex_count):
        v = int(vertex_count); n = 3 * v; m = np.asarray(packed, np.float32)
        self.vertex_count = v
        self.template = m[:n].reshape(v, 3)
        self.shapes = m[n:21 * n].reshape(SHAPE_TERMS, n)
        self.correctives = m[21 * n:507 * n].reshape(POSE_TERMS, n)
        self.weights = m[507 * n:507 * n + v * SMPLX_JOINTS].reshape(v, SMPLX_JOINTS)

    def corrective(self, pose):
        return (np.asarray(pose[880:880 + POSE_TERMS], np.float32) @ self.correctives).reshape(-1, 3)


def skin_smplx(model: SmplxModel, pose):
    """mesh_gl.SKIN: shape + pose blendshapes, 55-joint LBS, root translation."""
    pose = np.asarray(pose, np.float32)
    rest = model.template + (pose[1366:1366 + SHAPE_TERMS] @ model.shapes).reshape(-1, 3) + model.corrective(pose)
    return _apply(_blend(model.weights, _affine(pose, SMPLX_JOINTS)), rest) + pose[1386:1389]


def skin_bound(model: SmplxModel, pose, binding):
    """bound_mesh_gl.SKIN: correctives transported into a canonical actor binding."""
    pose = np.asarray(pose, np.float32); binding = np.asarray(binding, np.float32)
    v = model.vertex_count; n = 3 * v
    bound = binding[:n].reshape(v, 3)
    frames = binding[n:n + SMPLX_JOINTS * 9].reshape(SMPLX_JOINTS, 9)
    rotation = (model.weights @ frames).reshape(v, 3, 3)
    rest = bound + np.einsum('vij,vj->vi', rotation, model.corrective(pose))
    return _apply(_blend(model.weights, _affine(pose, SMPLX_JOINTS)), rest)


def skin_rigid(packed, vertex_count, frame, bone_count):
    """rig_gl.SKIN (rawMode=0): one bone per vertex, positions and normals."""
    m = np.asarray(packed, np.float32)[:vertex_count * 7].reshape(vertex_count, 7)
    bones = m[:, 6].astype(np.int64)
    transform = _affine(frame, bone_count)[bones]
    positions = _apply(transform, m[:, :3])
    normals = _normalize(np.einsum('vij,vj->vi', transform[:, :, :3], m[:, 3:6]))
    return positions, normals


def skin_sparse(packed, vertex_count, pose, palette):
    """character_gl.SKIN: variable-length (bone, weight) influence lists."""
    v = int(vertex_count); m = np.asarray(packed, np.float32)
    rest = m[:3 * v].reshape(v, 3)
    rows = m[3 * v:5 * v].reshape(v, 2).astype(np.int64)
    pairs = m[5 * v:].reshape(-1, 2)
    owner = np.repeat(np.arange(v), rows[:, 1])
    index = np.concatenate([np.arange(a, a + c) for a, c in rows]) if len(rows) else np.zeros(0, np.int64)
    bone = pairs[index, 0].astype(np.int64); weight = pairs[index, 1]
    affine = _affine(pose, palette)
    contribution = weight[:, None] * _apply(affine[bone], rest[owner])
    out = np.zeros((v, 3), np.float32)
    for axis in range(3):
        out[:, axis] = np.bincount(owner, contribution[:, axis], minlength=v)
    return out


class NormalTopology:
    """mesh_gl.NORMALS: area-weighted fan normals from (offsets, neighbor pairs)."""

    def __init__(self, offsets, neighbors, vertex_count):
        offsets = np.asarray(offsets, np.int64); pairs = np.asarray(neighbors, np.int64).reshape(-1, 2)
        self.vertex_count = int(vertex_count)
        counts = np.diff(offsets[:self.vertex_count + 1])
        self.owner = np.repeat(np.arange(self.vertex_count), counts)
        start = int(offsets[0]); self.b = pairs[start:start + len(self.owner), 0]; self.c = pairs[start:start + len(self.owner), 1]

    def __call__(self, positions):
        center = positions[self.owner]
        face = np.cross(positions[self.b] - center, positions[self.c] - center)
        out = np.zeros((self.vertex_count, 3), np.float32)
        for axis in range(3):
            out[:, axis] = np.bincount(self.owner, face[:, axis], minlength=self.vertex_count)
        return _normalize(out)


def _normalize(vectors):
    length = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, length, out=np.zeros_like(vectors), where=length > 0)


def interleave(positions, normals):
    """Pack (V,3) positions and normals as std430 Vertex records (V,8)."""
    out = np.zeros((len(positions), 8), np.float32)
    out[:, :3] = positions; out[:, 3] = 1; out[:, 4:7] = normals
    return out
