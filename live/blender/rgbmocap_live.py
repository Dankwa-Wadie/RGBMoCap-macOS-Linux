"""RGB MoCap Live receiver for Blender: webcam face + body + hands onto a humanoid avatar.

Install: Edit > Preferences > Add-ons > Install from Disk > this file, enable
"RGB MoCap Live". In the 3D Viewport press N > RGB MoCap Live:

* Start: listen for live/capture.py on udp://127.0.0.1:49170 and pose the avatar.
* Record / Stop Recording: bake the take into a new Action (bones and face).

The avatar is an armature prepared by live/blender/prepare_vrm.py (any VRM 1.0
model), which stores the humanoid bone map and VRM expressions on the object.
Expressions are driven through VRM presets; meshes that carry ARKit-named shape
keys (eyeBlinkLeft, jawOpen, ...) are also driven directly.
"""
bl_info = {
    'name': 'RGB MoCap Live', 'author': 'RGB MoCap macOS/Linux port', 'version': (0, 1, 0),
    'blender': (4, 2, 0), 'location': 'View3D > N panel > RGB MoCap Live', 'category': 'Animation',
}

import json
import math
import socket

import bpy
from mathutils import Matrix, Quaternion, Vector

PORT = 49170
SPIKE, SPIKE_FRAMES = 80.0, 3        # deg, frames: hold a sudden big flip until it persists
MAX_STEP, MAX_STEP_HAND = 45.0, 30.0  # deg per frame
SMOOTH, SMOOTH_HAND = 3.0, 8.0        # deg: changes this size move half way per frame; much larger ones at once
TORSO_SPAN = 0.35                     # shoulder span / (half torso or nose-shoulder height) below this = unreadable
LYING_FRAMES = 8                      # ~0.4 s of agreement before switching between upright and lying
EXPR_CONFIDENCE, EXPR_FRAMES = 0.35, 3   # visor: calibrated expression confidence; frames before it shows
GRACE_FRAMES = 6                      # frames a bone that lost tracking holds before easing to rest

# MediaPipe pose landmark indices
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN, L_FT, R_FT = 23, 24, 25, 26, 27, 28, 31, 32
NOSE, L_EAR, R_EAR = 0, 7, 8

# humanoid bone -> (from landmark, to landmark) for direction-aimed limbs
LIMBS = {
    'leftUpperArm': (L_SH, L_EL), 'leftLowerArm': (L_EL, L_WR),
    'rightUpperArm': (R_SH, R_EL), 'rightLowerArm': (R_EL, R_WR),
    'leftUpperLeg': (L_HIP, L_KN), 'leftLowerLeg': (L_KN, L_AN), 'leftFoot': (L_AN, L_FT),
    'rightUpperLeg': (R_HIP, R_KN), 'rightLowerLeg': (R_KN, R_AN), 'rightFoot': (R_AN, R_FT),
}
FINGERS = {  # humanoid suffix -> (from, to) hand landmark
    'ThumbMetacarpal': (1, 2), 'ThumbProximal': (2, 3), 'ThumbDistal': (3, 4),
    'IndexProximal': (5, 6), 'IndexIntermediate': (6, 7), 'IndexDistal': (7, 8),
    'MiddleProximal': (9, 10), 'MiddleIntermediate': (10, 11), 'MiddleDistal': (11, 12),
    'RingProximal': (13, 14), 'RingIntermediate': (14, 15), 'RingDistal': (15, 16),
    'LittleProximal': (17, 18), 'LittleIntermediate': (18, 19), 'LittleDistal': (19, 20),
}


def avg(face, *names):
    return sum(face.get(n, 0.0) for n in names) / len(names)


def vrm_expressions(face):
    """ARKit blendshapes (0..1) -> VRM 1.0 preset expression weights."""
    clamp = lambda v: max(0.0, min(1.0, v))
    return {
        'blinkLeft': clamp(face.get('eyeBlinkLeft', 0) * 1.3), 'blinkRight': clamp(face.get('eyeBlinkRight', 0) * 1.3),
        'aa': clamp(face.get('jawOpen', 0) * 1.4), 'oh': clamp(face.get('mouthFunnel', 0) * 1.5),
        'ou': clamp(face.get('mouthPucker', 0) * 1.2), 'ee': clamp(avg(face, 'mouthStretchLeft', 'mouthStretchRight') * 2),
        'ih': clamp(avg(face, 'mouthRollLower', 'mouthShrugUpper') * 1.5),
        'happy': clamp(avg(face, 'mouthSmileLeft', 'mouthSmileRight') * 1.3),
        'sad': clamp(avg(face, 'mouthFrownLeft', 'mouthFrownRight') * 2),
        'angry': clamp(avg(face, 'browDownLeft', 'browDownRight') * 1.5),
        'surprised': clamp(avg(face, 'browInnerUp', 'eyeWideLeft', 'eyeWideRight') * 1.5),
        'lookUp': clamp(avg(face, 'eyeLookUpLeft', 'eyeLookUpRight')),
        'lookDown': clamp(avg(face, 'eyeLookDownLeft', 'eyeLookDownRight')),
        'lookLeft': clamp(avg(face, 'eyeLookOutLeft', 'eyeLookInRight')),
        'lookRight': clamp(avg(face, 'eyeLookOutRight', 'eyeLookInLeft')),
    }


def frame_from(x_axis, up):
    """Orthonormal 3x3 (columns x, y=forward, z=up) from a side axis and an up hint."""
    x = Vector(x_axis).normalized()
    forward = Vector(up).cross(x).normalized()
    z = x.cross(forward).normalized()
    return Matrix((x, forward, z)).transposed()


class Avatar:
    def __init__(self, arm):
        self.arm = arm
        self.human = json.loads(arm.get('rgbmocap_humanoid', '{}'))
        self.expressions = json.loads(arm.get('rgbmocap_expressions', '{}'))
        self.bones = {h: arm.pose.bones[b] for h, b in self.human.items() if b in arm.pose.bones}
        self.order = sorted(arm.pose.bones, key=lambda pb: len(pb.parent_recursive))
        rest = lambda h: self.bones[h].bone.head_local
        # Tracking is relative to the performer facing the camera, so it applies in the avatar's
        # own frame: turning or moving the avatar object turns the whole performance with it.
        self.to_local = Matrix.Identity(3)
        # Rest frames from the rig itself (armature space): side axis + up.
        has = lambda *hs: all(h in self.bones for h in hs)
        self.rest_frames = {}
        if has('hips', 'leftUpperLeg', 'rightUpperLeg', 'neck'):
            self.rest_frames['hips'] = frame_from(rest('leftUpperLeg') - rest('rightUpperLeg'), rest('neck') - rest('hips'))
        if has('chest', 'leftUpperArm', 'rightUpperArm', 'neck', 'hips'):
            self.rest_frames['chest'] = frame_from(rest('leftUpperArm') - rest('rightUpperArm'), rest('neck') - rest('hips'))
        self.rest_frames['head'] = frame_from((1, 0, 0), (0, 0, 1))
        for side in ('left', 'right'):
            names = [side + n for n in ('Hand', 'MiddleProximal', 'IndexProximal', 'LittleProximal')]
            if has(*names):
                hand, middle, index, little = (rest(n) for n in names)
                self.rest_frames[side + 'Hand'] = self.hand_frame(middle - hand, index - little)
        self.lying = False
        self.lying_votes = 0
        self.lost = {}                      # bone name -> frames since it was last tracked
        self.applied, self.spikes = {}, {}   # bone name -> last applied basis / frames a big jump has held
        self.hand_bones = {pb.name for h, pb in self.bones.items() if 'Hand' in h or any(f in h for f in ('Thumb', 'Index', 'Middle', 'Ring', 'Little'))}
        self.last_basis = {}   # bone name -> last tracked basis rotation
        self.weight = {}       # bone name -> 0..1 tracking weight (eases in/out)
        # Pixel-screen face (an avatar with 'rgbmocap_visor'): pre-painted images 'visor_face:<expr>_<blink>_<talk>_<gaze>'.
        visor = json.loads(arm.get('rgbmocap_visor', 'null'))
        mat = visor and bpy.data.materials.get(visor['material'])
        self.visor_expr, self.visor_next, self.visor_votes = 'neutral', None, 0
        self.visor_node = mat and next((n for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE'), None)
        self.shape_targets = [o for o in bpy.data.objects if o.type == 'MESH' and o.data.shape_keys
                              and any(m.type == 'ARMATURE' and m.object == arm for m in o.modifiers)]

    @staticmethod
    def hand_frame(forward, side):
        f = Vector(forward).normalized()
        z = f.cross(Vector(side)).normalized()
        x = f.cross(z).normalized()
        return Matrix((x, f, z)).transposed()

    def apply(self, packet):
        targets = {}  # humanoid -> ('dir', Vector) | ('frame', Matrix3)
        pose = packet.get('pose')
        if pose:
            P = [self.to_local @ Vector(p) for p in pose]
            valid = packet.get('pose_valid') or [True] * len(P)
            ok = lambda *ids: all(valid[i] for i in ids)
            mid_sh = (P[L_SH] + P[R_SH]) / 2
            up = self.to_local @ Vector((0, 0, 1))
            if ok(L_SH, R_SH) and self.torso_readable(packet, ok):
                legs_seen = ok(L_KN, R_KN) or ok(L_KN, L_AN) or ok(R_KN, R_AN)
                if ok(L_HIP, R_HIP):
                    # Upright (standing/sitting) vs lying: past ~45 deg from vertical the whole
                    # avatar follows the torso; back under ~35 deg the seated rule returns.
                    torso = ((P[L_SH] + P[R_SH]) / 2 - (P[L_HIP] + P[R_HIP]) / 2).normalized()
                    tilt = math.degrees(torso.angle(up)) if torso.length else 0.0
                    want = tilt > (35.0 if self.lying else 45.0)
                    image = packet.get('pose_image')
                    if want and image:  # and lying in the picture: shoulders beside the hips, not above them
                        dx = (image[L_SH][0] + image[R_SH][0] - image[L_HIP][0] - image[R_HIP][0]) / 2
                        dy = (image[L_SH][1] + image[R_SH][1] - image[L_HIP][1] - image[R_HIP][1]) / 2
                        want = abs(dx) > abs(dy) * (0.8 if self.lying else 1.0)
                    # switch only after it has held LYING_FRAMES (a tipped 3D guess flickers frame to frame)
                    self.lying_votes = self.lying_votes + 1 if want != self.lying else 0
                    if self.lying_votes >= LYING_FRAMES:
                        self.lying, self.lying_votes = want, 0
                if ok(L_HIP, R_HIP) and (legs_seen or self.lying):
                    mid_hip = (P[L_HIP] + P[R_HIP]) / 2
                    targets['hips'] = ('frame', frame_from(P[L_HIP] - P[R_HIP], mid_sh - mid_hip))
                    targets['chest'] = ('frame', frame_from(P[L_SH] - P[R_SH], mid_sh - mid_hip))
                elif ok(L_HIP, R_HIP):
                    # Upper body only (seated, legs out of view): the pelvis and resting legs hold
                    # still; leaning and turning go to the spine and chest.
                    mid_hip = (P[L_HIP] + P[R_HIP]) / 2
                    targets['chest'] = ('frame', frame_from(P[L_SH] - P[R_SH], mid_sh - mid_hip))
                else:
                    # Hips out of view (or switched off): the pelvis and legs hold still;
                    # only the chest follows the shoulders.
                    targets['chest'] = ('frame', frame_from(P[L_SH] - P[R_SH], up))
            for h, (a, b) in LIMBS.items():
                if ok(a, b):
                    targets[h] = ('dir', P[b] - P[a])
            # An arm the camera cannot see (below the frame at a desk, or shoulders out of the top when
            # close to the camera) hangs at the side instead of easing back to the rig's A-/T-pose.
            down = -(mid_sh - (P[L_HIP] + P[R_HIP]) / 2 if ok(L_SH, R_SH, L_HIP, R_HIP) else up).normalized()
            for side in ('left', 'right'):
                if side + 'UpperArm' not in targets:
                    targets[side + 'UpperArm'] = ('dir', down)
                if side + 'LowerArm' not in targets:
                    targets[side + 'LowerArm'] = ('dir', targets[side + 'UpperArm'][1])
            if ok(L_SH, R_SH, L_EAR, R_EAR):
                self.reach_head(P, ok, targets)
            if ok(NOSE, L_EAR, R_EAR) or ok(NOSE, L_SH, R_SH):
                ear_x = P[L_EAR] - P[R_EAR]; forward = P[NOSE] - (P[L_EAR] + P[R_EAR]) / 2
                targets['head'] = ('frame', frame_from(ear_x, forward.cross(ear_x)))
        valid_now = packet.get('pose_valid') or []
        for side, points in (packet.get('hands') or {}).items():
            # A hand only moves with a tracked wrist; otherwise it would twist on a resting arm.
            wrist = L_WR if side == 'left' else R_WR
            if not (pose and valid_now and valid_now[wrist]):
                continue
            H = [self.to_local @ Vector(p) for p in points]
            targets[side + 'Hand'] = ('frame', self.hand_frame(H[9] - H[0], H[5] - H[17]))
            for suffix, (a, b) in FINGERS.items():
                targets[side + suffix] = ('dir', H[b] - H[a])
        if 'chest' in targets and 'spine' in self.bones:
            targets['spine'] = ('blend', ('hips', 'chest'))
        if 'chest' in targets and 'head' in targets and 'neck' in self.bones:
            targets['neck'] = ('blend', ('chest', 'head'))
        self.pose_bones(targets)
        if packet.get('face'):
            self.apply_face(packet['face'], packet.get('face_expr'))

    @staticmethod
    def torso_readable(packet, ok):
        """False when the shoulders collapse to a point in the picture (torso hidden behind knees, a phone,
        or seen edge-on): the side axis is then noise and the torso would spin; it holds its last pose instead."""
        image = packet.get('pose_image')
        if not image:
            return True
        span = math.dist(image[L_SH], image[R_SH])
        ref = math.dist(image[NOSE], [(image[L_SH][0] + image[R_SH][0]) / 2, (image[L_SH][1] + image[R_SH][1]) / 2])
        if ok(L_HIP, R_HIP):
            ref = max(ref, math.dist([(image[L_SH][0] + image[R_SH][0]) / 2, (image[L_SH][1] + image[R_SH][1]) / 2],
                                     [(image[L_HIP][0] + image[R_HIP][0]) / 2, (image[L_HIP][1] + image[R_HIP][1]) / 2]) * 0.5)
        return span > TORSO_SPAN * ref

    def world_rotation(self, human, kind, value, absolute):
        rest = self.rest_frames.get(human)
        if kind == 'frame' and rest is not None:
            return (value @ rest.transposed()).to_quaternion()
        if kind == 'blend':
            # An untracked end (e.g. hidden hips) blends from its rest orientation.
            a, b = (absolute.get(n, Quaternion()) for n in value)
            if value[1] in absolute:
                return a.slerp(b, 0.5)
        return None

    def pose_bones(self, targets):
        by_bone = {pb.name: h for h, pb in self.bones.items()}
        posed = {}      # bone name -> pose matrix (armature space)
        absolute = {}   # humanoid -> applied world rotation (for blends)
        for pb in self.order:
            rest = pb.bone.matrix_local
            if pb.parent:
                inherited = posed[pb.parent.name] @ pb.parent.bone.matrix_local.inverted() @ rest
            else:
                inherited = rest.copy()
            matrix = inherited
            human = by_bone.get(pb.name)
            target = targets.get(human)
            if target:
                kind, value = target
                if kind == 'dir' and value.length > 1e-6:
                    current = inherited.to_3x3() @ Vector((0, 1, 0))
                    swing = current.rotation_difference(value.normalized())
                    matrix = Matrix.Translation(inherited.to_translation()) @ (swing.to_matrix() @ inherited.to_3x3()).to_4x4()
                else:
                    rotation = self.world_rotation(human, kind, value, absolute)
                    if rotation is not None:
                        absolute[human] = rotation
                        matrix = Matrix.Translation(inherited.to_translation()) @ (rotation.to_matrix() @ rest.to_3x3()).to_4x4()
            tracked = matrix is not inherited
            if tracked:
                self.last_basis[pb.name] = (inherited.inverted() @ matrix).to_quaternion()
            # Ease tracked bones in, and untracked ones (out of frame) back to their rest pose.
            w = self.weight.get(pb.name, 0.0)
            # Grace: a bone that just lost tracking (a hand missed for a frame or two) holds its pose
            # for GRACE_FRAMES before easing back, so short dropouts do not flicker.
            self.lost[pb.name] = 0 if tracked else self.lost.get(pb.name, 0) + 1
            w = min(1.0, w + 0.25) if tracked else w if self.lost[pb.name] <= GRACE_FRAMES else max(0.0, w - 0.08)
            self.weight[pb.name] = w
            basis = self.steady(pb.name, Quaternion().slerp(self.last_basis.get(pb.name, Quaternion()), w))
            matrix = inherited @ basis.to_matrix().to_4x4()
            posed[pb.name] = matrix
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = basis
        self.posed = posed

    def steady(self, name, basis):
        """Tracking glitches (a hand misread for a frame or two, swapped hands) flip bones by 90-150 deg and
        back. A jump over SPIKE deg is held until it persists SPIKE_FRAMES frames; every bone then turns at
        most MAX_STEP deg per frame (hands/fingers less), so real fast moves still come through."""
        prev = self.applied.get(name)
        if prev is not None:
            if prev.dot(basis) < 0:
                basis = -basis
            angle = math.degrees(prev.rotation_difference(basis).angle)
            if angle > SPIKE:
                self.spikes[name] = self.spikes.get(name, 0) + 1
                if self.spikes[name] < SPIKE_FRAMES:
                    return prev
            else:
                self.spikes[name] = 0
            hand = name in self.hand_bones
            limit = MAX_STEP_HAND if hand else MAX_STEP
            # Adaptive smoothing (One-Euro-like on the rotation): small changes are mostly noise and are
            # damped hard; large ones are real motion and pass almost at once (little lag).
            soft = SMOOTH_HAND if hand else SMOOTH
            step = min(angle, limit) * (angle / (angle + soft) if angle > 1e-6 else 1.0)
            if angle > 1e-6 and step < angle:
                basis = prev.slerp(basis, step / angle)
        self.applied[name] = basis.copy()
        return basis

    def reach_head(self, P, ok, targets):
        """Hand at the face (touching, scratching, thinking): limb directions alone miss on an avatar with
        other proportions. Place the wrist where it is relative to the head, scaled to the avatar, and
        solve the arm as two bones, the elbow bending toward the performer's."""
        has = lambda *hs: all(h in self.bones for h in hs)
        if not has('head', 'leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'rightLowerArm', 'leftHand', 'rightHand'):
            return
        rest = lambda h: self.bones[h].bone.head_local
        scale = (rest('leftUpperArm') - rest('rightUpperArm')).length / max(1e-6, (P[L_SH] - P[R_SH]).length)
        ears = (P[L_EAR] + P[R_EAR]) / 2
        head = rest('head') + Vector((0, 0, 0.08))  # middle of the head (the head bone sits at its base)
        for side, sh, el, wr in (('left', L_SH, L_EL, L_WR), ('right', R_SH, R_EL, R_WR)):
            if not ok(sh, el, wr):
                continue
            near = (P[wr] - ears).length / max(1e-6, (P[L_SH] - P[R_SH]).length)  # in shoulder widths
            k = min(1.0, max(0.0, (1.1 - near) / 0.4))  # 0 beyond 1.1, full inside 0.7
            if k <= 0:
                continue
            S = rest(side + 'UpperArm'); E0 = rest(side + 'LowerArm'); W0 = rest(side + 'Hand')
            a, b = (E0 - S).length, (W0 - E0).length
            goal = head + (P[wr] - ears) * scale
            reach = goal - S
            d = min(reach.length, (a + b) * 0.999)
            axis = reach.normalized()
            pole = (P[el] - P[sh]) - (P[el] - P[sh]).project(P[wr] - P[sh])  # where the performer's elbow points
            pole = (pole - pole.project(axis)).normalized() if pole.length > 1e-6 else Vector((0, 0, -1))
            along = (a * a - b * b + d * d) / (2 * d)
            elbow = S + axis * along + pole * math.sqrt(max(0.0, a * a - along * along))
            wrist = S + axis * d
            for h, v in ((side + 'UpperArm', elbow - S), (side + 'LowerArm', wrist - elbow)):
                old_dir = targets[h][1] if h in targets and targets[h][0] == 'dir' else v
                targets[h] = ('dir', old_dir.normalized().lerp(v.normalized(), k))

    def apply_visor(self, face, named=None):
        e = vrm_expressions(face)
        if named and named[0] and named[1] >= EXPR_CONFIDENCE:
            expr = named[0]  # the performer's own calibrated expression (live/face_profile.py)
        else:
            expr = max(('happy', 'sad', 'angry', 'surprised'), key=e.get)
            expr = expr if e[expr] > 0.35 else 'neutral'
        # A new expression shows once it has held EXPR_FRAMES frames (no flicker between two close faces).
        if expr != self.visor_expr:
            self.visor_votes = self.visor_votes + 1 if expr == self.visor_next else 1
            self.visor_next = expr
            if self.visor_votes >= EXPR_FRAMES:
                self.visor_expr = expr
        expr = self.visor_expr
        blink = int(min(e['blinkLeft'], e['blinkRight']) > 0.5)
        talk = (0, 0.5, 1)[min(2, int(e['aa'] * 3))]
        g = e['lookLeft'] - e['lookRight']  # performer's left = the avatar's left = viewer's right (+x on the screen)
        gaze = 1 if g > 0.3 else -1 if g < -0.3 else 0
        image = bpy.data.images.get(f'visor_face:{expr}_{blink}_{talk:g}_{gaze}')
        if image and self.visor_node.image != image:
            self.visor_node.image = image

    def apply_face(self, face, named=None):
        if self.visor_node:
            self.apply_visor(face, named)
        weights = {}
        for name, value in vrm_expressions(face).items():
            for obj_name, key, w in self.expressions.get(name, []):
                weights[(obj_name, key)] = max(weights.get((obj_name, key), 0.0), value * w)
        for obj in self.shape_targets:
            blocks = obj.data.shape_keys.key_blocks
            for name, value in face.items():  # ARKit-named shape keys, when present
                if name in blocks:
                    blocks[name].value = value
            for (obj_name, key), value in weights.items():
                if obj_name == obj.name and key in blocks:
                    blocks[key].value = min(1.0, value)


# MediaPipe pose landmark -> humanoid bone whose head sits at that joint (camera matching).
JOINT_BONES = {11: 'leftUpperArm', 12: 'rightUpperArm', 13: 'leftLowerArm', 14: 'rightLowerArm',
               15: 'leftHand', 16: 'rightHand', 23: 'leftUpperLeg', 24: 'rightUpperLeg',
               25: 'leftLowerLeg', 26: 'rightLowerLeg', 27: 'leftFoot', 28: 'rightFoot'}


ROLL_PRIOR, LEG_WEIGHT = 2.0, 0.3     # camera matching: keep the webcam upright; knees/ankles count less


class CameraMatch:
    """A Blender camera that sees the avatar the way the webcam sees the performer: same framing, height, tilt
    and crop (sitting on the floor with only the upper body in view -> the avatar is framed the same).

    Each packet: the avatar's posed joints (world) and the performer's joints in the picture (normalized x, y)
    give a pinhole camera fit (position + yaw/pitch/roll, fixed horizontal FOV from the packet) by damped
    Gauss-Newton from the previous solution. Only confidently seen joints count; with fewer than MIN_POINTS,
    or a poor fit, the camera holds. The result is smoothed so the shot does not shake."""
    MIN_POINTS, MAX_ERROR, SMOOTH = 4, 0.06, 0.25

    def __init__(self):
        self.params = None          # solved [x, y, z, yaw, pitch, roll]
        self.shown = None           # smoothed params applied to the camera
        self.error = None

    @staticmethod
    def rotation(yaw, pitch, roll):
        import numpy as np
        cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
        # camera axes (columns): right, up, back (-view). Facing +Y at yaw = 0, pitch tilts the view up.
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        Rroll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # right = +X, up = +Z, back = -Y
        return Rz @ base @ Rx @ Rroll

    def project(self, p, X, aspect, f):
        import numpy as np
        R = self.rotation(*p[3:])
        q = (X - p[:3]) @ R                      # camera space: x right, y up, z back
        depth = np.maximum(-q[:, 2], 1e-3)
        return np.stack([0.5 + f * q[:, 0] / depth / aspect, 0.5 - f * q[:, 1] / depth], 1)

    def solve(self, X, uv, aspect, hfov, weights=None):
        import numpy as np
        f = 0.5 * aspect / math.tan(math.radians(hfov) / 2)   # in units of image height
        p = np.array(self.params if self.params is not None else
                     [X[:, 0].mean(), X[:, 1].mean() - 3.0, X[:, 2].mean(), 0.0, 0.0, 0.0], float)
        prior = p.copy()
        w = np.repeat(weights if weights is not None else np.ones(len(X)), 2)
        def residual(p):
            r = (self.project(p, X, aspect, f) - uv).ravel() * w
            # priors: webcams stand upright (roll ~0), and stay near the last solution when few joints constrain it
            return np.concatenate([r, [p[5] * ROLL_PRIOR], (p - prior) * np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05])])
        lam = 1e-3
        for _ in range(12):
            r = residual(p)
            J = np.empty((r.size, 6)); eps = 1e-4
            for k in range(6):
                d = np.zeros(6); d[k] = eps
                J[:, k] = (residual(p + d) - r) / eps
            step = np.linalg.solve(J.T @ J + lam * np.eye(6), -J.T @ r)
            if (residual(p + step) ** 2).sum() < (r ** 2).sum():
                p, lam = p + step, lam * 0.5
            else:
                lam *= 4
            if np.abs(step).max() < 1e-5:
                break
        err = float(np.sqrt(((self.project(p, X, aspect, f) - uv) ** 2).sum(1)).mean())
        return p, err

    def update(self, avatar, packet):
        """Fit to this packet; returns the smoothed params (or None before the first good fit)."""
        import numpy as np
        image, uv_all, valid = packet.get('image'), packet.get('pose_image'), packet.get('pose_valid')
        posed = getattr(avatar, 'posed', None)
        if not (image and uv_all and valid and posed):
            return self.shown
        ids = [i for i, h in JOINT_BONES.items() if valid[i] and h in avatar.bones]
        if len(ids) < self.MIN_POINTS:
            return self.shown
        world = avatar.arm.matrix_world
        X = np.array([tuple(world @ posed[avatar.bones[JOINT_BONES[i]].name].to_translation()) for i in ids])
        uv = np.array([uv_all[i] for i in ids], float)
        aspect = image[0] / max(1, image[1])
        # leg joints: least reliable in depth (knees toward the camera), so they steer the shot less
        weights = np.array([LEG_WEIGHT if i >= 25 else 1.0 for i in ids])
        p, err = self.solve(X, uv, aspect, image[2] if len(image) > 2 else 70.0, weights)
        self.error = err
        if err > self.MAX_ERROR or not np.all(np.isfinite(p)):
            return self.shown
        self.params = p
        self.shown = p if self.shown is None else self.shown + (p - self.shown) * self.SMOOTH
        return self.shown

    def apply(self, scene, packet):
        """Put the smoothed fit on the 'RGBMoCap_MatchCam' camera and make it the scene camera."""
        if self.shown is None:
            return
        from mathutils import Matrix
        cam = bpy.data.objects.get('RGBMoCap_MatchCam')
        if cam is None:
            cam = bpy.data.objects.new('RGBMoCap_MatchCam', bpy.data.cameras.new('RGBMoCap_MatchCam'))
            scene.collection.objects.link(cam)
        image = packet.get('image') or [16, 9, 70.0]
        cam.data.sensor_fit = 'HORIZONTAL'; cam.data.sensor_width = 36.0
        cam.data.lens = 18.0 / math.tan(math.radians(image[2] if len(image) > 2 else 70.0) / 2)
        R = self.rotation(*self.shown[3:])
        cam.matrix_world = Matrix([[R[0][0], R[0][1], R[0][2], self.shown[0]], [R[1][0], R[1][1], R[1][2], self.shown[1]],
                                   [R[2][0], R[2][1], R[2][2], self.shown[2]], [0, 0, 0, 1]])
        scene.camera = cam
        w, h = image[0], image[1]
        if scene.render.resolution_x * h != scene.render.resolution_y * w:  # same aspect as the webcam
            scene.render.resolution_y = round(scene.render.resolution_x * h / w)


class Receiver:
    def __init__(self):
        self.sock = None; self.avatar = None; self.recording = None; self.last = None; self.frames = 0
        self.live_keys = None  # while recording live: {'action', 't0', 'fps', 'start', 'frame', 'count'}
        self.backdrop_jpeg = None; self.backdrop_at = 0.0; self.backdrop_mirror = False
        self.camera_match = CameraMatch()

    def start(self, arm, port=PORT):
        self.avatar = Avatar(arm)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', port)); self.sock.setblocking(False)
        bpy.app.timers.register(self.tick, persistent=True)

    def stop(self):
        if self.sock:
            self.sock.close()
        self.sock = None

    def poll(self):
        """Newest pose packet; control commands (from the RGB MoCap app) run as they arrive."""
        packet = None
        while True:
            try:
                data, _ = self.sock.recvfrom(1 << 20)
            except (BlockingIOError, OSError):
                break
            if data[:2] == b'\xff\xd8':  # a JPEG camera frame for the backdrop
                self.backdrop_jpeg = data
                continue
            value = json.loads(data)
            if 'cmd' in value:
                self.command(value)
            else:
                packet = value
        return packet

    def update_backdrop(self):
        """Show the newest camera frame on a plane behind the avatar (~15 updates/s)."""
        import os, tempfile, time
        now = time.monotonic()
        if now - self.backdrop_at < 1 / 15:
            return
        self.backdrop_at = now
        path = os.path.join(tempfile.gettempdir(), 'rgbmocap_backdrop.jpg')
        with open(path, 'wb') as f:
            f.write(self.backdrop_jpeg)
        self.backdrop_jpeg = None
        image = bpy.data.images.get('RGBMoCap_Webcam')
        if image is None:
            image = bpy.data.images.load(path); image.name = 'RGBMoCap_Webcam'
            self.make_backdrop(image)
        else:
            image.reload()

    def make_backdrop(self, image):
        arm = self.avatar.arm if self.avatar else None
        height = 2.2; width = height * image.size[0] / max(1, image.size[1])
        mesh = bpy.data.meshes.new('RGBMoCap_Backdrop')
        mesh.from_pydata([(-width / 2, 0, 0), (width / 2, 0, 0), (width / 2, 0, height), (-width / 2, 0, height)], [], [(0, 1, 2, 3)])
        mesh.uv_layers.new().data.foreach_set('uv', self.backdrop_uv())
        plane = bpy.data.objects.new('RGBMoCap_Backdrop', mesh)
        plane.location = (arm.location.x if arm else 0, (arm.location.y if arm else 0) + 1.6, 0)
        material = bpy.data.materials.new('RGBMoCap_Webcam'); material.use_nodes = True
        nodes, links = material.node_tree.nodes, material.node_tree.links
        nodes.clear()
        tex = nodes.new('ShaderNodeTexImage'); tex.image = image
        emit = nodes.new('ShaderNodeEmission'); out = nodes.new('ShaderNodeOutputMaterial')
        links.new(tex.outputs[0], emit.inputs[0]); links.new(emit.outputs[0], out.inputs[0])
        mesh.materials.append(material)
        bpy.context.scene.collection.objects.link(plane)

    def backdrop_uv(self):
        # Mirrored: the image's right edge on the plane's left, like a mirror.
        return [1, 0, 0, 0, 0, 1, 1, 1] if self.backdrop_mirror else [0, 0, 1, 0, 1, 1, 0, 1]

    def set_backdrop_mirror(self, on):
        if on == self.backdrop_mirror:
            return
        self.backdrop_mirror = on
        plane = bpy.data.objects.get('RGBMoCap_Backdrop')
        if plane is not None and plane.data.uv_layers:
            plane.data.uv_layers[0].data.foreach_set('uv', self.backdrop_uv()); plane.data.update()

    def driven_shape_keys(self):
        keys = set()
        for binds in self.avatar.expressions.values():
            keys.update((o, k) for o, k, _ in binds)
        for obj in self.avatar.shape_targets:
            for kb in obj.data.shape_keys.key_blocks[1:]:
                if kb.name[:1].islower() and any(kb.name.startswith(p) for p in ('eye', 'mouth', 'jaw', 'brow', 'cheek', 'nose', 'tongue')):
                    keys.add((obj.name, kb.name))
        return keys

    def key_live(self, packet):
        state = self.live_keys
        if state['t0'] is None:
            state['t0'] = packet['t']
        frame = state['start'] + round((packet['t'] - state['t0']) * state['fps'])
        if frame == state['frame']:
            return
        state['frame'] = frame; state['count'] += 1
        for pb in self.avatar.bones.values():
            pb.keyframe_insert('rotation_quaternion', frame=frame, group=pb.name)
        for obj_name, key in state['shape_keys']:
            obj = bpy.data.objects.get(obj_name)
            if obj is not None and key in obj.data.shape_keys.key_blocks:
                obj.data.shape_keys.key_blocks[key].keyframe_insert('value', frame=frame)
        scene = bpy.context.scene
        scene.frame_current = frame
        scene.frame_end = max(scene.frame_end, frame)

    def command(self, value):
        if value['cmd'] == 'backdrop':
            self.set_backdrop_mirror(bool(value.get('mirror')))
        elif value['cmd'] == 'match_camera':
            bpy.context.scene.rgbmocap_match_camera = bool(value.get('on'))
        elif value['cmd'] == 'live_record_start' and self.avatar is not None:
            arm = self.avatar.arm; arm.animation_data_create()
            from datetime import datetime
            action = bpy.data.actions.new('RGBMoCap_Live_' + datetime.now().strftime('%H%M%S'))
            arm.animation_data.action = action
            for obj in self.avatar.shape_targets:  # fresh shape-key actions too
                obj.data.shape_keys.animation_data_create()
                obj.data.shape_keys.animation_data.action = bpy.data.actions.new(action.name + '_' + obj.name)
            scene = bpy.context.scene
            self.live_keys = dict(action=action, t0=None, fps=scene.render.fps / scene.render.fps_base,
                                  start=scene.frame_start, frame=None, count=0, shape_keys=self.driven_shape_keys())
            scene.frame_current = scene.frame_start
            print('RGB MoCap Live: recording into', action.name)
        elif value['cmd'] == 'live_record_stop' and self.live_keys is not None:
            state, self.live_keys = self.live_keys, None
            print(f"RGB MoCap Live: recorded {state['count']} frames into {state['action'].name}")
        elif value['cmd'] == 'bake_take':
            # Bake a recorded (optionally trimmed) take saved by the app: JSON lines of packets.
            rows = [json.loads(line) for line in open(value['path'], encoding='utf-8')]
            first, last = value.get('in', 0), value.get('out', len(rows) - 1)
            self.recording = rows[first:last + 1]
            scene = bpy.context.scene
            frames = self.bake(round(scene.render.fps / scene.render.fps_base))
            print(f'RGB MoCap Live: baked {frames} frames from {value["path"]}')

    def tick(self):
        if self.sock is None:
            return None
        packet = self.poll()
        if self.backdrop_jpeg is not None:
            self.update_backdrop()
        if packet:
            self.avatar.apply(packet); self.last = packet; self.frames += 1
            scene = bpy.context.scene
            if getattr(scene, 'rgbmocap_match_camera', False):
                self.camera_match.update(self.avatar, packet); self.camera_match.apply(scene, packet)
            if self.recording is not None:
                self.recording.append(packet)
            if self.live_keys is not None:
                self.key_live(packet)
        return 1 / 60

    def bake(self, fps=30):
        arm = self.avatar.arm; take = self.recording or []; self.recording = None
        if not take:
            return 0
        arm.animation_data_create()
        action = bpy.data.actions.new('RGBMoCap_Live_Take'); arm.animation_data.action = action
        t0 = take[0]['t']; scene = bpy.context.scene
        for packet in take:
            frame = scene.frame_start + round((packet['t'] - t0) * fps)
            self.avatar.apply(packet)
            for pb in self.avatar.bones.values():
                pb.keyframe_insert('rotation_quaternion', frame=frame)
            for obj in self.avatar.shape_targets:
                for kb in obj.data.shape_keys.key_blocks[1:]:
                    kb.keyframe_insert('value', frame=frame)
        scene.frame_end = max(scene.frame_end, frame)
        return len(take)


RECEIVER = Receiver()


def avatar_armature(context):
    obj = context.active_object
    if obj and obj.type == 'ARMATURE' and 'rgbmocap_humanoid' in obj:
        return obj
    return next((o for o in context.scene.objects if o.type == 'ARMATURE' and 'rgbmocap_humanoid' in o), None)


class RGBMOCAP_OT_live_start(bpy.types.Operator):
    bl_idname = 'rgbmocap.live_start'; bl_label = 'Start'
    def execute(self, context):
        arm = avatar_armature(context)
        if arm is None:
            self.report({'ERROR'}, 'No prepared avatar. Open live/avatars/Seed-san.blend or run prepare_vrm.py.')
            return {'CANCELLED'}
        RECEIVER.start(arm); return {'FINISHED'}


class RGBMOCAP_OT_live_stop(bpy.types.Operator):
    bl_idname = 'rgbmocap.live_stop'; bl_label = 'Stop'
    def execute(self, context):
        RECEIVER.stop(); return {'FINISHED'}


class RGBMOCAP_OT_live_record(bpy.types.Operator):
    bl_idname = 'rgbmocap.live_record'; bl_label = 'Record'
    def execute(self, context):
        if RECEIVER.recording is None:
            RECEIVER.recording = []
        else:
            n = RECEIVER.bake(round(context.scene.render.fps / context.scene.render.fps_base))
            self.report({'INFO'}, f'Baked {n} frames into RGBMoCap_Live_Take')
        return {'FINISHED'}


class RGBMOCAP_PT_live(bpy.types.Panel):
    bl_label = 'RGB MoCap Live'; bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = 'RGB MoCap Live'
    def draw(self, context):
        col = self.layout.column()
        listening = RECEIVER.sock is not None
        col.operator('rgbmocap.live_stop' if listening else 'rgbmocap.live_start')
        if listening:
            col.operator('rgbmocap.live_record', text='Stop Recording + Bake' if RECEIVER.recording is not None else 'Record')
            col.label(text=f'Frames received: {RECEIVER.frames}')
        col.prop(context.scene, 'rgbmocap_match_camera')
        if context.scene.rgbmocap_match_camera and RECEIVER.camera_match.error is not None:
            col.label(text=f'Camera fit error: {RECEIVER.camera_match.error * 100:.1f} % of the picture')
        arm = avatar_armature(context)
        col.label(text=f'Avatar: {arm.name if arm else "none"}')
        if arm is not None and arm.get('rgbmocap_credit'):
            col.label(text=str(arm['rgbmocap_credit'])[:60])


CLASSES = (RGBMOCAP_OT_live_start, RGBMOCAP_OT_live_stop, RGBMOCAP_OT_live_record, RGBMOCAP_PT_live)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.rgbmocap_match_camera = bpy.props.BoolProperty(
        name='Match webcam camera', default=False,
        description='Move a camera (RGBMoCap_MatchCam) so the avatar is framed like you are on the webcam')


def unregister():
    RECEIVER.stop()
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.rgbmocap_match_camera
