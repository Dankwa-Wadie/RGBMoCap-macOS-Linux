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
import socket

import bpy
from mathutils import Matrix, Quaternion, Vector

PORT = 49170

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
        self.to_local = arm.matrix_world.inverted().to_3x3()
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
        self.last_basis = {}   # bone name -> last tracked basis rotation
        self.weight = {}       # bone name -> 0..1 tracking weight (eases in/out)
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
            if ok(L_SH, R_SH):
                legs_seen = ok(L_KN, R_KN) or ok(L_KN, L_AN) or ok(R_KN, R_AN)
                if ok(L_HIP, R_HIP) and legs_seen:
                    mid_hip = (P[L_HIP] + P[R_HIP]) / 2
                    targets['hips'] = ('frame', frame_from(P[L_HIP] - P[R_HIP], mid_sh - mid_hip))
                    targets['chest'] = ('frame', frame_from(P[L_SH] - P[R_SH], mid_sh - mid_hip))
                elif ok(L_HIP, R_HIP):
                    # Upper body only (seated, legs out of view): the pelvis and resting legs hold
                    # still; leaning and turning go to the spine and chest.
                    mid_hip = (P[L_HIP] + P[R_HIP]) / 2
                    targets['chest'] = ('frame', frame_from(P[L_SH] - P[R_SH], mid_sh - mid_hip))
                else:
                    # Hips out of view: keep the pelvis upright, turned with the shoulders.
                    side = P[L_SH] - P[R_SH]; side = side - side.project(up)
                    targets['hips'] = ('frame', frame_from(side, up))
                    targets['chest'] = ('frame', frame_from(P[L_SH] - P[R_SH], up))
            for h, (a, b) in LIMBS.items():
                if ok(a, b):
                    targets[h] = ('dir', P[b] - P[a])
            if ok(NOSE, L_EAR, R_EAR) or ok(NOSE, L_SH, R_SH):
                ear_x = P[L_EAR] - P[R_EAR]; forward = P[NOSE] - (P[L_EAR] + P[R_EAR]) / 2
                targets['head'] = ('frame', frame_from(ear_x, forward.cross(ear_x)))
        for side, points in (packet.get('hands') or {}).items():
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
            self.apply_face(packet['face'])

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
            w = min(1.0, w + 0.25) if tracked else max(0.0, w - 0.08)
            self.weight[pb.name] = w
            basis = Quaternion().slerp(self.last_basis.get(pb.name, Quaternion()), w)
            matrix = inherited @ basis.to_matrix().to_4x4()
            posed[pb.name] = matrix
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = basis

    def apply_face(self, face):
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


class Receiver:
    def __init__(self):
        self.sock = None; self.avatar = None; self.recording = None; self.last = None; self.frames = 0
        self.live_keys = None  # while recording live: {'action', 't0', 'fps', 'start', 'frame', 'count'}

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
            value = json.loads(data)
            if 'cmd' in value:
                self.command(value)
            else:
                packet = value
        return packet

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
        if value['cmd'] == 'live_record_start' and self.avatar is not None:
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
        if packet:
            self.avatar.apply(packet); self.last = packet; self.frames += 1
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
        arm = avatar_armature(context)
        col.label(text=f'Avatar: {arm.name if arm else "none"}')
        if arm is not None and arm.get('rgbmocap_credit'):
            col.label(text=str(arm['rgbmocap_credit'])[:60])


CLASSES = (RGBMOCAP_OT_live_start, RGBMOCAP_OT_live_stop, RGBMOCAP_OT_live_record, RGBMOCAP_PT_live)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    RECEIVER.stop()
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
