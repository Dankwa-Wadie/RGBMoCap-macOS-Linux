"""Blender startup script: load the RGB MoCap Live receiver for this session and start it.

    Blender live/avatars/Seed-san.blend --python live/blender/start_live.py

The add-on is registered from this folder for the session only; Blender's
preferences are not changed. Install rgbmocap_live.py as an add-on to keep it.
"""
import importlib.util, sys
from pathlib import Path
import bpy

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rgbmocap_live', here / 'rgbmocap_live.py')
live = importlib.util.module_from_spec(spec); sys.modules['rgbmocap_live'] = live; spec.loader.exec_module(live)
live.register()


def begin():
    arm = live.avatar_armature(bpy.context)
    if arm is None:
        print('RGB MoCap Live: no prepared avatar in this file'); return None
    live.RECEIVER.start(arm)
    for area in bpy.context.screen.areas if bpy.context.screen else []:
        if area.type == 'VIEW_3D':
            area.spaces.active.show_region_ui = True
            area.spaces.active.shading.type = 'MATERIAL'
            # Face the avatar from the front and frame it.
            region = next(r for r in area.regions if r.type == 'WINDOW')
            for o in bpy.context.view_layer.objects:
                o.select_set(o == arm or o.parent == arm)
            bpy.context.view_layer.objects.active = arm
            with bpy.context.temp_override(area=area, region=region):
                if 'RoomView' in bpy.data.objects:      # room scan: look through the room camera
                    bpy.context.scene.camera = bpy.data.objects['RoomView']
                    area.spaces.active.region_3d.view_perspective = 'CAMERA'
                else:
                    bpy.ops.view3d.view_axis(type='FRONT')
                    bpy.ops.view3d.view_selected()
            for o in bpy.context.view_layer.objects:
                o.select_set(False)
    print('RGB MoCap Live: listening on udp://127.0.0.1:%d for %s' % (live.PORT, arm.name))
    return None


bpy.app.timers.register(begin, first_interval=1.0)


# Diagnostics: RGBMOCAP_LIVE_DIAG=folder saves a viewport screenshot and the received
# validity / bone angles every second, to compare against capture.py --snapshots.
import json, os, time
DIAG = os.environ.get('RGBMOCAP_LIVE_DIAG')


def diagnostics():
    r = live.RECEIVER
    if not DIAG or r.avatar is None:
        return 1.0
    folder = Path(DIAG); folder.mkdir(parents=True, exist_ok=True); stamp = time.time()
    p = r.last or {}
    av = r.avatar
    names = {11: 'L_sh', 12: 'R_sh', 23: 'L_hip', 24: 'R_hip', 25: 'L_knee', 26: 'R_knee', 27: 'L_ankle', 28: 'R_ankle'}
    info = dict(frames=r.frames, packet_frame=p.get('frame'),
                valid={n: (p.get('pose_valid') or [None] * 33)[i] for i, n in names.items()},
                visibility={n: (p.get('pose_visibility') or [None] * 33)[i] for i, n in names.items()},
                bones={h: round(av.bones[h].rotation_quaternion.angle * 57.2958, 1) for h in
                       ('hips', 'leftUpperLeg', 'rightUpperLeg', 'leftLowerLeg', 'rightLowerLeg', 'leftUpperArm', 'rightUpperArm')
                       if h in av.bones},
                weights={h: round(av.weight.get(av.bones[h].name, 0), 2) for h in ('leftUpperLeg', 'rightUpperLeg') if h in av.bones},
                all_bones={h: round(pb.rotation_quaternion.angle * 57.2958, 1) for h, pb in av.bones.items()},
                all_weights={h: round(av.weight.get(pb.name, 0), 2) for h, pb in av.bones.items()},
                pose_valid=p.get('pose_valid'), hands=sorted((p.get('hands') or {})), face='face' in p)
    (folder / f'blender_{stamp:.0f}.json').write_text(json.dumps(info))
    window = bpy.context.window_manager.windows[0]
    area = next((a for a in window.screen.areas if a.type == 'VIEW_3D'), None)
    if area is not None:
        with bpy.context.temp_override(window=window, area=area):
            bpy.ops.screen.screenshot_area(filepath=str(folder / f'blender_{stamp:.0f}.png'))
    return 1.0


if DIAG:
    bpy.app.timers.register(diagnostics, first_interval=3.0, persistent=True)
