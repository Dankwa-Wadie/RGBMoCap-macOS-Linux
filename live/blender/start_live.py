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
    print('RGB MoCap Live: listening on udp://127.0.0.1:%d for %s' % (live.PORT, arm.name))
    return None


bpy.app.timers.register(begin, first_interval=1.0)
