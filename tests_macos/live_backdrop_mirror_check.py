"""Blender headless: the 'backdrop' command mirrors the webcam backdrop's texture and back."""
import sys, importlib.util
from pathlib import Path
import bpy
jpg = Path(sys.argv[sys.argv.index('--') + 1]).read_bytes()
spec = importlib.util.spec_from_file_location('rgbmocap_live', Path(__file__).resolve().parents[1] / 'live/blender/rgbmocap_live.py')
live = importlib.util.module_from_spec(spec); spec.loader.exec_module(live)
r = live.RECEIVER; r.avatar = live.Avatar(bpy.data.objects['RGBMoCap_Avatar'])
r.command({'cmd': 'backdrop', 'mirror': True})
r.backdrop_jpeg = jpg; r.update_backdrop()
plane = bpy.data.objects['RGBMoCap_Backdrop']
u_left = lambda: plane.data.uv_layers[0].data[0].uv[0]   # u at the plane's left (-X) corner
print('mirror on : left corner shows image u =', u_left())
r.command({'cmd': 'backdrop', 'mirror': False})
print('mirror off: left corner shows image u =', u_left())
assert u_left() == 0.0
r.command({'cmd': 'backdrop', 'mirror': True}); assert u_left() == 1.0
print('BACKDROP MIRROR OK')
