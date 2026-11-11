"""Blender headless: a JPEG datagram creates and updates the webcam backdrop behind the avatar."""
import sys, importlib.util
from pathlib import Path
import bpy
jpg = Path(sys.argv[sys.argv.index('--') + 1]).read_bytes()
spec = importlib.util.spec_from_file_location('rgbmocap_live', Path(__file__).resolve().parents[1] / 'live/blender/rgbmocap_live.py')
live = importlib.util.module_from_spec(spec); spec.loader.exec_module(live)
r = live.RECEIVER; r.avatar = live.Avatar(bpy.data.objects['RGBMoCap_Avatar'])
r.backdrop_jpeg = jpg; r.update_backdrop()
plane = bpy.data.objects['RGBMoCap_Backdrop']; image = bpy.data.images['RGBMoCap_Webcam']
print('BACKDROP OK', tuple(image.size), 'plane at', tuple(round(v, 2) for v in plane.location),
      'size', tuple(round(v, 2) for v in plane.dimensions))
r.backdrop_at = 0; r.backdrop_jpeg = jpg; r.update_backdrop()
print('BACKDROP UPDATED', len(bpy.data.objects.keys()), 'objects (no duplicates:', 'RGBMoCap_Backdrop.001' not in bpy.data.objects, ')')
