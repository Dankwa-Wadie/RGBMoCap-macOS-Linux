"""Blender headless: render the avatar inside the imported room splat from two viewpoints."""
import sys, math
from pathlib import Path
import bpy
out = Path(sys.argv[sys.argv.index('--') + 1])
scene = bpy.context.scene
engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items]
scene.render.engine = next((e for e in engines if 'EEVEE' in e), 'BLENDER_WORKBENCH')
scene.render.resolution_x, scene.render.resolution_y = 960, 640
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); scene.collection.objects.link(cam); scene.camera = cam
cam.data.lens = 24
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN')); sun.rotation_euler = (0.8, 0, 0.4); scene.collection.objects.link(sun)
for name, loc, rot, ortho in (('front', (0, -1.3, 1.2), (math.radians(88), 0, 0), False),
                              ('top', (0, 0, 6), (0, 0, 0), True)):
    cam.location = loc; cam.rotation_euler = rot
    cam.data.type = 'ORTHO' if ortho else 'PERSP'; cam.data.ortho_scale = 6
    scene.render.filepath = str(out / f'room_{name}.png'); bpy.ops.render.render(write_still=True)
print('ROOM RENDERED', scene.render.engine)
