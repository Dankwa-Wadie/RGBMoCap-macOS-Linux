# Headless Blender: drive the live avatar with synthetic MediaPipe poses (avatar frame, metres) and render each.
import bpy, importlib.util, sys, os
from mathutils import Vector
spec = importlib.util.spec_from_file_location('l', os.path.join(os.getcwd(), 'live/blender/rgbmocap_live.py')); l = importlib.util.module_from_spec(spec); spec.loader.exec_module(l)
out = sys.argv[sys.argv.index('--') + 1]  # usage: Blender -b avatar.blend --python tests_macos/live_pose_check.py -- OUT_DIR
arm = next(o for o in bpy.data.objects if 'rgbmocap_humanoid' in o)
def body(**kw):
    P = {0: (0, -0.1, 1.62), 7: (0.075, 0, 1.6), 8: (-0.075, 0, 1.6), 11: (0.18, 0, 1.42), 12: (-0.18, 0, 1.42),
         23: (0.1, 0, 0.95), 24: (-0.1, 0, 0.95)}
    P.update(kw.get('P', {}))
    pose = [list(P.get(i, (0, 0, 0))) for i in range(33)]
    valid = [i in P and i < 23 or i in (23, 24) for i in range(33)]
    img = [[0.5, 0.5]] * 33
    for i in kw.get('off', ()): valid[i] = False; img[i] = [0.5, 1.1]
    return {'pose': pose, 'pose_valid': valid, 'pose_image': img}
CASES = {
 'arms_down': {13: (0.2, 0, 1.15), 14: (-0.2, 0, 1.15), 15: (0.21, 0, 0.9), 16: (-0.21, 0, 0.9)},
 'hand_face': {13: (0.2, 0, 1.15), 15: (0.21, 0, 0.9), 14: (-0.26, -0.12, 1.25), 16: (-0.07, -0.13, 1.56)},
 'hands_hips': {13: (0.32, 0.02, 1.15), 14: (-0.32, 0.02, 1.15), 15: (0.14, -0.02, 0.98), 16: (-0.14, -0.02, 0.98)},
 'arms_crossed': {13: (0.22, -0.12, 1.15), 14: (-0.22, -0.12, 1.15), 15: (-0.12, -0.18, 1.25), 16: (0.12, -0.2, 1.28)},
}
OFF = {'off_frame': (13, 14, 15, 16)}
CASES['off_frame'] = {13: (0.5, 0, 1.4), 14: (-0.5, 0, 1.4)}  # garbage positions; they are off frame
s = bpy.context.scene
cam = bpy.data.objects.new('C', bpy.data.cameras.new('C')); s.collection.objects.link(cam); s.camera = cam
cam.location = (0.8, -2.6, 1.3); cam.rotation_euler = (1.5708, 0, 0.3); cam.data.lens = 40
s.render.resolution_x, s.render.resolution_y = 360, 480; s.render.engine = 'BLENDER_WORKBENCH'
s.display.shading.light = 'STUDIO'; s.display.shading.color_type = 'MATERIAL'
for name, P in CASES.items():
    a = l.Avatar(arm)
    for _ in range(12): a.apply(body(P=P, off=OFF.get(name, ())))
    bpy.context.view_layer.update()
    s.render.filepath = f'{out}/pose_{name}.png'; bpy.ops.render.render(write_still=True)
    print('POSED', name)
