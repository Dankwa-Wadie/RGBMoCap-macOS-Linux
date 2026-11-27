# Blender: visor glTF -> live avatar .blend (rgbmocap_humanoid map, as prepare_vrm.py writes for a VRM).
import bpy, json, sys, os
src, out = sys.argv[sys.argv.index('--') + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src, import_shading='NORMALS', bone_heuristic='TEMPERANCE')
def aim_bones(arm):
    """glTF joints have no tails; the importer points many bones straight down. Point each at its child
    (live aiming and retargeting use the bone axis as the limb direction); leaves continue their parent."""
    prefer = {'LeftHand': 'middle.L', 'RightHand': 'middle.R', 'Hips': 'Spine'}
    bpy.context.view_layer.objects.active = arm; bpy.ops.object.mode_set(mode='EDIT')
    E = arm.data.edit_bones
    for b in sorted(E, key=lambda b: len(b.parent_recursive)):
        kids = b.children
        child = E.get(prefer.get(b.name, '')) or (kids[0] if len(kids) == 1 else None)
        if child: d = child.head - b.head
        elif b.parent: d = (b.parent.tail - b.parent.head).normalized() * min(b.length, 0.05)
        else: continue
        if d.length > 1e-4: b.tail = b.head + d
    bpy.ops.object.mode_set(mode='OBJECT')


def screen_material(d):
    m = bpy.data.materials.get('visor_screen')
    if not m: print('NO SCREEN MATERIAL'); return
    m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = bpy.data.images.load(d + '/visor_screen.png'); tex.interpolation = 'Closest'
    em = nt.nodes.new('ShaderNodeEmission'); out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(tex.outputs['Color'], em.inputs['Color']); nt.links.new(em.outputs[0], out.inputs['Surface'])
    tex.image.pack()
screen_material(os.path.dirname(os.path.abspath(sys.argv[sys.argv.index('--') + 1])) if 'prepare_visor.py'=='prepare_visor.py' else sys.argv[sys.argv.index('--') + 1])
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
h = {'hips': 'Hips', 'spine': 'Spine', 'chest': 'Spine1', 'upperChest': 'Spine2', 'neck': 'Neck', 'head': 'Head'}
for s, S, k in (('left', 'Left', 'L'), ('right', 'Right', 'R')):
    h.update({s+'Shoulder': S+'Shoulder', s+'UpperArm': S+'Arm', s+'LowerArm': S+'ForeArm', s+'Hand': S+'Hand',
              s+'UpperLeg': S+'UpLeg', s+'LowerLeg': S+'Leg', s+'Foot': S+'Foot', s+'Toes': S+'ToeBase',
              s+'ThumbMetacarpal': f'thumb.{k}', s+'ThumbProximal': f'thumb2.{k}'})
    for f in ('index', 'middle', 'ring', 'little'):
        F = f.capitalize(); h[s+F+'Proximal'] = f'{f}.{k}'; h[s+F+'Intermediate'] = f'{f}2.{k}'
aim_bones(arm)
missing = [b for b in h.values() if b not in arm.data.bones]
# Live face: every visor face state, packed; the receiver swaps the screen image to match the performer's face.
import glob
faces = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(src)), 'visor_faces', '*.png'))):
    im = bpy.data.images.load(f); im.name = 'visor_face:' + os.path.basename(f)[:-4]; im.pack(); im.use_fake_user = True; faces.append(im.name)
arm['rgbmocap_visor'] = json.dumps({'material': 'visor_screen', 'faces': len(faces)})
# Dual-quaternion skinning: a lifted thigh or a bent waist keeps its volume (linear skinning pinches the hip).
for o in bpy.data.objects:
    for m in o.modifiers:
        if m.type == 'ARMATURE': m.use_deform_preserve_volume = True
arm['rgbmocap_humanoid'] = json.dumps(h); arm['rgbmocap_expressions'] = '{}'
arm['rgbmocap_credit'] = 'visor (3js_ps2)'; arm.name = 'RGBMoCap_Avatar'
bpy.ops.wm.save_as_mainfile(filepath=out)
print('PREPARED', out, 'faces', len(faces), 'humanoid', len(h), 'missing', missing)
