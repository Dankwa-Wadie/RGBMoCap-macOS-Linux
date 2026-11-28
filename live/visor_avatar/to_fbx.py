import bpy, sys, os
d = sys.argv[sys.argv.index('--') + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=d + '/visor.gltf')
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
screen_material(os.path.dirname(os.path.abspath(sys.argv[sys.argv.index('--') + 1])) if 'to_fbx.py'=='prepare_visor.py' else sys.argv[sys.argv.index('--') + 1])
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
aim_bones(arm)
print('BONES', [b.name for b in arm.data.bones])
# Face screen: parented to the Head bone in glTF; the app imports skinned meshes only, so skin it 100 % to Head.
for o in [o for o in bpy.data.objects if o.type == 'MESH' and o.parent == arm and o.parent_type == 'BONE']:
    world = o.matrix_world.copy(); bone = o.parent_bone
    o.parent_type = 'OBJECT'; o.parent_bone = ''; o.matrix_world = world
    group = o.vertex_groups.new(name=bone); group.add(range(len(o.data.vertices)), 1.0, 'REPLACE')
    o.modifiers.new('Armature', 'ARMATURE').object = arm
    print('SKINNED', o.name, 'to', bone)
# Export only the character: the armature and the meshes it deforms (nothing a startup add-on may have added).
keep = [arm] + [o for o in bpy.data.objects if o.type == 'MESH' and o.parent == arm]
bpy.ops.object.select_all(action='DESELECT')
for o in keep: o.select_set(True)
bpy.context.view_layer.objects.active = arm
print('EXPORT', [o.name for o in keep])
bpy.ops.export_scene.fbx(filepath=d + '/visor_mixamo.fbx', use_selection=True, add_leaf_bones=False, bake_anim=False)
