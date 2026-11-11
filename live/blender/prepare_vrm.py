"""Blender (headless): turn a VRM 1.0 avatar into a .blend ready for RGB MoCap Live.

    Blender -b --factory-startup --python live/blender/prepare_vrm.py -- AVATAR.vrm OUT.blend

VRM is glTF binary, so Blender's built-in glTF importer loads the meshes, rig and
morph targets. The VRM extension (humanoid bone map, expression presets) is read
from the file and stored on the armature as JSON, so the live receiver needs no
VRM add-on.
"""
import json, struct, sys, shutil, tempfile
from pathlib import Path
import bpy

src, out = map(Path, sys.argv[sys.argv.index('--') + 1:][:2])
data = src.read_bytes()
length = struct.unpack_from('<I', data, 12)[0]
gltf = json.loads(data[20:20 + length])
vrm = gltf['extensions']['VRMC_vrm']
nodes = gltf['nodes']; meshes = gltf.get('meshes', [])

bpy.ops.wm.read_factory_settings(use_empty=True)
glb = Path(tempfile.mkdtemp()) / (src.stem + '.glb'); shutil.copy(src, glb)
bpy.ops.import_scene.gltf(filepath=str(glb), import_shading='NORMALS', bone_heuristic='TEMPERANCE')
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')

# humanoid bone -> Blender bone name (glTF node names become bone names)
human = {name: nodes[spec['node']].get('name') for name, spec in vrm['humanoid']['humanBones'].items()}
missing = [h for h, b in human.items() if b not in arm.data.bones]
# expression -> [(object name, shape key name, weight)]
def mesh_objects_for(node_index):
    node = nodes[node_index]
    return [o for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith(node.get('name', ''))]
expressions = {}
for group in ('preset', 'custom'):
    for name, expr in vrm.get('expressions', {}).get(group, {}).items():
        binds = []
        for b in expr.get('morphTargetBinds', []):
            mesh = meshes[nodes[b['node']]['mesh']]
            targets = mesh.get('extras', {}).get('targetNames') or []
            key_name = targets[b['index']] if b['index'] < len(targets) else None
            for o in mesh_objects_for(b['node']):
                keys = o.data.shape_keys.key_blocks if o.data.shape_keys else []
                kb = keys.get(key_name) if key_name else (keys[b['index'] + 1] if b['index'] + 1 < len(keys) else None)
                if kb: binds.append([o.name, kb.name, float(b.get('weight', 1.0))])
        expressions[name] = binds
arm['rgbmocap_humanoid'] = json.dumps(human)
arm['rgbmocap_expressions'] = json.dumps(expressions)
meta = vrm.get('meta', {})
arm['rgbmocap_credit'] = f"{meta.get('name')} by {', '.join(meta.get('authors', []))} ({meta.get('licenseUrl')})"
arm.name = 'RGBMoCap_Avatar'
bpy.ops.wm.save_as_mainfile(filepath=str(out))
print('PREPARED', out, '| bones', len(arm.data.bones), '| humanoid', len(human), 'missing', missing,
      '| expressions', {k: len(v) for k, v in expressions.items()})
