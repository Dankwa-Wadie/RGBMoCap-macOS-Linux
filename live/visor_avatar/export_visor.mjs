// Node: build the visor character and write a glTF (bones renamed to Mixamo names).
const P = '/Volumes/Samsung_ssd/3js_ps2';
const THREE = await import(P + '/node_modules/three/build/three.module.js');
const { GLTFExporter } = await import(P + '/node_modules/three/examples/jsm/exporters/GLTFExporter.js');
const { build } = await import(P + '/src/kit.js');
const spec = (await import(P + '/examples/visor.js')).default;
// shortcut: minimal FileReader for GLTFExporter's buffer->dataURI path in Node
globalThis.FileReader = class { readAsDataURL(b) { b.arrayBuffer().then(a => { this.result = 'data:application/octet-stream;base64,' + Buffer.from(a).toString('base64'); this.onloadend?.(); }); }
  readAsArrayBuffer(b) { b.arrayBuffer().then(a => { this.result = a; this.onloadend?.(); }); } };
const { mesh, bones } = build(spec);
// Visor screen: in three it copies the head bone's matrix at draw time; for export parent it to the head bone for real.
const screen = mesh.getObjectByName('visor');
screen.removeFromParent(); screen.onBeforeRender = () => {}; screen.matrixAutoUpdate = screen.matrixWorldAutoUpdate = true;
screen.position.set(0, 0, 0); bones.head.add(screen);
// Its painted face (DataTexture) -> PNG, rows as-is (the glTF importer flips V).
const { data, width, height } = screen.material.map.image, zlib = await import('zlib'), fs = await import('fs');
const raw = Buffer.alloc((width * 4 + 1) * height);
for (let y = 0; y < height; y++) { raw[y * (width * 4 + 1)] = 0; Buffer.from(data.buffer, data.byteOffset + y * width * 4, width * 4).copy(raw, y * (width * 4 + 1) + 1); }
const crcT = Array.from({ length: 256 }, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
const crc = b => { let c = ~0; for (const x of b) c = crcT[(c ^ x) & 255] ^ (c >>> 8); return ~c >>> 0; };
const chunk = (t, d) => { const b = Buffer.concat([Buffer.from(t), d]), o = Buffer.alloc(d.length + 12); o.writeUInt32BE(d.length); b.copy(o, 4); o.writeUInt32BE(crc(b), d.length + 8); return o; };
const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(width); ihdr.writeUInt32BE(height, 4); ihdr.set([8, 6, 0, 0, 0], 8);
fs.writeFileSync('visor_screen.png', Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]));
screen.material.name = 'visor_screen';
// Robot buddy accessory: not part of the body; left out of the mocap export.
mesh.userData.garments.buddy.removeFromParent();
const M = { hips:'Hips', spine:'Spine', spine1:'Spine1', chest:'Spine2', neck:'Neck', head:'Head' };
for (const s of ['L','R']) { const S = s==='L'?'Left':'Right';
  Object.assign(M, {[`shoulder.${s}`]:S+'Shoulder',[`toe.${s}`]:S+'ToeBase',[`upperArm.${s}`]:S+'Arm',[`forearm.${s}`]:S+'ForeArm',[`hand.${s}`]:S+'Hand',[`thigh.${s}`]:S+'UpLeg',[`shin.${s}`]:S+'Leg',[`foot.${s}`]:S+'Foot'}); }
for (const b of mesh.skeleton.bones) if (M[b.name]) b.name = M[b.name];
const meshes = []; mesh.traverse(o => { if (o.isMesh) meshes.push(o.type + ':' + o.name + (o.material?.map ? '(tex)' : '')); });
console.log('bones', mesh.skeleton.bones.map(b => b.name).join(' ')); console.log('meshes', meshes.join(' '));
// shortcut: textures (canvas-painted face/visor) dropped in Node; colours stay
mesh.traverse(o => { if (o.isMesh) for (const m of [o.material].flat()) m.map = null; });
const scene = new THREE.Scene(); scene.add(mesh);
const json = await new GLTFExporter().parseAsync(scene, { binary: false, onlyVisible: false });
(await import('fs')).writeFileSync('visor.gltf', JSON.stringify(json));
console.log('ok');
