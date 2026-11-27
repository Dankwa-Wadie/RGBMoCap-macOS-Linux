// Paint every visor face state the live receiver can pick (expr x blink x talk x gaze) as PNGs in ./visor_faces.
const P = '/Volumes/Samsung_ssd/3js_ps2';
const { PAINTERS } = await import(P + '/src/kit.js');
const spec = (await import(P + '/examples/visor.js')).default;
const zlib = await import('zlib'), fs = await import('fs');
const { garment, y0, y1, r0, r1, rows, z0, arc, seg, bone, ...paint } = spec.visor; // painter options only
const crcT = Array.from({ length: 256 }, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
const crc = b => { let c = ~0; for (const x of b) c = crcT[(c ^ x) & 255] ^ (c >>> 8); return ~c >>> 0; };
const chunk = (t, d) => { const b = Buffer.concat([Buffer.from(t), d]), o = Buffer.alloc(d.length + 12); o.writeUInt32BE(d.length); b.copy(o, 4); o.writeUInt32BE(crc(b), d.length + 8); return o; };
const png = (px, N) => { const raw = Buffer.alloc((N * 4 + 1) * N); for (let y = 0; y < N; y++) Buffer.from(px.buffer, y * N * 4, N * 4).copy(raw, y * (N * 4 + 1) + 1);
  const h = Buffer.alloc(13); h.writeUInt32BE(N); h.writeUInt32BE(N, 4); h.set([8, 6, 0, 0, 0], 8);
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', h), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]); };
fs.mkdirSync('visor_faces', { recursive: true }); let n = 0;
for (const expr of ['neutral', 'happy', 'sad', 'angry', 'surprised']) for (const blink of [0, 1]) for (const talk of [0, 0.5, 1]) for (const gx of [-1, 0, 1]) {
  const { px } = PAINTERS.visor({ ...paint, expr, blink, talk, gaze: [gx * 0.6, 0] });
  fs.writeFileSync(`visor_faces/${expr}_${blink}_${talk}_${gx}.png`, png(px, 64)); n++;
}
console.log('painted', n);
