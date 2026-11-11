"""Export a recorded live take as MP4: webcam, avatar, or side by side.

    .venv/bin/python live/export_take.py TAKE_DIR --kind webcam  --out take.mp4
    .venv/bin/python live/export_take.py TAKE_DIR --kind avatar  --out avatar.mp4 [--quality]
    .venv/bin/python live/export_take.py TAKE_DIR --kind side    --out side.mp4

Uses the take's In/Out trim (take.json) unless --first/--last are given. Video
runs in real time at --fps (frames are picked by timestamp). Avatar frames are
rendered by a separate headless Blender (live/blender/render_take.py), so an
open Blender session is not disturbed. Encodes H.264 (yuv420p) with FFmpeg.
Progress lines: "PROGRESS <0-100> <message>".
"""
from __future__ import annotations

import argparse, bisect, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def progress(value, message):
    print(f'PROGRESS {int(value)} {message}', flush=True)


def blender_path():
    if os.environ.get('RGBMOCAP_BLENDER'):
        return os.environ['RGBMOCAP_BLENDER']
    for folder in (Path('/Applications'), Path.home() / 'Applications'):
        for app in sorted(folder.glob('Blender*.app'), reverse=True):
            exe = app / 'Contents/MacOS/Blender'
            if exe.is_file():
                return str(exe)
    return shutil.which('blender')


def avatar_blend(blender, vrm):
    blend = ROOT / 'build/avatars' / (Path(vrm).stem + '.blend')
    if not blend.exists() or Path(vrm).stat().st_mtime > blend.stat().st_mtime:
        blend.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([blender, '-b', '--factory-startup', '--python-exit-code', '1', '--python',
                        str(ROOT / 'live/blender/prepare_vrm.py'), '--', str(vrm), str(blend)], check=True,
                       stdout=subprocess.DEVNULL)
    return blend


class Encoder:
    def __init__(self, path, width, height, fps):
        self.proc = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
             '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
             '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
        self.size = (width, height)

    def write(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError('FFmpeg failed to encode the video')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('take', type=Path)
    ap.add_argument('--kind', choices=('webcam', 'avatar', 'side'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--first', type=int); ap.add_argument('--last', type=int)
    ap.add_argument('--avatar', default=str(ROOT / 'live/avatars/Seed-san.vrm'))
    ap.add_argument('--quality', action='store_true', help='EEVEE instead of Workbench for the avatar')
    args = ap.parse_args()

    rows = [json.loads(line) for line in (args.take / 'take.jsonl').open(encoding='utf-8')]
    meta = json.loads((args.take / 'take.json').read_text()) if (args.take / 'take.json').exists() else {}
    first = args.first if args.first is not None else int(meta.get('in', 0))
    last = args.last if args.last is not None else int(meta.get('out', len(rows) - 1))
    rows = rows[first:last + 1]
    times = [r['t'] - rows[0]['t'] for r in rows]
    count = int(times[-1] * args.fps) + 1
    frames = (args.take / 'frames.bin').open('rb')

    def webcam(k):
        i = max(0, bisect.bisect_right(times, k / args.fps) - 1)
        offset, length = rows[i]['jpg']; frames.seek(offset)
        return cv2.imdecode(np.frombuffer(frames.read(length), np.uint8), cv2.IMREAD_COLOR)

    sample = webcam(0); cam_h, cam_w = sample.shape[:2]
    work = Path(tempfile.mkdtemp(prefix='rgbmocap-export-'))
    try:
        avatar_dir = None
        if args.kind in ('avatar', 'side'):
            blender = blender_path()
            if not blender:
                sys.exit('Blender was not found; set RGBMOCAP_BLENDER')
            progress(1, 'Preparing avatar')
            blend = avatar_blend(blender, args.avatar)
            avatar_dir = work / 'avatar'
            # Side by side: avatar panel matches the webcam height; alone: a 9:16-ish portrait panel.
            aw, ah = (int(cam_h * 0.75) // 2 * 2, cam_h) if args.kind == 'side' else (720, 1280)
            cmd = [blender, '-b', str(blend), '--python-exit-code', '1', '--python', str(ROOT / 'live/blender/render_take.py'),
                   '--', str(args.take / 'take.jsonl'), str(first), str(last), str(args.fps), str(avatar_dir), str(aw), str(ah),
                   'quality' if args.quality else 'fast']
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                if line.startswith('RENDER ') and '/' in line:
                    done, total = map(int, line.split()[1].split('/'))
                    progress(5 + 75 * done / total, f'Rendering avatar {done}/{total}')
            if proc.wait() != 0:
                raise RuntimeError('Blender could not render the avatar')
        if args.kind == 'webcam':
            size = (cam_w, cam_h)
        elif args.kind == 'avatar':
            size = cv2.imread(str(avatar_dir / '00000.png')).shape[1::-1]
        else:
            av = cv2.imread(str(avatar_dir / '00000.png'))
            size = (cam_w + av.shape[1], cam_h)
        size = (size[0] // 2 * 2, size[1] // 2 * 2)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        encoder = Encoder(args.out, size[0], size[1], args.fps)
        for k in range(count):
            if args.kind == 'webcam':
                frame = webcam(k)
            else:
                avatar = cv2.imread(str(avatar_dir / f'{k:05d}.png'))
                frame = avatar if args.kind == 'avatar' else np.hstack([webcam(k), avatar])
            encoder.write(frame)
            if k % 10 == 0:
                progress((80 if avatar_dir else 0) + (20 if avatar_dir else 100) * k / count, f'Encoding {k}/{count}')
        encoder.close()
        progress(100, f'Saved {args.out}')
        print(json.dumps({'out': str(args.out), 'frames': count, 'seconds': round(count / args.fps, 2),
                          'size': list(size)}), flush=True)
    finally:
        frames.close()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
