"""RGB MoCap Live: webcam (or video) -> face expressions, body and hands -> Blender.

    .venv-live/bin/python live/capture.py                 # Mac webcam 0, preview window
    .venv-live/bin/python live/capture.py --camera 1      # another camera
    .venv-live/bin/python live/capture.py --video clip.mp4 --no-preview --frames 300

Per frame, MediaPipe (Apache-2.0) gives 52 ARKit face blendshapes, 33 metric body
landmarks and 21 landmarks per hand. They are smoothed with a One Euro filter and
sent as one JSON datagram to the Blender receiver (live/blender/rgbmocap_live.py),
by default on udp://127.0.0.1:49170.

Coordinates are converted from MediaPipe's camera frame (x right, y down, z away)
to Blender's (character facing -Y, Z up): (x, z, -y).
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models' / 'live'
PORT = 49170
POSE_WRISTS = {'left': 15, 'right': 16}
STABLE_FRAMES = 6  # ~0.5 s at webcam rates before a newly seen landmark is trusted
LEGS = range(23, 33)


def landmark_valid(index, visibility, x, y):
    """Legs get a strict rule (MediaPipe invents legs outside the picture); the rest a normal one."""
    if index in LEGS:
        return visibility > 0.75 and 0.03 < x < 0.97 and 0.03 < y < 0.97
    return visibility > 0.5 and 0.0 < x < 1.0 and 0.0 < y < 1.0


class OneEuro:
    """One Euro filter (Casiez et al. 2012) on arrays: smooth when still, responsive when fast."""

    def __init__(self, min_cutoff=1.2, beta=0.03, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, np.float64)
        if self.x is None or self.x.shape != x.shape:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(1e-3, t - self.t)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


def to_blender(points):
    p = np.asarray(points, np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def make_detectors(running_mode):
    base = lambda name: mp_tasks.BaseOptions(model_asset_path=str(MODELS / name))
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=base('face_landmarker.task'), running_mode=running_mode, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=False))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=base('pose_landmarker_full.task'), running_mode=running_mode, num_poses=1))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=base('hand_landmarker.task'), running_mode=running_mode, num_hands=2))
    return face, pose, hands


class Tracker:
    def __init__(self, smooth=True):
        self.face, self.pose, self.hands = make_detectors(vision.RunningMode.VIDEO)
        self.smooth = smooth
        self.filters = {}
        self.streak = None
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(3)

    def filt(self, key, value, t, **kw):
        if not self.smooth:
            return np.asarray(value, np.float64)
        if key not in self.filters:
            self.filters[key] = OneEuro(**kw)
        return self.filters[key](value, t)

    def __call__(self, rgb, t_ms):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t = t_ms / 1000.0
        packet = {'t': t}
        # The three models are independent: run them at the same time (MediaPipe releases the GIL).
        jobs = [self.pool.submit(d.detect_for_video, image, t_ms) for d in (self.face, self.pose, self.hands)]
        face, pose, hands = (j.result() for j in jobs)
        if face.face_blendshapes:
            names = [c.category_name for c in face.face_blendshapes[0]]
            scores = self.filt('face', [c.score for c in face.face_blendshapes[0]], t, min_cutoff=2.0, beta=0.5)
            packet['face'] = {n: round(float(s), 4) for n, s in zip(names, scores) if n != '_neutral'}
        pose_image = None
        if pose.pose_world_landmarks:
            world = [[l.x, l.y, l.z] for l in pose.pose_world_landmarks[0]]
            vis = [l.visibility for l in pose.pose_world_landmarks[0]]
            pose_image = np.array([[l.x, l.y] for l in pose.pose_landmarks[0]])
            packet['pose'] = np.round(self.filt('pose', to_blender(world), t), 4).tolist()
            packet['pose_visibility'] = np.round(vis, 3).tolist()
            # A landmark drives the avatar only when MediaPipe is confident, it is clearly inside
            # the picture, and that has held for STABLE_FRAMES frames; off-frame legs are
            # otherwise hallucinated and flicker in at the image edge. Losing it is immediate.
            image_lms = pose.pose_landmarks[0]
            raw = [landmark_valid(i, l.visibility, l.x, l.y) for i, l in enumerate(image_lms)]
            packet['pose_image'] = [[round(l.x, 3), round(l.y, 3)] for l in image_lms]
            if self.streak is None or len(self.streak) != len(raw):
                self.streak = [0] * len(raw)
            self.streak = [s + 1 if r else 0 for s, r in zip(self.streak, raw)]
            packet['pose_valid'] = [s >= STABLE_FRAMES for s in self.streak]
        if hands.hand_world_landmarks:
            packet['hands'] = {}
            for image_points, world_points in zip(hands.hand_landmarks, hands.hand_world_landmarks):
                side = self.side_of(np.array([image_points[0].x, image_points[0].y]), pose_image)
                if side is None or side in packet['hands']:
                    continue
                pts = to_blender([[l.x, l.y, l.z] for l in world_points])
                packet['hands'][side] = np.round(self.filt('hand_' + side, pts, t), 4).tolist()
        return packet, (face, pose, hands)

    @staticmethod
    def side_of(wrist, pose_image):
        """Assign a hand to the person's left/right by the nearest pose wrist.

        MediaPipe's handedness label assumes a mirrored selfie image; matching to
        the body is correct for any camera orientation.
        """
        if pose_image is None:
            return None
        distances = {s: float(np.linalg.norm(pose_image[i] - wrist)) for s, i in POSE_WRISTS.items()}
        side = min(distances, key=distances.get)
        return side if distances[side] < 0.15 else None


class LatestFrame:
    """Read the camera on its own thread and keep only the newest frame, so tracking never
    waits a whole camera interval (synchronous reads halve the rate when work > 1 frame)."""

    def __init__(self, cap):
        import threading
        self.cap, self.frame, self.id, self.running = cap, None, 0, True
        self.ready = threading.Condition()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            with self.ready:
                self.frame = frame if ok else None
                self.id += 1
                self.running = ok
                self.ready.notify_all()

    def read(self, last_id):
        with self.ready:
            self.ready.wait_for(lambda: self.id != last_id or not self.running)
            return self.frame is not None, self.frame, self.id


def draw(frame, results, fps, caption=None):
    face, pose, hands = results
    h, w = frame.shape[:2]
    pts = lambda lms: [(int(l.x * w), int(l.y * h)) for l in lms]
    if pose.pose_landmarks:
        p = pts(pose.pose_landmarks[0])
        for a, b in ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
                     (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)):
            cv2.line(frame, p[a], p[b], (80, 220, 80), 2)
    for hand in hands.hand_landmarks or []:
        for x, y in pts(hand):
            cv2.circle(frame, (x, y), 2, (240, 180, 60), -1)
    if face.face_landmarks:
        for x, y in pts(face.face_landmarks[0])[::6]:
            cv2.circle(frame, (x, y), 1, (200, 200, 255), -1)
    cv2.putText(frame, caption or f'{fps:4.1f} fps  ->  Blender udp:{PORT}   [q] quit', (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--video', type=Path)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--app', action='store_true',
                    help='RGB MoCap app mode: send b"P"+packet and b"F"+JPEG preview datagrams to --port '
                         '(the app forwards to Blender, records and plays back)')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--no-preview', action='store_true')
    ap.add_argument('--no-smooth', action='store_true')
    ap.add_argument('--track', choices=('full', 'upper', 'head'), default='full',
                    help='body parts that drive the avatar (upper: legs still; head: head and face only)')
    ap.add_argument('--no-overlay', action='store_true', help='app mode: send the plain camera image')
    ap.add_argument('--frames', type=int, default=0, help='stop after N frames (testing)')
    ap.add_argument('--snapshots', type=Path, help='save an annotated camera frame here every second (diagnostics)')
    ap.add_argument('--crop', help='x,y,w,h region of the frame to track (e.g. a screen recording)')
    ap.add_argument('--dump', type=Path, help='also write every packet to this JSON-lines file')
    args = ap.parse_args()

    if args.video:
        cap = cv2.VideoCapture(str(args.video))
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if not cap.isOpened():
        sys.exit('Cannot open the camera. On macOS, allow camera access for your terminal app in '
                 'System Settings > Privacy & Security > Camera, then retry.')
    tracker = Tracker(smooth=not args.no_smooth)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # macOS limits a datagram to the send buffer (9 KB by default); previews need ~40 KB.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    dump = args.dump.open('w') if args.dump else None
    start = time.monotonic(); count = 0; fps = 0.0; last = start; last_snapshot = start
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    camera = None if args.video else LatestFrame(cap); frame_id = 0
    print(f'Streaming to udp://{args.host}:{args.port}. In Blender: N panel > RGB MoCap Live > Start.', flush=True)
    try:
        while True:
            if camera:
                ok, frame, frame_id = camera.read(frame_id)
            else:
                ok, frame = cap.read()
            if not ok:
                break
            if args.crop:
                x, y, w, h = map(int, args.crop.split(','))
                frame = np.ascontiguousarray(frame[y:y + h, x:x + w])
            t_ms = int(count * 1000 / video_fps) if args.video else int((time.monotonic() - start) * 1000)
            packet, results = tracker(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), t_ms)
            packet['frame'] = count
            if args.track != 'full':
                from track_modes import apply_mode
                packet = apply_mode(packet, args.track)
            data = json.dumps(packet, separators=(',', ':')).encode()
            if args.app:
                sock.sendto(b'P' + data, (args.host, args.port))
                shown = draw(frame.copy(), results, fps, f'{fps:4.1f} fps') if not args.no_overlay else frame
                for width, quality in ((960, 75), (640, 70)):  # largest that fits one datagram
                    small = cv2.resize(shown, (width, int(frame.shape[0] * width / frame.shape[1])))
                    ok_jpg, jpg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if ok_jpg and len(jpg) < 63000:
                        break
                if ok_jpg and len(jpg) < 63000:
                    try:
                        sock.sendto(b'F' + jpg.tobytes(), (args.host, args.port))
                    except OSError:
                        pass  # a dropped preview frame is harmless
            else:
                sock.sendto(data, (args.host, args.port))
            if dump:
                dump.write(data.decode() + '\n')
            count += 1
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(1e-3, now - last) if count > 1 else 0.0
            last = now
            if args.snapshots and (count == 1 or int(now) != int(last_snapshot)):
                last_snapshot = now
                args.snapshots.mkdir(parents=True, exist_ok=True)
                stamp = time.time()
                cv2.imwrite(str(args.snapshots / f'camera_{stamp:.0f}.jpg'), draw(frame.copy(), results, fps))
                (args.snapshots / f'camera_{stamp:.0f}.json').write_text(json.dumps(packet))
            if not args.no_preview:
                small = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))  # cheap preview
                cv2.imshow('RGB MoCap Live', draw(small, results, fps))
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
            if args.frames and count >= args.frames:
                break
    finally:
        if camera:
            camera.running = False
        cap.release(); cv2.destroyAllWindows()
        if dump:
            dump.close()
    elapsed = time.monotonic() - start
    print(json.dumps(dict(frames=count, seconds=round(elapsed, 1), fps=round(count / max(elapsed, 1e-6), 1))), flush=True)


if __name__ == '__main__':
    main()
