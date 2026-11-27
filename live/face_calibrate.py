"""Guided face calibration for RGB MoCap Live: learn this performer's expressions.

    .venv-live/bin/python live/face_calibrate.py [--camera 0] [--seconds 3] [--samples FILE]

A window shows one prompt at a time (neutral, smile, sad, angry, surprised, eyes closed, mouth open):
"get ready" for 2 s, then it records the face blendshapes until it has --seconds of frames with a face
(waiting for the performer, up to 30 s per step). The profile is written to
config/face_profile.json (git-ignored personal data) and capture.py uses it automatically from then on.
Prints the accuracy on held-out frames of each expression ("ACCURACY ...").
--samples FILE rebuilds the profile from earlier saved samples without the camera.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import face_profile as FP

SAMPLES = FP.ROOT / 'config' / 'face_samples.json'
WAIT_LIMIT = 30.0  # s a step waits for a face before moving on


def record(camera, seconds, ready=2.0):
    import cv2, mediapipe as mp
    from mediapipe.tasks import python as T
    from mediapipe.tasks.python import vision as V
    model = FP.ROOT / 'models' / 'live' / 'face_landmarker.task'
    face = V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(
        base_options=T.BaseOptions(model_asset_path=str(model)), running_mode=V.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True))
    cap = cv2.VideoCapture(camera, cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY)
    if not cap.isOpened():
        sys.exit('Camera not available (close the live capture first).')
    samples, start = {}, time.monotonic()
    try:
        for label, prompt in FP.STEPS:
            frames, t0, seen, last = [], time.monotonic(), 0.0, None
            print('STEP', label, flush=True)
            while True:
                ok, frame = cap.read()
                if not ok:
                    sys.exit('Camera stopped.')
                clock = time.monotonic(); now = clock - t0
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = face.detect_for_video(image, int((clock - start) * 1000))
                found = bool(result.face_blendshapes)
                recording = now >= ready and found
                # Only time with a face counts: a step waits for the performer instead of timing out empty.
                if recording:
                    seen += clock - last if last is not None else 0.0
                    frames.append({c.category_name: round(float(c.score), 4)
                                   for c in result.face_blendshapes[0] if c.category_name != '_neutral'})
                last = clock if recording else None
                view = cv2.flip(frame, 1)  # mirror: easier to make faces at yourself
                h, w = view.shape[:2]
                cv2.rectangle(view, (0, 0), (w, 90), (0, 0, 0), -1)
                cv2.putText(view, f'{FP.STEPS.index((label, prompt)) + 1}/{len(FP.STEPS)}  {prompt}', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                status = (f'Get ready... {ready - now:.1f} s' if now < ready else
                          f'Recording... {max(0.0, seconds - seen):.1f} s' if found else
                          'Waiting for your face: come closer to the camera')
                colour = (60, 60, 255) if recording else (200, 200, 200)
                cv2.putText(view, status, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
                cv2.imshow('RGB MoCap - face calibration (Esc cancels)', view)
                if cv2.waitKey(1) == 27:
                    sys.exit('Cancelled.')
                if seen >= seconds or now > ready + WAIT_LIMIT:
                    break
            print('RECORDED', label, len(frames), flush=True)
            samples[label] = frames
    finally:
        cap.release(); cv2.destroyAllWindows()
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--seconds', type=float, default=3.0)
    ap.add_argument('--samples', type=Path, help='rebuild from saved samples (no camera)')
    args = ap.parse_args()
    if args.samples:
        samples = json.loads(args.samples.read_text(encoding='utf-8'))
    else:
        samples = record(args.camera, args.seconds)
        SAMPLES.parent.mkdir(parents=True, exist_ok=True)
        SAMPLES.write_text(json.dumps(samples), encoding='utf-8')
    missing = [l for l, _ in FP.STEPS if len(samples.get(l, [])) < 10]
    if 'neutral' in missing:
        sys.exit('Not enough neutral-face frames (face not found?). Run the calibration again.')
    acc, per = FP.evaluate(samples)
    FP.save(FP.build(samples))
    print('ACCURACY', round(acc, 3), json.dumps({k: round(v, 2) for k, v in per.items()}))
    if missing:
        print('WARNING few frames for:', ', '.join(missing))
    print('SAVED', FP.DEFAULT)


if __name__ == '__main__':
    main()
