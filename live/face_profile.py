"""Personal face profile: MediaPipe's face blendshape scores rescaled to one performer's face, plus a small
nearest-centroid expression classifier trained on that performer's own faces.

People differ a lot: one person's smile scores 0.5, another's 0.9; a resting face can read 0.3 "frown".
A short calibration (live/face_calibrate.py) records a few seconds of each expression; from that:

  neutral[name]  resting value          (median over the neutral recording)
  top[name]      that person's maximum   (95th percentile over all recordings)
  normalize      (v - neutral) / (top - neutral), clamped to 0..1, so every avatar gets the full range
  centroids      mean normalized vector per expression; classify = nearest centroid with a confidence

The profile is personal data: it stays in config/ (git-ignored), never in projects or recordings' metadata.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'config' / 'face_profile.json'
EXPRESSIONS = ('neutral', 'happy', 'sad', 'angry', 'surprised')   # what the classifier picks between
STEPS = (  # calibration prompts: label, instruction
    ('neutral', 'Relax: neutral face, look at the camera'),
    ('happy', 'Smile (big, happy)'),
    ('sad', 'Sad face: frown, corners of the mouth down'),
    ('angry', 'Angry face: brows down, tense'),
    ('surprised', 'Surprised: brows up, eyes wide'),
    ('blink', 'Close both eyes'),
    ('talk', 'Open your mouth wide (say "aah")'),
)
MIN_SPAN = 0.08
SIZE_WEIGHT = 0.9  # 0 = every expression equal; 1 = scores scale with the expression's size
OFF_WEIGHT = 0.15  # how much movement unlike the expression counts against it
STRENGTH = 0.2  # an expression shows once the face is this far (0..1) from neutral toward it
# Gaze and blinks say nothing about the emotion (a glance during the "sad" step must not make looking
# left read as sad): the classifier ignores them. They are still rescaled for the avatars.
IGNORE = ('eyeLook', 'eyeBlink')
features = lambda names: [n for n in names if not n.startswith(IGNORE)]  # a blendshape that barely moves for this person keeps a sane scale


def percentile(values, q):
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def build(samples):
    """samples: {label: [ {blendshape: score}, ... ]} -> profile dict."""
    names = sorted({n for frames in samples.values() for f in frames for n in f})
    neutral = {n: percentile([f.get(n, 0.0) for f in samples.get('neutral', [])], 0.5) for n in names}
    every = [f for frames in samples.values() for f in frames]
    top = {n: max(neutral[n] + MIN_SPAN, percentile([f.get(n, 0.0) for f in every], 0.95)) for n in names}
    profile = {'version': 1, 'names': names, 'neutral': neutral, 'top': top}
    profile['features'] = features(names)
    profile['centroids'] = {label: mean_vector([normalize(f, profile) for f in samples[label]], profile['features'])
                            for label in EXPRESSIONS if samples.get(label)}
    return profile


def mean_vector(faces, names):
    return [sum(f.get(n, 0.0) for f in faces) / max(1, len(faces)) for n in names]


def normalize(face, profile):
    n0, top = profile['neutral'], profile['top']
    out = {}
    for name, v in face.items():
        if name in n0:
            out[name] = round(min(1.0, max(0.0, (v - n0[name]) / max(1e-6, top[name] - n0[name]))), 4)
        else:
            out[name] = v
    return out


def classify(face_norm, profile):
    """-> (label, confidence 0..1). Each expression is a direction away from this performer's neutral face;
    the score is how far along it the face has moved (0 = neutral, 1 = the calibrated full expression). So a
    natural, half-strength smile still reads as happy (nearest-centroid called it neutral: 7 % on a fresh round).
    Below STRENGTH on every direction -> neutral."""
    names, cents = profile.get('features') or profile['names'], profile.get('centroids') or {}
    if 'neutral' not in cents:
        return None, 0.0
    x = [face_norm.get(n, 0.0) for n in names]
    n0 = cents['neutral']
    # one yardstick for sideways movement: the typical neutral->expression distance (a subtle expression's
    # short direction would otherwise turn ordinary noise into a huge penalty)
    spans = sorted(math.dist(c, n0) for l, c in cents.items() if l != 'neutral')
    ref = spans[len(spans) // 2] if spans else 1.0
    best, score = 'neutral', 0.0
    for label, c in cents.items():
        if label == 'neutral':
            continue
        d = [ci - ni for ci, ni in zip(c, n0)]
        dd = sum(v * v for v in d)
        if dd < 1e-9:
            continue
        along = sum((xi - ni) * di for xi, ni, di in zip(x, n0, d)) / dd   # fraction of the way to `label`
        off = math.dist([xi - ni for xi, ni in zip(x, n0)], [along * di for di in d]) / ref  # sideways
        # capped, and weighted by how big a change the expression is (SIZE_WEIGHT): a subtle expression's short
        # direction must not win by sheer distance (a weak angry face read as the performer's subtle sad)
        size = min(1.0, math.sqrt(dd) / ref) ** SIZE_WEIGHT
        s_ = min(along, 1.2) * size - OFF_WEIGHT * off
        if s_ > score:
            best, score = label, s_
    if score < STRENGTH:
        return 'neutral', round(min(1.0, 1 - score / STRENGTH), 3)
    return best, round(min(1.0, score), 3)


def evaluate(samples):
    """Train on the first 2/3 of each expression's frames, test on the last 1/3 -> (accuracy, per-label)."""
    train = {k: v[: max(1, len(v) * 2 // 3)] for k, v in samples.items()}
    profile = build(train)
    per = {}
    for label in EXPRESSIONS:
        test = samples.get(label, [])[len(train.get(label, [])):]
        if test:
            per[label] = sum(classify(normalize(f, profile), profile)[0] == label for f in test) / len(test)
    acc = sum(per.values()) / max(1, len(per))
    return acc, per


def load(path=DEFAULT):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def save(profile, path=DEFAULT):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(profile), encoding='utf-8'); tmp.replace(path)


if __name__ == '__main__':  # self-check on synthetic faces
    import random
    random.seed(1)
    base = {'mouthSmileLeft': 0.3, 'mouthFrownLeft': 0.25, 'browDownLeft': 0.1, 'browInnerUp': 0.1, 'jawOpen': 0.05}
    shape = {'neutral': {}, 'happy': {'mouthSmileLeft': 0.55}, 'sad': {'mouthFrownLeft': 0.6},
             'angry': {'browDownLeft': 0.7}, 'surprised': {'browInnerUp': 0.8}}
    noisy = lambda d: {k: min(1, max(0, v + random.gauss(0, 0.03))) for k, v in d.items()}
    samples = {k: [noisy({**base, **s}) for _ in range(60)] for k, s in shape.items()}
    acc, per = evaluate(samples)
    assert acc > 0.95, per
    p = build(samples)
    hn = normalize(samples['happy'][0], p)
    assert hn['mouthSmileLeft'] > 0.6 and normalize(samples['neutral'][0], p)['mouthSmileLeft'] < 0.3, hn
    assert classify(normalize({**base, 'browDownLeft': 0.75}, p), p)[0] == 'angry'
    print('face_profile ok', round(acc, 3))
