# Progress: visor character in RGB MoCap (macOS branch)

Handover notes for continuing this work with another agent. Newest work last.

## Goal
Drive the **visor** character from the three.js project `/Volumes/Samsung_ssd/3js_ps2` (`examples/visor.js`,
built by `src/kit.js`) with RGB MoCap: Live mode (webcam -> Blender) and the offline app's
Import Character / Retarget.

## How to run it
```bash
live/visor_avatar/build.sh                      # 3js_ps2 -> build/avatars/visor.blend (+ visor_mixamo.fbx)
/Applications/Blender.app/Contents/MacOS/Blender build/avatars/visor.blend --python live/blender/start_live.py &
.venv-live/bin/python live/capture.py --camera 0 --unmirror        # --track auto is the default
```
`live/start_live.sh` only takes a VRM; for the visor launch the two commands above.
App: File > Import Character... > `build/avatars/visor_mixamo.fbx`.

## What was done

### 1. Rig with the full Mixamo body set (3js_ps2 project, not this repo)
- `src/kit.js`: optional bones `spine1`, `shoulder.L/R`, `toe.L/R`, skinned by position like the rest
  (shoulder = upper chest beside the neck, toe = foot ahead of the ball). Characters without them are unchanged.
- `src/retarget.js`: MIXAMO map includes Spine1 / Shoulder / ToeBase.
- `examples/visor.js`: has the three bones (spine1 at y 1.08; higher made the jacket clip 5-9 mm in Run).
- 3js_ps2 tests: 13/13 pass. `viewer.html` rebuilt; its `PROGRESS.md` updated. (3js_ps2 is not a git repo.)

### 2. Export pipeline (`live/visor_avatar/`)
- `export_visor.mjs` (Node): builds the visor, renames bones to Mixamo names, writes `visor.gltf`.
  - Visor face screen: in three it copies the head bone's matrix at draw time, so a plain export left it at
    the feet (the white block by the legs). It is now parented to the Head bone; its painted face is written as PNG.
  - Robot "buddy" garment (white bot on the left thigh) is left out of the export.
  - Textures painted on canvas are dropped (Node has no canvas); colours stay.
- `paint_faces.mjs`: paints 90 face images (5 expressions x blink x 3 mouth sizes x 3 gaze) for live expressions.
- `prepare_visor.py` (Blender): glTF -> live avatar `.blend`: `rgbmocap_humanoid` map (42 bones incl. fingers),
  emissive screen material, face images packed, `rgbmocap_visor` marker, dual-quaternion skinning
  (keeps hip/waist volume), and **bone axes re-aimed at their children** (see 5).
- `to_fbx.py` (Blender): glTF -> Mixamo-style FBX for the app (same bone re-aim and screen material).
- Verified: the app's own import step (`mocap.retarget.worker.invoke('import', ...)`) imports the FBX as
  `mixamo`, 42 bones; `body_mapping('mixamo', ...)` finds all 22 body bones.

### 3. Live receiver changes (`live/blender/rgbmocap_live.py`)
- **Visor expressions**: `apply_visor` maps the ARKit face blendshapes (via the existing `vrm_expressions`) to one of
  the packed images: happy/sad/angry/surprised above 0.35 else neutral, blink, mouth open (3 sizes), gaze L/R.
  Only for avatars with `rgbmocap_visor`; others unchanged. Recording does not bake the screen image.
- **Hand to face** (`reach_head`): when a wrist is near the head (< 1.1 shoulder widths, full by 0.7), the wrist is
  placed relative to the avatar's head (scaled by shoulder width) and the arm solved as two bones with the elbow
  toward the performer's. Fixes hands missing the face on avatars with other proportions. All avatars.
- **Unseen arms hang down**: an arm without tracking (below the frame at a desk, or shoulders out of the top when
  close to the camera) hangs at the side instead of easing back to the rig's A/T rest pose.
- **Glitch filter** (`steady`): a bone jump > 80 deg is held until it persists 3 frames (one-frame hand misreads,
  swapped hands); every bone then turns at most 45 deg/frame (hands/fingers 30).

### 4. Capture changes (`live/capture.py`)
- `leg_chain`: leg landmarks count only as a connected chain from the hip (knee needs hip, ankle needs knee,
  heel/toe need ankle), so a guessed ankle/knee under a desk or past the frame edge never drives a leg.
  Partly visible legs drive only the visible part.
- `--track upper` (existing) locks the legs entirely; recommended for desk/close-up use.

### 5. Root cause of "hands can't go to the side / touch the face"
Blender's glTF importer pointed every limb bone straight down (upper arm 33 deg off its forearm). The live driver
aims a bone's Y axis along the limb, so all arm poses were rotated. `aim_bones` in `prepare_visor.py`/`to_fbx.py`
points each bone at its child (hand -> middle finger, foot -> toe).

## Tests run (and how to repeat)
- `tests_macos/live_pose_check.py -- OUT_DIR`: synthetic poses (arms down, hand on face, hands on hips, arms crossed,
  arms off-frame) rendered on an avatar. Visor and Seed-san both correct after the fixes.
- `tests_macos/live_jitter_check.py -- REC_DIR`: replays `capture.py --dump REC/packets.jsonl` and lists bones
  turning > 60 deg in one frame. Real 6960-frame webcam recording: before, dozens of 90-150 deg finger/hand flips
  and 85 deg arm jumps; after, none.
- `tests_macos/live_replay_render.py` (existing; now also applies `leg_chain`): renders the avatar at every camera
  snapshot (`capture.py --snapshots REC`) for side-by-side review. A 12-sample review of the real recording matched
  arms down, hands in front, arm out, arm bent up, wave by the head, hand across the face, leaning in.
- Record a test session: `capture.py ... --dump REC/packets.jsonl --snapshots REC` (contains camera images: keep
  out of git).

### 6. Auto tracking mode (smart switch) - now the default
- `live/track_modes.py`: new `auto` mode (first in the app's Live panel list, default for `capture.py --track`).
  Hips always count when seen (lying detection, bending at the waist); a leg drives only where clearly seen
  (knee visibility >= 0.9, ankle/heel/toe >= 0.85) and connected from the hip (`leg_chain`, moved here from
  capture.py and shared by capture, the replay test and every mode). When the torso is lying (> 45 deg from
  upright) the normal leg rule applies (legs are in view in bed; MediaPipe rates them 0.4-0.7).
  `packet['track_mode']` reports `auto:full` / `auto:partial` / `auto:upper`.
- `live/capture.py`: hips get their own validity rule (visibility > 0.9, inside the picture up to y < 1.04): the
  strict leg edge margin dropped the hips of a body lying at the bottom edge, so lying was never detected.
- Tested on the real recording (validity recomputed from raw visibility): lying shots now lie down with legs,
  standing/desk shots unchanged (legs still: knees at the bottom edge, ankles off frame), no bone > 60 deg/frame.
- Old `upper` mode still exists; it also masks the hips (no lying / waist lean), by design.

### 7. Body models (SMPL-X / MANO)
- Not committed and must not be: their licenses forbid redistribution (see docs/MACOS.md). MANO is installed
  locally (`models/body_models/mano/MANO_RIGHT.pkl`); SMPL-X Neutral was still missing on this machine
  (`.venv/bin/python tools/fetch_models.py --check`). Users get them automatically by putting
  `models_smplx_v1_1.zip` / `mano_v1_2.zip` in ~/Downloads, or `tools/fetch_models.py --mpi-email ...`.

### 8. Stability pass
- `tests_macos/live_stability_check.py -- PACKETS.jsonl [MODE]`: per body group, jitter (deg a bone sits off the
  midpoint of its neighbours' frames), flicker (tracked<->untracked switches /100 frames) and lag (frames behind
  the raw target). `LIVE_RECEIVER=path` scores another receiver version (A/B).
- Receiver: grace (a bone that lost tracking holds 6 frames before easing to rest); adaptive smoothing in
  `steady` (One-Euro-like on rotations: `SMOOTH, SMOOTH_HAND = 3.0, 8.0` deg; slow steady moves trail ~4-5 deg);
  lying needs the 3D tilt AND the 2D picture layout (shoulders beside the hips) held `LYING_FRAMES = 8`
  (sitting on the floor with knees up made the 3D torso guess tip over -> false lying).
- Capture: hand detector thresholds 0.3 (default 0.5): +59 % hands on snapshots, all but 1 at a tracked wrist.
- `track_modes.lying` (Auto legs when lying) uses the same 2D check.
- Second real session (1185 packets, sitting on the floor with a phone), committed receiver vs new:
  torso jitter 0.46 -> 0.28, head 1.10 -> 0.70, arms 1.04 -> 0.78, hands 2.16 -> 1.19, fingers 1.77 -> 1.04,
  legs 0.26 -> 0.18 deg; flicker torso 0.86 -> 0.46, legs 0.37 -> 0.22, hands 1.10 -> 1.01; hands detected per
  frame with a wrist in view 0.95 -> 1.56. No false lying on the floor; the lying session still lies down.
- Head lag reads 3 frames on that session after the lying change (was 0); head barely moved there, so the
  estimate may be noise - check on a recording with head turns.

## Not done / known limits
- Offline **Retarget** in the app not tested: the sample project's ExMR result is still pending (no finished person).
- Hand covering the nose can drop head tracking briefly (head eases to rest).
- An arm raised out of the top of the frame now hangs down (trade-off of "unseen arms hang").
- Lying down with `--track upper`: avatar stays upright (hips/legs locked by design); use `auto`.
- Auto: the first second of lying down can show the avatar half-tilted while the hip landmarks stabilise
  (6-frame streak) and bones ease in.
- Remaining hip crease when a leg is raised near 90 deg (kit hip skin bands in 3js_ps2).
- Fingers can still move up to 30 deg/frame while hand tracking is noisy (capped, not removed).
- The visor screen texture shows one fixed expression in the FBX (expressions only in Live).
- None of this is committed: changes on branch `macos` are `live/blender/rgbmocap_live.py`, `live/capture.py`,
  `tests_macos/live_replay_render.py`, new `live/visor_avatar/`, `tests_macos/live_pose_check.py`,
  `tests_macos/live_jitter_check.py`, this file. `build/` holds generated avatars.
