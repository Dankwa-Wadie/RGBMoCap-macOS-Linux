#!/usr/bin/env bash
# Phone video of a room -> Gaussian splat -> photoreal "In My Room" exports + Blender room scene.
#   live/splat/make_room.sh ~/Downloads/IMG_2522.MOV [ROOM_DIR=build/splat]
# Walk slowly around the room, keep it still, ~1 minute. Takes ~25 min on an M4 Pro.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
VIDEO="$1"; W="${2:-build/splat}"; mkdir -p "$W/all" "$W/images"; rm -f "$W"/all/*.jpg "$W"/images/*.jpg
ffmpeg -v error -i "$VIDEO" -vf "fps=8,scale=-2:1600" -q:v 2 "$W/all/%05d.jpg"
.venv/bin/python - "$W" <<'PY'
import cv2, sys, shutil
from pathlib import Path
w = Path(sys.argv[1]); frames = sorted((w / 'all').glob('*.jpg'))
sharp = lambda f: cv2.Laplacian(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), cv2.CV_64F).var()
for k, i in enumerate(range(0, len(frames), 2)):
    shutil.copy(max(frames[i:i + 2], key=sharp), w / 'images' / f'{k:04d}.jpg')
PY
.venv-exmr/bin/python -u live/splat/colmap_solve.py "$W"
mkdir -p "$W/undistorted/sparse/0"; mv "$W"/undistorted/sparse/*.bin "$W/undistorted/sparse/0/" 2>/dev/null || true
.venv-exmr/bin/python live/splat/align.py "$W/undistorted/sparse/0" "$W/align.json"
.venv-splat/bin/python -u live/splat/train_room.py "$W/undistorted" "$W/room"
.venv-splat/bin/python live/splat/render_room.py "$W"
BLENDER="${RGBMOCAP_BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
"$BLENDER" -b build/avatars/Seed-san.blend --python-exit-code 1 --python live/splat/import_splat.py -- \
  "$W/room.ply" "$W/align.json" build/avatars/Seed-san_room.blend 150000 | grep SPLAT
echo "Room ready: Export > In My Room in the Live panel; Blender scene build/avatars/Seed-san_room.blend"
