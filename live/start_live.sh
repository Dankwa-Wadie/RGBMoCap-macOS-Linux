#!/usr/bin/env bash
# RGB MoCap Live: Blender avatar + Mac webcam capture, in one command.
#   live/start_live.sh                 Seed-san with webcam 0
#   live/start_live.sh path/to.vrm     your own VRM 1.0 avatar
#   live/start_live.sh "" 1            another camera index
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
BLENDER="${RGBMOCAP_BLENDER:-}"
if [ -z "$BLENDER" ]; then for app in /Applications/Blender*.app "$HOME"/Applications/Blender*.app; do
  [ -x "$app/Contents/MacOS/Blender" ] && BLENDER="$app/Contents/MacOS/Blender"; done; fi
[ -n "$BLENDER" ] || { echo "Blender not found; set RGBMOCAP_BLENDER"; exit 1; }
VRM="${1:-live/avatars/Seed-san.vrm}"; CAMERA="${2:-0}"
BLEND="build/avatars/$(basename "${VRM%.*}").blend"
if [ ! -f "$BLEND" ] || [ "$VRM" -nt "$BLEND" ]; then
  mkdir -p build/avatars
  "$BLENDER" -b --factory-startup --python-exit-code 1 --python live/blender/prepare_vrm.py -- "$VRM" "$BLEND" | grep PREPARED
fi
"$BLENDER" "$BLEND" --python live/blender/start_live.py >/dev/null 2>&1 &
sleep 6
exec .venv-live/bin/python live/capture.py --camera "$CAMERA" ${RGBMOCAP_LIVE_DIAG:+--snapshots "$RGBMOCAP_LIVE_DIAG"}
