#!/usr/bin/env bash
# Rebuild the visor avatar from the 3js_ps2 project: glTF + face images -> live .blend and app FBX.
#   live/visor_avatar/build.sh [/path/to/3js_ps2]      (default /Volumes/Samsung_ssd/3js_ps2)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; HERE="$ROOT/live/visor_avatar"
SRC="${1:-/Volumes/Samsung_ssd/3js_ps2}"; OUT="$ROOT/build/avatars/visor_src"
B="${RGBMOCAP_BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
mkdir -p "$OUT"; cd "$OUT"
sed "s|^const P = .*|const P = '$SRC';|" "$HERE/export_visor.mjs" > export.mjs && node export.mjs | grep -v GLTFExporter | cut -c1-80
sed "s|^const P = .*|const P = '$SRC';|" "$HERE/paint_faces.mjs" > paint.mjs && node paint.mjs
"$B" -b --factory-startup --python-exit-code 1 --python "$HERE/prepare_visor.py" -- "$OUT/visor.gltf" "$ROOT/build/avatars/visor.blend" | grep PREPARED
"$B" -b --factory-startup --python-exit-code 1 --python "$HERE/to_fbx.py" -- "$OUT" | grep BONES | cut -c1-60
cp "$OUT/visor_mixamo.fbx" "$ROOT/build/avatars/visor_mixamo.fbx"
echo "built build/avatars/visor.blend and build/avatars/visor_mixamo.fbx"
