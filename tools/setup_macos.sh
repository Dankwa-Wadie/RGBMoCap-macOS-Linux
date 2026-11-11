#!/usr/bin/env bash
# RGB MoCap: macOS (Apple Silicon) setup.
#
# Builds the same separated runtimes as install.cmd, as per-profile virtual
# environments, from requirements/macos/*.txt (the Windows pins with CUDA wheel
# tags removed). Inference uses Apple's MPS GPU through mocap/torch_mps.py.
#
#   tools/setup_macos.sh                  shared/GUI + ExMR runtimes, motion correction
#   tools/setup_macos.sh --all            also FlowHMR, HandFlow and HTD runtimes
#   tools/setup_macos.sh --blender-addon  also install the Bone Stream add-on into Blender
#   tools/setup_macos.sh --mpi-email you@example.com
#                                         also download SMPL-X/MANO with your own account
#   tools/setup_macos.sh --no-models      skip model downloads
#
# Model files are fetched by tools/fetch_models.py from their official sources.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ALL=0; ADDON=0; MODELS=1; MPI_EMAIL_ARG=""
shift_next=0
for arg in "$@"; do
  if [ $shift_next = 1 ]; then MPI_EMAIL_ARG="$arg"; shift_next=0; continue; fi
  case "$arg" in
    --all) ALL=1 ;;
    --blender-addon) ADDON=1 ;;
    --no-models) MODELS=0 ;;
    --mpi-email) shift_next=1 ;;
    --mpi-email=*) MPI_EMAIL_ARG="${arg#*=}" ;;
    -h|--help) sed -n 2,15p "$0"; exit 0 ;;
    *) echo "Unknown option: $arg"; exit 2 ;;
  esac
done

say() { printf '\n==> %s\n' "$*"; }
[ "$(uname -s)" = Darwin ] || { echo "This script is for macOS. On Windows run install.cmd."; exit 1; }
[ "$(uname -m)" = arm64 ] || echo "Warning: not Apple Silicon; inference will run on the CPU."

say "Checking tools"
command -v uv >/dev/null || { echo "uv is required: brew install uv  (or https://docs.astral.sh/uv/)"; exit 1; }
command -v cmake >/dev/null || { echo "CMake is required: brew install cmake"; exit 1; }
command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null || { echo "FFmpeg is required: brew install ffmpeg"; exit 1; }
xcode-select -p >/dev/null 2>&1 || { echo "Xcode Command Line Tools are required: xcode-select --install"; exit 1; }
if ! [ -e /opt/homebrew/opt/jpeg-turbo/lib/libturbojpeg.dylib ] && ! [ -e /usr/local/opt/jpeg-turbo/lib/libturbojpeg.dylib ]; then
  echo "Note: ExMR's PyTurboJPEG needs libjpeg-turbo: brew install jpeg-turbo"
fi
free_gb=$(df -g "$ROOT" | awk 'NR==2{print $4}')
echo "Free disk space: ${free_gb} GB (ExMR models need ~12 GB; all offline models far more)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"

make_env() { # folder python requirements
  local folder=$1 python=$2 req=$3
  say "Runtime $folder (Python $python) from $req"
  uv python install "$python" >/dev/null
  [ -x "$folder/bin/python" ] || uv venv -q --python "$python" "$folder"
  # chumpy (SMPL loaders) builds with pip and NumPy present, without isolation.
  if grep -q '^chumpy==' "$req"; then
    uv pip install -q --python "$folder/bin/python" pip "$(grep '^setuptools==' "$req")" wheel "$(grep '^numpy==' "$req")"
    uv pip install -q --python "$folder/bin/python" --no-build-isolation "$(grep '^chumpy==' "$req")"
  fi
  # The pins are a complete, validated closure (as on Windows): no re-resolution.
  uv pip install -q --python "$folder/bin/python" --no-deps -r "$req"
  "$folder/bin/python" -c "import torch;print('  torch',torch.__version__,'MPS',torch.backends.mps.is_available())"
}

make_env .venv 3.10 requirements/macos/runtime-shared.txt
make_env .venv-exmr 3.12 requirements/macos/runtime-exmr.txt
if [ $ALL = 1 ]; then
  make_env .venv-flow 3.10 requirements/macos/runtime-flow.txt
  make_env .venv-hand 3.10 requirements/macos/runtime-hand.txt
  make_env .venv-htd 3.10 requirements/macos/runtime-htd.txt
fi

say "Building Motion Correction (arm64, SSE->NEON via sse2neon)"
BUILD="$ROOT/build/motion-correction"
cmake -S third_party/KimodoMotionCorrection -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$ROOT/.venv/bin/python" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >"$ROOT/build/motion-correction.log" 2>&1
cmake --build "$BUILD" --target _motion_correction -j "$(sysctl -n hw.ncpu)" >>"$ROOT/build/motion-correction.log" 2>&1
mkdir -p third_party/KimodoMotionCorrection/bin
cp "$BUILD"/_motion_correction*.so third_party/KimodoMotionCorrection/bin/
.venv/bin/python -c "import sys;sys.path.insert(0,'.');from mocap.correction import backend;m=backend();assert m.rgb_mocap_anatomical_version==1;print('  motion correction OK:',m.__file__)"

if [ $ADDON = 1 ]; then
  say "Installing the RGB MoCap Bone Stream add-on"
  BLENDER="${RGBMOCAP_BLENDER:-}"
  if [ -z "$BLENDER" ]; then
    for app in /Applications/Blender*.app "$HOME"/Applications/Blender*.app; do
      [ -x "$app/Contents/MacOS/Blender" ] && BLENDER="$app/Contents/MacOS/Blender"
    done
  fi
  [ -n "$BLENDER" ] || { echo "Blender not found. Set RGBMOCAP_BLENDER=/path/to/Blender"; exit 1; }
  "$BLENDER" -b --factory-startup --python-exit-code 1 --python-expr "
import bpy, addon_utils
bpy.ops.preferences.addon_install(filepath='$ROOT/blender_addon/rgbmocap_bone_stream.zip', overwrite=True)
addon_utils.enable('rgbmocap_bone_stream', default_set=True, persistent=True)
bpy.ops.wm.save_userpref()
print('RGB MoCap Bone Stream enabled in', bpy.app.version_string)" 2>&1 | grep -E "enabled|Error" || true
fi

if [ $MODELS = 1 ]; then
  say "Model files (official sources; SMPL-X/MANO from ~/Downloads or your own account)"
  .venv/bin/python tools/fetch_models.py ${MPI_EMAIL_ARG:+--mpi-email "$MPI_EMAIL_ARG"} || \
    echo "Some model files still need you (see above). Rerun: .venv/bin/python tools/fetch_models.py"
fi

say "Done. Start the app with ./RGBMoCap.command (or double-click it in Finder)."
echo "Check models any time: .venv/bin/python tools/fetch_models.py --check"
