#!/usr/bin/env bash
# RGB MoCap launcher for macOS. Double-click in Finder, or run ./RGBMoCap.command
# Install first with: tools/setup_macos.sh
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then
  echo "RGB MoCap is not installed. Run tools/setup_macos.sh first."
  read -r -p "Press Return to close." _; exit 1
fi
export RGB_MOCAP_DATA="$ROOT" PYTHONDONTWRITEBYTECODE=1 \
  HF_HOME="$ROOT/cache/huggingface" TORCH_HOME="$ROOT/cache/torch" XDG_CACHE_HOME="$ROOT/cache" \
  PYTORCH_ENABLE_MPS_FALLBACK=1
unset PYTHONHOME PYTHONPATH
.venv/bin/python -X utf8 -u -B app.py "$@"
code=$?
if [ $code -ne 0 ]; then
  echo; echo "RGB MoCap exited with code $code. See the messages above."
  read -r -p "Press Return to close." _
fi
exit $code
