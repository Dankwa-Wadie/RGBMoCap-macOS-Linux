"""Platform differences for the macOS (Apple Silicon) port, in one dependency-free place.

Windows keeps its original behaviour everywhere: every helper here returns the
Windows value on ``win32``. On macOS:

* Python environments use ``bin/python`` instead of ``Scripts/python.exe``.
* OpenGL is capped at 4.1 core, so compute shaders and SSBOs are unavailable;
  mesh skinning runs on the CPU (see ``mocap.cpu_skin``) and the draw shaders
  compile as GLSL 410.
* Torch inference uses CUDA when present, else Apple's MPS, else the CPU.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

# GPU compute skinning needs OpenGL 4.3 (compute shaders + std430 SSBOs).
# RGB_MOCAP_CPU_SKIN=1 forces the CPU path on any platform for testing.
GL_COMPUTE = not IS_MAC and os.environ.get("RGB_MOCAP_CPU_SKIN") != "1"
GL_VERSION = (4, 3) if GL_COMPUTE else (4, 1)
GLSL_HEADER = "#version 430 core" if GL_COMPUTE else "#version 410 core"


def glsl(source: str) -> str:
    """Retarget a ``#version 430 core`` draw shader to the context's GLSL version."""
    return source.replace("#version 430 core", GLSL_HEADER, 1)


def venv_python(folder: Path) -> Path:
    folder = Path(folder)
    return folder / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def runtime_python(folder: Path) -> Path:
    """Interpreter of a managed (installer-built) runtime folder."""
    folder = Path(folder)
    return folder / ("python.exe" if IS_WINDOWS else "bin/python3")


def torch_device(prefer: str | None = None) -> str:
    """Best available torch device: an explicit choice, then CUDA, MPS, CPU."""
    choice = prefer or os.environ.get("RGB_MOCAP_DEVICE")
    if choice:
        return choice
    import torch
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def configure_torch_fallbacks():
    """Let unsupported MPS operators fall back to the CPU instead of failing."""
    if IS_MAC:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def kill_tree(pid: int) -> None:
    """POSIX equivalent of ``taskkill /PID pid /T /F``: children first, then the parent."""
    import psutil
    try:
        parent = psutil.Process(int(pid))
    except psutil.Error:
        return
    for process in [*parent.children(recursive=True), parent]:
        try:
            process.kill()
        except psutil.Error:
            pass


def blender_candidates() -> list[Path]:
    """Usual Blender install locations outside Windows (newest first)."""
    if not IS_MAC:
        return []
    apps = [Path("/Applications"), Path.home() / "Applications"]
    found = []
    for folder in apps:
        found += sorted(folder.glob("Blender*.app"), reverse=True)
    return [app / "Contents/MacOS/Blender" for app in found]
