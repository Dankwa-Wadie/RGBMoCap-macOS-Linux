# RGB MoCap on macOS (Apple Silicon)

This is a port of the Windows Git/CMD distribution to macOS on Apple Silicon
(M1–M4). The app, its 3D viewport, motion correction, the exporters and the
Blender Bone Stream all run natively. AI inference runs on the Apple GPU
through PyTorch's MPS backend.

## Install

Requirements: macOS 14 or later on Apple Silicon, the Xcode Command Line Tools,
Homebrew, and about 15 GB free for the base runtimes and ExMR's model files.

```sh
brew install uv cmake ffmpeg jpeg-turbo
git clone https://github.com/toyxyz/toyxyz_RGBMocap.git
cd toyxyz_RGBMocap
tools/setup_macos.sh --blender-addon      # add --all for the FlowHMR/HandFlow/HTD runtimes
./RGBMoCap.command                        # or double-click RGBMoCap.command in Finder
```

`setup_macos.sh` builds the same separated runtimes as `install.cmd`, as
virtual environments (`.venv`, `.venv-exmr`, and with `--all`, `.venv-flow`,
`.venv-hand` and `.venv-htd`). It uses the pins in `requirements/macos/`,
which are the Windows pins with the CUDA wheel tags removed. It also compiles
the motion correction module for arm64 and, with `--blender-addon`, installs
and enables the Bone Stream add-on in the newest Blender in `/Applications`.

Prepare model files in **Model Manager**, as on Windows. SMPL-X and MANO
need a free account on their providers' sites. SAM 3D Body (used by ExMR and
FlowHMR) is gated on Hugging Face: request access on
`facebook/sam-3d-body-dinov3`, then run `hf auth login` before using
**Auto Download**.

## What changed for macOS

| Area | Windows | macOS port |
| --- | --- | --- |
| Viewport | OpenGL 4.3 compute-shader skinning | OpenGL 4.1 (the macOS maximum). The same skinning math runs in NumPy (`mocap/cpu_skin.py`) and fills the same vertex buffer. The draw shaders compile as GLSL 410. |
| Inference | CUDA | MPS. `mocap/torch_mps.py` maps the `cuda` device and the `torch.cuda` memory, seed and sync APIs onto MPS in worker processes. Unsupported operators fall back to the CPU. Autocast stays float32 (set `RGB_MOCAP_MPS_AUTOCAST=1` for reduced precision). |
| Motion correction | Prebuilt `cp310-win_amd64.pyd` (SSE/AVX) | Built from the same source for arm64, with SSE mapped to NEON by `sse2neon` in precise mode. |
| Runtimes | `runtime*/python.exe` | `.venv*/bin/python` (`runtime_registry.activate_posix`) |
| Process control | `taskkill /T` | Process-tree kill through psutil |
| Blender | `blender.exe` | `/Applications/Blender*.app`, or `RGBMOCAP_BLENDER` |

Set `RGB_MOCAP_DEVICE=cpu` to force CPU inference.

## Verification on an M4 Pro (24 GB)

- The GUI starts, opens a 1920×1080, 60 fps video, and decodes frames and the timeline.
- CPU skinning matches a line-by-line transliteration of every GLSL kernel (`tests_macos/test_cpu_skin.py`).
- A real GL 4.1 core context renders lit, depth-tested CPU-skinned meshes (`tests_macos/gl41_render_check.py`).
- Motion correction: on the same synthetic 120-frame clip with contacts, the arm64 (NEON) build's output is identical to an x86_64 (SSE) build run under Rosetta (`tests_macos/correction_parity.py`).
- The MPS layer covers device remapping, modules, memory APIs, events, autocast and `torch.load` (`tests_macos/test_torch_mps.py`).

Run the tests with:

```sh
PYTHONPATH=. .venv/bin/python -m pytest -q tests_macos
PYTHONPATH=. .venv/bin/python tests_macos/gl41_render_check.py /tmp/gl41.png
```

## Not supported on macOS

- **MAMMA** and **PromptHMR** need custom CUDA kernels (SAM 2 and SDF builds,
  DroidCalib/LieTorch) that have no Metal port.
- **SAM 3.1 tracking** and **PEAR live** have not been ported. Webcam capture
  uses DirectShow on Windows.
- **FlowHMR, GVHMR and GEM-X are untested on macOS.** FlowHMR has optional
  CUDA paths (flash-attn, cupy, compiled kernels) that must fall back to eager
  PyTorch code. See the status table in the port notes below.

Expect inference to be several times slower than on a high-end NVIDIA GPU.
Unified memory lets large models load, but long clips use a lot of RAM.
