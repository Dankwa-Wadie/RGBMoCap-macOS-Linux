"""Run CUDA-written inference code on Apple Silicon (MPS) without editing every call site.

RGB MoCap and its vendored model code (ExMR, GVHMR, FlowHMR, HaMeR, ...) address the
GPU as ``cuda``. On macOS, ``install()`` makes that name mean the MPS device:

* a ``TorchFunctionMode`` rewrites ``device='cuda'`` / ``.to('cuda')`` /
  ``torch.zeros(..., device='cuda:0')`` to ``mps`` for every torch call;
* ``Tensor.cuda`` / ``Module.cuda`` move to MPS; ``pin_memory`` is a no-op;
* ``torch.cuda.*`` memory, seed, sync, stream and event helpers are backed by
  ``torch.mps`` so budgeting and diagnostics keep working;
* ``torch.load(map_location='cuda')`` and ``torch.autocast('cuda')`` follow.

Autocast stays float32 by default (the Windows build also validates float32);
set ``RGB_MOCAP_MPS_AUTOCAST=1`` to allow reduced precision. Unsupported MPS
operators fall back to the CPU (``PYTORCH_ENABLE_MPS_FALLBACK=1``).
Windows/CUDA machines never call this module.
"""
from __future__ import annotations

import contextlib
import os
import platform
import subprocess
import time
from types import SimpleNamespace

_INSTALLED = False
_PEAK = [0]


def _target():
    return os.environ.get('RGB_MOCAP_DEVICE', 'mps')


def _map(device):
    import torch
    if isinstance(device, str):
        if device == 'cuda' or device.startswith('cuda:'):
            return _target()
        return device
    if isinstance(device, torch.device) and device.type == 'cuda':
        return torch.device(_target())
    if isinstance(device, int):  # .cuda(0) / .to(0) style ordinals
        return torch.device(_target())
    return device


def _chip():
    try:
        return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True,
                              text=True, timeout=2).stdout.strip() or platform.processor()
    except (OSError, subprocess.SubprocessError):
        return 'Apple Silicon'


def _note_peak(mps):
    value = mps.driver_allocated_memory()
    _PEAK[0] = max(_PEAK[0], value)
    return value


def install():
    """Idempotent; call once at worker start, before model code imports torch.cuda users."""
    global _INSTALLED
    if _INSTALLED:
        return
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    import torch
    from torch.overrides import TorchFunctionMode
    if _target() == 'mps' and not torch.backends.mps.is_available():
        os.environ['RGB_MOCAP_DEVICE'] = 'cpu'
    mps = torch.mps
    on_mps = _target() == 'mps'

    class CudaAsMps(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if 'device' in kwargs:
                kwargs['device'] = _map(kwargs['device'])
            if func in _TO_LIKE and len(args) > 1:
                args = (args[0], _map(args[1]), *args[2:])
            try:
                return func(*args, **kwargs)
            except TypeError as error:
                # MPS casts before copying, so tensor.to('cpu', torch.float64) fails where
                # CUDA succeeds. Copy to the CPU first, then cast (same result).
                if (func in _TO_LIKE and 'float64' in str(error) and args
                        and isinstance(args[0], torch.Tensor) and args[0].device.type == 'mps'):
                    return func(args[0].cpu(), *args[1:], **kwargs)
                raise

    _TO_LIKE = {torch.Tensor.to}

    # --- device movement -----------------------------------------------------------------
    def tensor_cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        return self.to(_target(), non_blocking=non_blocking, memory_format=memory_format)

    def module_cuda(self, device=None):
        return self.to(_target())

    torch.Tensor.cuda = tensor_cuda
    torch.nn.Module.cuda = module_cuda
    torch.Tensor.pin_memory = lambda self, *a, **k: self
    torch.Tensor.is_pinned = lambda self, *a, **k: False

    original_load = torch.load

    def load(*args, **kwargs):
        location = kwargs.get('map_location')
        if location is not None and not callable(location) and not isinstance(location, dict):
            kwargs['map_location'] = _map(location)
        elif location is None:
            # CUDA-saved checkpoints must not try to deserialize onto CUDA.
            kwargs['map_location'] = 'cpu'
        return original_load(*args, **kwargs)

    torch.load = load

    original_autocast = torch.autocast

    class autocast(original_autocast):
        def __init__(self, device_type='cuda', dtype=None, enabled=True, cache_enabled=None):
            if device_type == 'cuda':
                device_type = _target()
                if os.environ.get('RGB_MOCAP_MPS_AUTOCAST') != '1':
                    enabled = False
            if device_type == 'cpu' and dtype is None:
                dtype = torch.bfloat16
            super().__init__(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)

    torch.autocast = autocast

    # --- torch.cuda API backed by MPS ------------------------------------------------------
    cuda = torch.cuda
    total = mps.recommended_max_memory() if on_mps else 0
    name = _chip() + (' (MPS)' if on_mps else ' (CPU)')
    props = SimpleNamespace(name=name, total_memory=total, major=7, minor=0,
                            multi_processor_count=os.cpu_count() or 1, is_integrated=True)

    class Event:
        def __init__(self, enable_timing=False, blocking=False, interprocess=False):
            self.t = None

        def record(self, stream=None):
            if on_mps:
                mps.synchronize()
            self.t = time.perf_counter()

        def synchronize(self):
            if on_mps:
                mps.synchronize()

        def query(self):
            return True

        def wait(self, stream=None):
            pass

        def elapsed_time(self, end):
            return (end.t - self.t) * 1000.0

    class Stream:
        def __init__(self, *a, **k):
            pass

        def synchronize(self):
            if on_mps:
                mps.synchronize()

        def wait_stream(self, other):
            pass

        def wait_event(self, event):
            pass

        def record_event(self, event=None):
            event = event or Event(); event.record(); return event

        def query(self):
            return True

    def mem_get_info(device=None):
        used = _note_peak(mps) if on_mps else 0
        return max(0, total - used), total

    def synchronize(device=None):
        if on_mps:
            mps.synchronize()

    def empty_cache():
        if on_mps:
            mps.empty_cache()

    def allocated(device=None):
        return mps.current_allocated_memory() if on_mps else 0

    def reserved(device=None):
        return _note_peak(mps) if on_mps else 0

    def peak(device=None):
        if on_mps:
            _note_peak(mps)
        return _PEAK[0]

    def reset_peak(device=None):
        _PEAK[0] = mps.driver_allocated_memory() if on_mps else 0

    def seed(value):
        # torch.manual_seed itself calls torch.cuda.manual_seed_all: seed MPS directly.
        if on_mps:
            mps.manual_seed(int(value))

    replacements = dict(
        is_available=lambda: True, is_initialized=lambda: True, init=lambda: None,
        device_count=lambda: 1, current_device=lambda: 0, set_device=lambda device: None,
        get_device_name=lambda device=None: name, get_device_properties=lambda device=None: props,
        get_device_capability=lambda device=None: (props.major, props.minor),
        mem_get_info=mem_get_info, synchronize=synchronize, empty_cache=empty_cache,
        memory_allocated=allocated, memory_reserved=reserved, max_memory_allocated=peak,
        max_memory_reserved=peak, reset_peak_memory_stats=reset_peak, reset_max_memory_allocated=reset_peak,
        reset_accumulated_memory_stats=lambda device=None: None, ipc_collect=lambda: None,
        set_per_process_memory_fraction=lambda fraction, device=None: None,
        get_per_process_memory_fraction=lambda device=None: 1.0,
        manual_seed=seed, manual_seed_all=seed, is_bf16_supported=lambda *a, **k: on_mps,
        Event=Event, Stream=Stream, current_stream=lambda device=None: Stream(),
        default_stream=lambda device=None: Stream(), stream=lambda s=None: contextlib.nullcontext(),
        device=lambda d=None: contextlib.nullcontext(),
        memory_summary=lambda *a, **k: 'MPS allocated %.2f GiB' % (allocated() / 1024 ** 3),
    )
    for key, value in replacements.items():
        setattr(cuda, key, value)
    cuda.amp.autocast = lambda enabled=True, dtype=None, cache_enabled=True: autocast(
        'cuda', dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    # torch.cuda.OutOfMemoryError is torch.OutOfMemoryError, which the MPS allocator
    # also raises, so the app's OOM recovery (compute_policy.retry_cuda) still works.
    if os.environ.get('RGB_MOCAP_MPS_COMPILE') != '1':
        # Inductor on MPS is experimental; run eagerly like an uncompiled CUDA fallback.
        def compile(model=None, **kwargs):
            return model if model is not None else (lambda fn: fn)
        torch.compile = compile

    CudaAsMps().__enter__()
    _INSTALLED = True


class _AfterTorchImport:
    """meta_path hook: run install() as soon as ``torch`` finishes importing."""

    def find_spec(self, name, path=None, target=None):
        if name != 'torch':
            return None
        import importlib.machinery
        import sys
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        original = loader.exec_module

        def exec_module(module):
            original(module)
            install()
        loader.exec_module = exec_module
        return spec


def install_on_import():
    """Workers call this first; it is a no-op off macOS and when torch is already loaded."""
    import sys
    if sys.platform != 'darwin' or os.environ.get('RGB_MOCAP_DEVICE') == 'native':
        return
    if 'torch' in sys.modules:
        install()
    elif not any(isinstance(f, _AfterTorchImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _AfterTorchImport())
