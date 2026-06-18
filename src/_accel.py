"""ARM / Apple-Silicon acceleration helpers.

Cadenza runs ARM end-to-end: Apple Silicon for local dev/sim, Jetson/ARM SBCs
on the robots. This module makes the whole stack use that hardware maximally —
native thread tuning for the M-series core layout and automatic selection of the
fastest available torch backend (Metal/MPS locally, CUDA on Jetson, CPU else).

No heavy imports at module load (only stdlib); torch is imported lazily so this
stays cheap to import from ``cadenza/__init__``.
"""

from __future__ import annotations

import functools
import os
import platform
import subprocess


@functools.lru_cache(maxsize=1)
def physical_perf_cores() -> int:
    """Number of high-performance physical cores to schedule compute on.

    On Apple Silicon, BLAS/torch work is fastest pinned to the P-cores; the
    E-cores hurt latency for bursty inference. Returns the P-core count there,
    and a sensible physical-core estimate on other ARM/x86 hosts.
    """
    if platform.system() == "Darwin":
        for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", key], text=True).strip()
                if out and int(out) > 0:
                    return int(out)
            except (subprocess.SubprocessError, ValueError, OSError):
                continue
    # Linux/Jetson and fallbacks: assume no SMT on ARM, so cpu_count is physical.
    return max(1, os.cpu_count() or 1)


def tune_threads() -> int:
    """Set thread-count env defaults tuned for the host's performance cores.

    Uses ``setdefault`` so an explicit user/operator override always wins. Must
    run before numpy/torch import to take effect for OpenBLAS-style backends;
    Cadenza calls it at package import. Returns the thread count chosen.
    """
    n = str(physical_perf_cores())
    for var in (
        "VECLIB_MAXIMUM_THREADS",  # Apple Accelerate (numpy's BLAS here)
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, n)
    # Let unsupported MPS ops fall back to CPU instead of crashing.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return int(n)


@functools.lru_cache(maxsize=1)
def torch_device():
    """Best available torch device: MPS (Apple) > CUDA (Jetson) > CPU."""
    import torch

    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    return torch.device(dev)


def torch_dtype(device=None):
    """Half precision on accelerators (MPS/CUDA), float32 on CPU."""
    import torch

    device = device or torch_device()
    return torch.float16 if device.type in ("mps", "cuda") else torch.float32
