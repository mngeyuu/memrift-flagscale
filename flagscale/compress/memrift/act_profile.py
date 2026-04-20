"""
Activation path profiling (MemRift).

Enable with:
  export MEMRIFT_ACT_PROFILE=1

Optional:
  MEMRIFT_ACT_PROFILE_LOG_EVERY=500   # print rolling summary every N samples (default 500)

Interpreting results (GPU ANS activation path):
  - dec_cpu_future_result_ms: time in decode thread waiting for encode Future (forward D2H/encode backlog).
  - dec_cpu_h2d_ms / dec_cpu_decode_ms / dec_cpu_clone_sync_ms: decode thread phases; decode+clone
    are GPU kernels on tl_stream; if dec_cpu_decode_ms ~ step slowdown, SM contention with GEMM is likely.
  - unpack_ready_evt_wait_ms: main-thread Python blocks until decode thread calls ready_evt.set().
    If large, the backward consumer waited for decode scheduling/completion.
  - gpu_ms_hook_to_decode_end: CUDA elapsed between training-stream event at bwd_pre_hook and
    decode-stream event after decode (requires both streams to have completed those points).
    If this is << dec_cpu_decode_ms + GEMM time, GPU overlap occurred; if ~= sequential sum, little overlap.

Hook / unpack:
  - bwd_pre_hook_total_ms: full backward_pre_hook for one DecoderLayerWrapper (submit current + look-ahead).
  - unpack_wait_event_cpu_ms: CPU time for training_stream.wait_event(CtoD_copy_evt) after ready_evt.

Optional:
  MEMRIFT_ACT_PROFILE_ATEXIT=1  -> print full summary at process exit (dump).
"""
from __future__ import annotations

import os
import threading
import statistics
from collections import defaultdict
from typing import Dict, List, Optional

_lock = threading.Lock()
_samples: Dict[str, List[float]] = defaultdict(list)
_enabled: Optional[bool] = None
_log_every: int = 500
_atexit_registered: bool = False


def _ensure_atexit_dump() -> None:
    global _atexit_registered
    if _atexit_registered or os.environ.get("MEMRIFT_ACT_PROFILE_ATEXIT", "0") != "1":
        return
    _atexit_registered = True

    import atexit

    def _on_exit():
        dump("atexit")

    atexit.register(_on_exit)


def is_enabled() -> bool:
    global _enabled
    if _enabled is None:
        _enabled = os.environ.get("MEMRIFT_ACT_PROFILE", "0") == "1"
        global _log_every
        try:
            _log_every = max(1, int(os.environ.get("MEMRIFT_ACT_PROFILE_LOG_EVERY", "500")))
        except ValueError:
            _log_every = 500
        if _enabled:
            _ensure_atexit_dump()
    return bool(_enabled)


def reset() -> None:
    """Clear collected samples (e.g. between benchmark runs)."""
    global _enabled
    with _lock:
        _samples.clear()
    _enabled = None  # re-read env


def add_ms(name: str, ms: float) -> None:
    if not is_enabled():
        return
    with _lock:
        lst = _samples[name]
        if len(lst) < 100_000:
            lst.append(ms)


def _pctl(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = int(round((len(xs) - 1) * q))
    k = max(0, min(k, len(xs) - 1))
    return xs[k]


def maybe_log_rolling(tag: str = "") -> None:
    """Call from hot path; prints summary every MEMRIFT_ACT_PROFILE_LOG_EVERY samples on key."""
    if not is_enabled():
        return
    key = "unpack_ready_evt_wait_ms"
    with _lock:
        n = len(_samples.get(key, ()))
        if n == 0 or n % _log_every != 0:
            return
        snap = {k: list(v) for k, v in _samples.items()}
    _print_summary(snap, tag)


def dump(tag: str = "") -> None:
    """Print full summary now."""
    if not is_enabled():
        return
    with _lock:
        snap = {k: list(v) for k, v in _samples.items()}
    _print_summary(snap, tag)


def _print_summary(snap: Dict[str, List[float]], tag: str) -> None:
    lines = [f"[MEMRIFT_ACT_PROFILE] {tag}".rstrip()]
    names = sorted(snap.keys())
    for name in names:
        xs = snap[name]
        if not xs:
            continue
        lines.append(
            f"  {name}: n={len(xs)} mean={statistics.mean(xs):.3f}ms "
            f"p50={_pctl(xs, 0.50):.3f} p95={_pctl(xs, 0.95):.3f} "
            f"max={max(xs):.3f}ms"
        )
    print("\n".join(lines), flush=True)
