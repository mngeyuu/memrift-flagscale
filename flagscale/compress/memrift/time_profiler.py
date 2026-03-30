import atexit
import json
import os
import threading
import time


_OUTPUT_PATH = os.environ.get("MEMRIFT_LAYER_TIME_PROFILE_PATH")
_ENABLED = bool(_OUTPUT_PATH) and os.environ.get("RANK", "0") in ("0", "")
_LOCK = threading.Lock()
_DATA = {}
_LIVE_DUMP_SEC = float(os.environ.get("MEMRIFT_LAYER_TIME_PROFILE_LIVE_SEC", "0") or "0")
_LAST_LIVE_DUMP_TS = 0.0


def is_enabled() -> bool:
    return _ENABLED


def add_time(layer_name: str, metric: str, ms: float) -> None:
    global _LAST_LIVE_DUMP_TS
    if not _ENABLED:
        return
    if not layer_name:
        layer_name = "unknown"
    with _LOCK:
        layer_stats = _DATA.setdefault(layer_name, {})
        metric_stats = layer_stats.setdefault(
            metric,
            {"sum_ms": 0.0, "count": 0, "max_ms": 0.0},
        )
        metric_stats["sum_ms"] += float(ms)
        metric_stats["count"] += 1
        metric_stats["max_ms"] = max(metric_stats["max_ms"], float(ms))
    if _LIVE_DUMP_SEC > 0:
        now = time.time()
        if now - _LAST_LIVE_DUMP_TS >= _LIVE_DUMP_SEC:
            _LAST_LIVE_DUMP_TS = now
            try:
                dump()
            except Exception:
                pass


def _layer_sort_key(name: str):
    try:
        if "layers." in name:
            idx = int(name.split("layers.", 1)[1].split(".", 1)[0])
            return (0, idx, name)
    except Exception:
        pass
    return (1, name)


def dump() -> None:
    if not _ENABLED or not _OUTPUT_PATH:
        return

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pid": os.getpid(),
        "layers": [],
    }

    with _LOCK:
        for layer_name in sorted(_DATA.keys(), key=_layer_sort_key):
            metrics = {}
            for metric_name, stats in sorted(_DATA[layer_name].items()):
                avg_ms = stats["sum_ms"] / stats["count"] if stats["count"] else 0.0
                metrics[metric_name] = {
                    "avg_ms": avg_ms,
                    "sum_ms": stats["sum_ms"],
                    "count": stats["count"],
                    "max_ms": stats["max_ms"],
                }
            payload["layers"].append({"layer": layer_name, "metrics": metrics})

    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
    tmp_path = f"{_OUTPUT_PATH}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, _OUTPUT_PATH)


atexit.register(dump)
