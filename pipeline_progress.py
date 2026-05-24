import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm


PROGRESS_ENV = "PIPELINE_PROGRESS"


def progress_path():
    path = os.environ.get(PROGRESS_ENV)
    return Path(path) if path else None


def write_progress(desc, current, total=None, unit="", done=False):
    path = progress_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    total_value = int(total) if total is not None else None
    current_value = int(current)
    percent = None
    if total_value and total_value > 0:
        percent = max(0.0, min(100.0, 100.0 * current_value / total_value))
    payload = {
        "desc": str(desc or ""),
        "current": current_value,
        "total": total_value,
        "unit": str(unit or ""),
        "percent": percent,
        "done": bool(done),
        "updated_at": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def clear_progress():
    path = progress_path()
    if path is not None and path.exists():
        path.unlink()


def tqdm_disable_for_web(disable=None):
    if disable is not None:
        return disable
    return progress_path() is not None and not sys.stderr.isatty()


class ProgressBar:
    def __init__(self, total=None, desc="", unit="", **kwargs):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.current = 0
        kwargs.setdefault("disable", tqdm_disable_for_web(kwargs.get("disable")))
        self.bar = tqdm(total=total, desc=desc, unit=unit, **kwargs)
        write_progress(desc, 0, total, unit)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        done = exc_type is None
        write_progress(self.desc, self.current, self.total, self.unit, done=done)
        self.bar.close()
        return False

    def update(self, n=1):
        self.current += int(n)
        self.bar.update(n)
        write_progress(self.desc, self.current, self.total, self.unit)


def progress_bar(total=None, desc="", unit="", **kwargs):
    return ProgressBar(total=total, desc=desc, unit=unit, **kwargs)


def progress_iter(iterable, desc="", unit="", total=None, **kwargs):
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    kwargs.setdefault("disable", tqdm_disable_for_web(kwargs.get("disable")))
    write_progress(desc, 0, total, unit)
    current = 0
    for item in tqdm(iterable, desc=desc, unit=unit, total=total, **kwargs):
        yield item
        current += 1
        write_progress(desc, current, total, unit)
    write_progress(desc, current, total, unit, done=True)
