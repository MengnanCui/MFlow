# -*- coding: utf-8 -*-
"""
Thresholds from mflow/thresholds.yaml, optionally overridden by ./mflow_thresholds.yaml or the
file named in $MFLOW_THRESHOLDS (deep-merged: only the keys you set change).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

DEFAULT = Path(__file__).with_name("thresholds.yaml")


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in (over or {}).items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


@lru_cache(maxsize=None)
def thresholds() -> dict:
    import yaml
    data = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    for cand in (os.environ.get("MFLOW_THRESHOLDS"), "mflow_thresholds.yaml"):
        if cand and Path(cand).is_file():
            data = _merge(data, yaml.safe_load(Path(cand).read_text(encoding="utf-8")) or {})
    return data


def section(name: str) -> dict:
    return thresholds()[name]
