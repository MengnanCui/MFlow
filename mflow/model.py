# -*- coding: utf-8 -*-
"""Locating, loading and wrapping MACE models."""

from __future__ import annotations

import os
import sys

import numpy as np

from .log import LOG

from .config import default_model

DEFAULT_MODEL = default_model()   # mflow config set model …  > $MFLOW_MODEL > built-in (see config.py)
MODEL_ALIASES = {"mp": "medium", "mp0": "medium", "mpa": "medium-mpa-0", "mpa0": "medium-mpa-0",
                 "omat": "medium-omat-0", "omat0": "medium-omat-0"}


def resolve_model(spec: str) -> tuple[str, str]:
    """-> ('file', path) for a local model, else ('foundation', name to download)."""
    path = os.path.expanduser(spec or "")
    if os.path.isfile(path):
        return "file", path
    if "/" in path or path.endswith(".model"):
        sys.exit(f"model file not found: {path}\n"
                 f"  pass -model <file or foundation name>, or set a default: mflow config set model <file>")
    return "foundation", MODEL_ALIASES.get(spec.lower(), spec)


def load_model(spec: str, device: str, dtype: str):
    """Raw torch module + device, from a local file or a foundation model name."""
    import torch
    from mace.tools import torch_tools

    torch_tools.set_default_dtype(dtype)
    dev = torch_tools.init_device(device)
    kind, value = resolve_model(spec)
    if kind == "file":
        try:
            model = torch.load(value, map_location=dev, weights_only=False)
        except TypeError:
            model = torch.load(value, map_location=dev)
    else:
        from mace.calculators import mace_mp
        model = mace_mp(model=value, device=str(dev), default_dtype=dtype, return_raw_model=True)

    model = (model.double() if dtype == "float64" else model.float()).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    LOG(f"model [key]{value}[/key] · {sum(p.numel() for p in model.parameters()):,} params · "
        f"r_max {float(model.r_max):.1f} Å · {dev} · {dtype}")
    return model, dev


def build_calculator(spec: str, device: str, dtype: str, head: str | None):
    """ASE calculator for a local .model file or a foundation model name."""
    kind, value = resolve_model(spec)
    kwargs = {"device": device, "default_dtype": dtype, **({"head": head} if head else {})}
    if kind == "file":
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=value, **kwargs)
    from mace.calculators import mace_mp
    return mace_mp(model=value, **kwargs)

def versions() -> str:
    import platform
    parts = [f"python {platform.python_version()}", f"numpy {np.__version__}"]
    for name in ("ase", "torch", "mace"):
        try:
            parts.append(f"{name} {__import__(name).__version__}")
        except Exception:
            parts.append(f"{name} —")
    try:
        import torch
        parts.append(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only")
    except Exception:
        pass
    return " · ".join(parts)


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
