# -*- coding: utf-8 -*-
"""
mflow doctor — is this machine ready, and what is missing for which command.

Nothing is installed or changed; every check says which commands need it and how to get it.
-lmp also inspects the LAMMPS build (`lmp -h` only — no run; use `mflow lmp-check` for that).
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

from . import __version__
from .log import LOG
from .report import Report

PACKAGES = [
    # module, pip name, needed for, required?
    ("numpy", "numpy", "everything", True),
    ("ase", "ase", "everything", True),
    ("matplotlib", "matplotlib", "figures", True),
    ("rich", "rich", "terminal output", True),
    ("yaml", "pyyaml", "configs, thresholds", True),
    ("torch", "torch  (the build matching your CUDA driver, see pytorch.org)", "calc, relax, check -model, train, validate, al", False),
    ("mace", "mace-torch==0.3.16", "calc, relax, check -model, train, validate, al", False),
    ("cuequivariance_torch", "cuequivariance-torch", "deploy (ML-IAP export), fast training on GPU", False),
    ("phonopy", "phonopy", "validate phonon", False),
    ("seekpath", "seekpath", "phonon band paths", False),
    ("pytest", "pytest", "running the test-suite", False),
]


def version_of(module) -> str:
    return str(getattr(module, "__version__", "") or "installed")


def check_packages(rep: Report) -> dict:
    found = {}
    for mod, pip, needed, required in PACKAGES:
        try:
            m = importlib.import_module(mod)
            found[mod] = version_of(m)
            status, msg = "pass", found[mod]
            if mod == "mace" and found[mod] not in ("0.3.16", "installed"):
                status, msg = "warn", f"{found[mod]} (MFlow is tested with 0.3.16)"
            rep.check(f"package {mod}", status, msg,
                      suggest="pip install mace-torch==0.3.16 — other versions change the training CLI")
        except Exception as exc:
            found[mod] = None
            rep.check(f"package {mod}", "fail" if required else "warn",
                      f"missing — needed for {needed}" if isinstance(exc, ImportError) else f"import failed: {exc}",
                      suggest=f"pip install {pip}")
    return found


def check_gpu(rep: Report, found: dict):
    if not found.get("torch"):
        return
    import torch
    if torch.cuda.is_available():
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        rep.check("GPU", "pass", f"{len(names)} × {names[0]} · CUDA {torch.version.cuda}")
    else:
        rep.check("GPU", "warn", "torch sees no CUDA device — training, MD and deploy will be slow or impossible",
                  suggest="install the torch build that matches your CUDA driver (pytorch.org), then re-run mflow doctor")


def gpu_ok(found: dict) -> bool:
    if not found.get("torch"):
        return False
    import torch
    return bool(torch.cuda.is_available())


def check_model(rep: Report):
    from .config import lookup
    value, source = lookup("model")
    if "/" in value or value.endswith(".model"):
        ok = Path(value).expanduser().is_file()
        rep.check("default model", "pass" if ok else "fail", f"{value} ({source})",
                  suggest="mflow config set model /path/to/your.model (or a foundation name: mpa, omat)")
    else:
        rep.check("default model", "pass", f"foundation model '{value}', downloaded by mace on first use"
                  + ("" if source.startswith("default") else f" ({source})"))


def check_lammps(rep: Report, explicit: str | None, inspect: bool):
    from .lmp import env
    exe = env.find_lmp(explicit)
    if not exe:
        rep.check("LAMMPS", "warn", "no lmp found — only needed for deploy / lmp-check / md / al (engine lammps)",
                  suggest="build LAMMPS with KOKKOS(CUDA) + ML-IAP + ML-SNAP + PYTHON (README 3.1), then "
                          "mflow config set lmp /path/to/lmp")
        return
    if not inspect:
        rep.check("LAMMPS", "pass", f"{exe} (add -lmp to inspect the build)")
        return
    try:
        info = env.lmp_info(exe)
    except Exception as exc:
        rep.check("LAMMPS", "fail", f"{exe}: lmp -h failed ({exc})")
        return
    missing = env.missing_for_mliap(info)
    rep.check("LAMMPS", "warn" if missing else "pass",
              f"{exe} · {info['version']}" + (f" · missing for MACE: {', '.join(missing)}" if missing else " · KOKKOS/ML-IAP/PYTHON"),
              suggest="this build can run classical potentials but not MACE: rebuild per README 3.1")


def check_skills(rep: Report):
    from .skills import installed_state
    dest, state = installed_state()
    if state["installed"] == state["total"]:
        rep.check("agent skills", "pass", f"{state['total']} skills in {dest}")
    else:
        rep.check("agent skills", "warn", f"{state['installed']}/{state['total']} installed in {dest}",
                  suggest="mflow skills install   (links the skills where pi and other agents look for them)")


def check_thresholds(rep: Report):
    import os
    over = [p for p in (os.environ.get("MFLOW_THRESHOLDS"), "mflow_thresholds.yaml") if p and Path(p).exists()]
    rep.check("thresholds", "pass", f"override in effect: {over[0]}" if over else "built-in (mflow/thresholds.yaml)")


def cmd_doctor(args) -> int:
    LOG.rule(f"MFlow {__version__} · doctor")
    from .config import config_path
    LOG.kv({"python": f"{platform.python_version()} ({sys.executable})", "mflow": str(Path(__file__).parent),
            "config": str(config_path()) + ("" if config_path().exists() else " (not created yet)")})
    rep = Report("doctor")
    found = check_packages(rep)
    check_gpu(rep, found)
    check_model(rep)
    check_lammps(rep, args.lmp, inspect=bool(args.lmp) or args.full)
    check_skills(rep)
    check_thresholds(rep)
    rep.show()
    if args.json:
        rep.write(args.json)
    ready = {
        "check / calc / relax / train / validate": all(found.get(m) for m in ("torch", "mace")),
        "deploy (GPU)": all(found.get(m) for m in ("torch", "mace", "cuequivariance_torch")) and gpu_ok(found),
        "validate phonon": bool(found.get("phonopy")),
    }
    LOG("ready for: " + " · ".join(f"[{'ok' if v else 'warn'}]{'✓' if v else '✗'} {k}[/]" for k, v in ready.items()))
    LOG("[dim]inside the MFlow repository the extras install everything at once: pip install -e \".\\[all]\"[/dim]")
    return 0 if not rep.failed else 2
