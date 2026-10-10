# -*- coding: utf-8 -*-
"""
Where is LAMMPS, what was it built with, and how is it launched.

The MACE ML-IAP path needs KOKKOS (with a CUDA backend), ML-IAP and PYTHON. Everything
here reads `lmp -h`, so it works for any build — conda, pip wheel or your own cmake.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

REQUIRED_MLIAP = ("KOKKOS", "ML-IAP", "PYTHON")
KOKKOS_ARGS = ["-sf", "kk", "-pk", "kokkos", "newton", "on", "neigh", "half"]


def find_lmp(explicit: str | None = None) -> str | None:
    """-lmp > $MFLOW_LMP > PATH (lmp, then the usual build names)."""
    for cand in (explicit, os.environ.get("MFLOW_LMP")):
        if cand:
            path = shutil.which(cand) or (cand if Path(cand).exists() else None)
            if path:
                return str(Path(path).resolve())
    for name in ("lmp", "lmp_kokkos_cuda_mpi", "lmp_kokkos_cuda", "lmp_mpi", "lmp_serial"):
        path = shutil.which(name)
        if path:
            return path
    return None


def lmp_info(exe: str) -> dict:
    """Version, packages and accelerator configuration, parsed from `lmp -h`."""
    out = subprocess.run([exe, "-h"], capture_output=True, text=True, timeout=120).stdout
    info = {"exe": exe, "version": None, "packages": [], "kokkos": None, "kokkos_cuda": False,
            "kokkos_precision": None}
    first = out.strip().splitlines()[0] if out.strip() else ""
    m = re.search(r"Simulator - (.+)$", first)
    info["version"] = m.group(1).strip() if m else first
    m = re.search(r"Installed packages:\s*\n\s*\n(.*?)\n\s*\n", out, re.S)
    if m:
        info["packages"] = m.group(1).split()
    m = re.search(r"KOKKOS package API:(.*)", out)
    if m:
        info["kokkos"] = m.group(1).strip()
        info["kokkos_cuda"] = "CUDA" in m.group(1)
    m = re.search(r"KOKKOS package precision:(.*)", out)
    if m:
        info["kokkos_precision"] = m.group(1).strip()
    info["kokkos_lines"] = [line.strip() for line in out.splitlines() if "KOKKOS package" in line or line.startswith("Kokkos")]
    return info


def missing_for_mliap(info: dict) -> list[str]:
    missing = [p for p in REQUIRED_MLIAP if p not in info["packages"]]
    if "KOKKOS" in info["packages"] and not info["kokkos_cuda"]:
        missing.append("KOKKOS with a CUDA backend")
    return missing


def command(exe: str, infile: str, ngpu: int = 1, kokkos: bool = True, skiprun: bool = False,
            mpirun: str | None = None) -> list[str]:
    """[mpirun -np N] lmp [-k on g N -sf kk -pk kokkos newton on neigh half] [-sr] -log none -in FILE."""
    cmd = []
    if ngpu > 1:
        launcher = mpirun or shutil.which("mpirun") or shutil.which("mpiexec")
        if not launcher:
            raise RuntimeError("more than one GPU needs mpirun / mpiexec on PATH")
        cmd += [launcher, "-np", str(ngpu)]
    cmd.append(exe)
    if kokkos:
        cmd += ["-k", "on", "g", str(ngpu), *KOKKOS_ARGS]
    if skiprun:
        cmd.append("-sr")
    return cmd + ["-log", "none", "-in", infile]


def run(cmd: list[str], cwd: Path, out_file: Path | None = None, timeout: float | None = None,
        env: dict | None = None) -> subprocess.CompletedProcess:
    """Blocking run; screen output kept (and written to out_file) for diagnosis."""
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                          env=dict(os.environ, **(env or {})), stdin=subprocess.DEVNULL)
    if out_file:
        out_file.write_text(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
    return proc


def lammps_errors(text: str) -> list[str]:
    """ERROR lines LAMMPS prints before it aborts — the first one is usually the real cause."""
    return [line.strip() for line in text.splitlines() if line.lstrip().startswith("ERROR")]
