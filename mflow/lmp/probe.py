# -*- coding: utf-8 -*-
"""
mflow lmp-check — can this LAMMPS run MACE through ML-IAP + KOKKOS on this GPU?

    1. lmp -h            KOKKOS (CUDA backend), ML-IAP, PYTHON, version
    2. embedded python   mace / torch / cuequivariance / cupy importable *inside* LAMMPS,
                         CUDA visible, versions against deploy.json
    3. with -deploy      2 atoms, run 0, through the exported model on the GPU
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .. import __version__
from ..log import LOG
from ..report import Report
from . import env
from .deploy import load_deploy

CMAKE_HINT = ("cmake ../cmake -D BUILD_MPI=ON -D BUILD_SHARED_LIBS=ON -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON "
              "-D Kokkos_ARCH_<YOURGPU>=ON -D PKG_ML-IAP=ON -D PKG_ML-SNAP=ON -D PKG_PYTHON=ON "
              "-D MLIAP_ENABLE_PYTHON=ON   (MACE docs: guide/lammps_mliap)")

PY_PROBE = '''python mflowprobe here """
def mflowprobe():
    import importlib, json
    out = {}
    for name in ("mace", "torch", "cuequivariance", "cuequivariance_torch", "cupy"):
        try:
            out[name] = getattr(importlib.import_module(name), "__version__", "?")
        except Exception as exc:
            out[name] = None
    try:
        import torch, sys
        out["cuda"] = torch.cuda.is_available()
        out["gpu"] = torch.cuda.get_device_name(0) if out["cuda"] else None
        out["python"] = sys.executable
    except Exception:
        pass
    print("MFLOWPROBE " + json.dumps(out), flush=True)
"""
python mflowprobe invoke
'''


def probe_python(exe: str, workdir: Path) -> dict | None:
    inp = workdir / "in.pyprobe"
    inp.write_text(PY_PROBE, encoding="utf-8")
    proc = env.run([exe, "-log", "none", "-in", inp.name], workdir, workdir / "pyprobe.out", timeout=300)
    m = re.search(r"MFLOWPROBE (\{.*\})", proc.stdout)
    return json.loads(m.group(1)) if m else None


def probe_model(exe: str, deploy: dict, workdir: Path, kokkos: bool) -> tuple[bool, str]:
    """Two atoms of the first element, 2.5 Å apart, in a 12 Å box: does the model load and evaluate?"""
    el = deploy["elements"][0]
    pair_coeff = deploy.get("pair_coeff", "* * {species}").format(species=el)
    lines = ["units metal", "atom_style atomic", "atom_modify map yes", "newton on", "boundary p p p",
             "region box block 0 12 0 12 0 12", "create_box 1 box",
             "create_atoms 1 single 5 6 6", "create_atoms 1 single 7.5 6 6", "mass 1 1.0",
             f"pair_style {deploy['pair_style']}", f"pair_coeff {pair_coeff}", *deploy.get("pair_extra", []),
             "thermo_style custom step pe press", "run 0"]
    inp = workdir / "in.modelprobe"
    inp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    proc = env.run(env.command(exe, inp.name, 1, kokkos), workdir, workdir / "modelprobe.out", timeout=900)
    errs = env.lammps_errors(proc.stdout + proc.stderr)
    if proc.returncode or errs:
        return False, errs[0] if errs else f"exit {proc.returncode}"
    m = re.search(r"^\s*0\s+(\S+)\s+(\S+)\s*$", proc.stdout, re.M)
    return True, f"run 0 ok · pe {m.group(1)} eV" if m else "run 0 ok"


def cmd_lmp_check(args) -> int:
    LOG.rule(f"MFlow {__version__} · lmp-check")
    workdir = Path(args.outdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    rep = Report("lmp-check")
    exe = env.find_lmp(args.lmp)
    if not exe:
        rep.check("lmp executable", "fail", "not found (tried -lmp, $MFLOW_LMP, PATH)",
                  suggest="pass -lmp /path/to/lmp or export MFLOW_LMP")
        return _done(rep, workdir)
    info = env.lmp_info(exe)
    rep.data["lammps"] = info
    LOG.kv({"lmp": exe, "version": info["version"], "KOKKOS": info["kokkos"] or "—",
            "precision": info["kokkos_precision"] or "—"})
    for pkg in env.REQUIRED_MLIAP:
        rep.check(f"package {pkg}", "pass" if pkg in info["packages"] else "fail",
                  "installed" if pkg in info["packages"] else "missing", suggest=f"rebuild LAMMPS: {CMAKE_HINT}")
    if "KOKKOS" in info["packages"]:
        rep.check("KOKKOS backend", "pass" if info["kokkos_cuda"] else "fail", info["kokkos"] or "?",
                  suggest=f"KOKKOS was built without CUDA: {CMAKE_HINT}")

    if "PYTHON" in info["packages"]:
        py = probe_python(exe, workdir)
        rep.data["embedded_python"] = py
        if py is None:
            rep.check("embedded python", "fail", "python command did not run — see pyprobe.out")
        else:
            LOG(f"[dim]LAMMPS python: {py.get('python')}[/dim]")
            for name in ("mace", "torch", "cuequivariance_torch", "cupy"):
                ok = py.get(name) is not None
                rep.check(f"python: {name}", "pass" if ok else "fail", py.get(name) or "not importable",
                          suggest=f"install {name} into the Python environment LAMMPS was built against "
                                  f"({py.get('python')})")
            rep.check("python: CUDA", "pass" if py.get("cuda") else "fail", py.get("gpu") or "no CUDA device",
                      suggest="torch inside LAMMPS sees no GPU: CPU torch build, or CUDA_VISIBLE_DEVICES")

    if args.deploy:
        deploy = load_deploy(args.deploy)
        py = rep.data.get("embedded_python") or {}
        want = deploy.get("versions", {})
        for name in ("mace", "cuequivariance_torch"):
            if want.get(name) and py.get(name) and want[name] != py[name]:
                rep.check(f"version {name}", "warn", f"exported with {want[name]}, LAMMPS has {py[name]}",
                          suggest="export the model with the same mace / cuequivariance versions LAMMPS uses")
        kokkos = bool(deploy.get("kokkos", True))
        if kokkos and env.missing_for_mliap(info):
            rep.check("model in LAMMPS", "fail", "skipped — the build lacks " + ", ".join(env.missing_for_mliap(info)))
        else:
            ok, msg = probe_model(exe, deploy, workdir, kokkos)
            rep.check("model in LAMMPS", "pass" if ok else "fail", msg,
                      suggest="the exported model does not run: see modelprobe.out")
    return _done(rep, workdir)


def _done(rep: Report, workdir: Path) -> int:
    LOG.rule("verdict")
    rep.show()
    rep.write(workdir / "lmp_env.json")
    return 0 if not rep.failed else 2
