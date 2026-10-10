# -*- coding: utf-8 -*-
"""
mflow deploy — turn a .model into what LAMMPS loads, plus deploy.json that pins it down.

    deploy/
      <stem>.model                      copy of the source (mace_create_lammps_model writes next to it)
      <stem>.model-mliap_lammps.pt      pair_style mliap unified <this> 0
      deploy.json                       source + export hashes, head, elements, r_max, dtype, versions

`mflow md` reads only deploy.json, so model ↔ head ↔ element types ↔ precision come from
one place. The `reference` block tells `md` how to recompute the same energies and forces
outside LAMMPS for its consistency check (the MACE ASE calculator on the source model).
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .. import __version__
from ..log import LOG
from ..report import Report


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inspect_model(path: Path) -> dict:
    """Heads, elements and cutoff of a MACE .model, read on the CPU."""
    import torch
    from ase.data import chemical_symbols
    try:
        net = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        net = torch.load(str(path), map_location="cpu")
    z = [int(x) for x in net.atomic_numbers]
    return {"heads": list(getattr(net, "heads", []) or []), "atomic_numbers": z,
            "elements": [chemical_symbols[x] for x in z], "r_max": float(net.r_max),
            "model_dtype": str(next(net.parameters()).dtype).replace("torch.", "")}


def package_versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("mace", "torch", "cuequivariance", "cuequivariance_torch", "cupy"):
        try:
            out[name] = getattr(__import__(name), "__version__", "?")
        except Exception:
            out[name] = None
    try:
        import torch
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        out["cuda"] = torch.version.cuda
    except Exception:
        pass
    return out


def mace_cli(name: str) -> str:
    return shutil.which(name) or str(Path(sys.executable).with_name(name))


def cmd_deploy(args) -> int:
    src = Path(args.model).expanduser().resolve()
    outdir = Path(args.outdir).resolve()
    LOG.rule(f"MFlow {__version__} · deploy")
    LOG.kv({"model": str(src), "head": args.head or "auto", "dtype": args.dtype, "outdir": str(outdir)})
    rep = Report("deploy", source=str(src))
    if not src.is_file():
        rep.check("model file", "fail", f"not found: {src}")
        rep.show()
        return 2

    info = inspect_model(src)
    heads = info["heads"]
    LOG(f"model: [key]{len(info['elements'])}[/key] elements ({' '.join(info['elements'])}) · "
        f"r_max {info['r_max']:.2f} Å · heads {', '.join(heads) or '—'} · stored as {info['model_dtype']}")
    head = args.head
    if len(heads) > 1 and not head:
        rep.check("head", "fail", f"multi-head model, choose one with -head: {', '.join(heads)}",
                  suggest="ML-IAP runs a single head: pass -head (usually your fine-tuned head, "
                          "e.g. Default, not pt_head)")
    elif head and heads and head not in heads:
        rep.check("head", "fail", f"head {head!r} not in {', '.join(heads)}")
    else:
        head = head or (heads[0] if heads else None)
        rep.check("head", "pass", head or "single-head model")

    versions = package_versions()
    gpu = versions.get("gpu")
    rep.check("export device", "pass" if gpu else "warn", gpu or "no CUDA device visible",
              suggest="export on the GPU (same architecture) that will run LAMMPS — the MACE docs require it")
    if not versions.get("cuequivariance_torch"):
        rep.check("cuequivariance", "fail", "cuequivariance_torch not importable",
                  suggest="pip install cuequivariance-torch (and cupy-cuda12x): the mliap export converts "
                          "the model to cuEquivariance kernels")
    if rep.failed:
        rep.show()
        rep.write(outdir / "deploy_report.json")
        return 2

    outdir.mkdir(parents=True, exist_ok=True)
    local = outdir / src.name
    if local != src:
        shutil.copy2(src, local)
    cmd = [mace_cli("mace_create_lammps_model"), str(local), "--format", "mliap", "--dtype", args.dtype]
    if head:
        cmd += ["--head", head]
    LOG(f"[dim]{' '.join(cmd)}[/dim]")
    proc = subprocess.run(cmd, cwd=outdir, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    (outdir / "create_lammps_model.out").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    exported = Path(str(local) + "-mliap_lammps.pt")
    if proc.returncode != 0 or not exported.exists():
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-20:]
        rep.check("export", "fail", f"mace_create_lammps_model exit {proc.returncode}", value=tail,
                  suggest=f"see {outdir / 'create_lammps_model.out'}")
        LOG("[err]export failed:[/err]\n" + "\n".join(tail))
        rep.show()
        rep.write(outdir / "deploy_report.json")
        return 2
    rep.check("export", "pass", exported.name)

    deploy = {
        "mflow": __version__, "created": datetime.now().isoformat(timespec="seconds"),
        "source": str(src), "source_sha256": sha256(src),
        "exported": str(exported), "exported_sha256": sha256(exported),
        "format": "mliap", "head": head, "dtype": args.dtype,
        "elements": info["elements"], "atomic_numbers": info["atomic_numbers"], "r_max": info["r_max"],
        "pair_style": f"mliap unified {exported} 0", "kokkos": True,
        "reference": {"type": "mace", "model": str(src), "head": head, "dtype": args.dtype},
        "versions": versions,
    }
    path = outdir / "deploy.json"
    path.write_text(json.dumps(deploy, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG(f"deploy [key]{path}[/key]")
    rep.show()
    rep.write(outdir / "deploy_report.json")
    LOG("next: [hi]mflow lmp-check -deploy " + str(path) + "[/hi]")
    return 0


def load_deploy(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "deploy.json"
    deploy = json.loads(path.read_text(encoding="utf-8"))
    deploy["_path"] = str(path.resolve())
    return deploy
