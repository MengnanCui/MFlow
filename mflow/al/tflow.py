# -*- coding: utf-8 -*-
"""
Hand-off to TFlow (DFT labelling) and back.

Export (MFlow -> TFlow), round_NN/export/:
    candidates.extxyz   geometry only; info: mflow_id (letter first, [A-Za-z0-9._-]), mflow_round,
                        mflow_source, mflow_dev, config_type. No energy/forces/stress keys: ASE would
                        turn them into a calculator and TFlow would carry them into job_map.json.
    run_label.py        per-job script for `vaspctl submit --src-run` (static single point with
                        labelling settings; edit the INCAR block to match your foundation model)
    manifest.json       counts, elements, ids, the TFlow commands
    README.txt          the commands

    vaspctl prepare <calc_dir> candidates.extxyz --id-key mflow_id --tag mflow_rNN
    vaspctl submit  <calc_dir> --src-run run_label.py --src-submit <your submit script> --max-concurrent 20 --watch
    vaspctl report  <calc_dir> --task-dirs static --no-queue

Import (TFlow -> MFlow), either
    -tflow <calc_dir>   read job_map.json, keep entries whose info.mflow_id belongs to this round,
                        parse <job>/static/OUTCAR with ASE, and REJECT a frame when
                          - .htcflow_result.json status != done
                          - the last ionic step used >= NELM electronic steps (SCF not converged:
                            TFlow does not flag this for static runs)
                          - an error fix ran (static/error_job_* exists) or KPOINTS was modified
                          - the structure changed (TFlow's volume-scaling fixes rewrite POSCAR)
    -in <labelled.xyz>  any extxyz with mflow_id + reference labels (REF_ / dft_ / plain)

REF_energy is VASP's free energy (TOTEN, whose derivative the forces are); the sigma->0 energy is
kept as REF_energy_sigma0. REF_stress is ASE's convention (eV/Å³, VASP sign flipped).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import ase.io
from ase import Atoms

from ..log import LOG, table

RUN_LABEL = '''#!/usr/bin/env python3
# Written by MFlow for TFlow: one DFT single point for MLIP labelling, in ./static/.
# Keep these settings IDENTICAL across all rounds of a campaign, and consistent with the
# foundation model you fine-tune (see MFlow README 1.1). Values below follow MatPES-PBE-like
# labelling settings; switch addU / MAGMOM / ENCUT / KSPACING to your convention once.
import ase.io
from htcflow import MPFlow
from htcflow.overrides import apply_overrides

atoms = ase.io.read("POSCAR")
flow = MPFlow(atoms=atoms, outdir=".", tasktype="custom")
job = flow.create_static_job(mode="native", addU=False)
job.append_custom_settings(dict(
    ENCUT=680, KSPACING=0.22, EDIFF=1e-5, ISMEAR=0, SIGMA=0.05, ISPIN=2, LASPH=True,
    LREAL=False, ALGO="Normal", NELM=200, PREC="Accurate",
    LWAVE=False, LCHARG=False, LAECHG=False, LVTOT=False, LORBIT="none", NEDOS="none",
))
flow.jobs.append(job)
apply_overrides(flow)
flow.chain_jobs()
flow.run_flow()
'''


# ============================================================================ #
#  export
# ============================================================================ #

def clean_for_export(atoms: Atoms) -> Atoms:
    keep = {k: v for k, v in atoms.info.items()
            if (k.startswith("mflow_") or k == "config_type") and np.isscalar(v)}
    out = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
    out.info.update(keep)
    return out


def export_round(camp, round_: int, selected: list) -> Path:
    out = camp.round_dir(round_, "export")
    frames = [clean_for_export(a) for a in selected]
    ase.io.write(out / "candidates.extxyz", frames, format="extxyz")
    (out / "run_label.py").write_text(RUN_LABEL, encoding="utf-8")
    tag = f"mflow_r{round_:02d}"
    cmds = [f"vaspctl prepare <calc_dir> {out / 'candidates.extxyz'} --id-key mflow_id --tag {tag}",
            f"vaspctl submit <calc_dir> --src-run {out / 'run_label.py'} --src-submit <submit.sh> "
            f"--max-concurrent 20 --auto-resume --watch",
            "vaspctl report <calc_dir> --task-dirs static --no-queue",
            f"mflow al import -name {camp.root.name} -round {round_} -tflow <calc_dir>"]
    elements = sorted({s for a in frames for s in a.get_chemical_symbols()})
    manifest = {"round": round_, "tag": tag, "n": len(frames), "elements": elements,
                "atoms_total": int(sum(len(a) for a in frames)),
                "ids": [a.info["mflow_id"] for a in frames], "commands": cmds}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "README.txt").write_text("\n".join(cmds) + "\n", encoding="utf-8")
    LOG(f"export [key]{out / 'candidates.extxyz'}[/key] · {len(frames)} structures · "
        f"{manifest['atoms_total']:,} atoms · elements {' '.join(elements)}")
    LOG("[hi]hand-off to TFlow:[/hi]\n  " + "\n  ".join(cmds))
    return out


# ============================================================================ #
#  import
# ============================================================================ #

def geometry_key(atoms: Atoms) -> tuple:
    """Order-independent fingerprint, like TFlow's structure_hash (cell 3 dp, fractional 6 dp)."""
    if atoms.cell.rank == 3:
        cell = tuple(np.round(np.array(atoms.cell), 3).ravel())
        pos = np.round(atoms.get_scaled_positions(wrap=True), 4) % 1.0
    else:
        cell, pos = None, np.round(atoms.get_positions(), 4)
    return cell, tuple(sorted((s, tuple(p)) for s, p in zip(atoms.get_chemical_symbols(), pos.tolist())))


def scf_converged(static: Path) -> tuple[bool | None, str]:
    """Electronic steps of the last ionic step in OSZICAR against NELM in INCAR."""
    osz, incar = static / "OSZICAR", static / "INCAR"
    if not osz.exists():
        return None, "no OSZICAR"
    nelm = 60
    if incar.exists():
        m = re.search(r"^\s*NELM\s*=\s*(\d+)", incar.read_text(errors="replace"), re.M | re.I)
        nelm = int(m.group(1)) if m else nelm
    steps, last = 0, 0
    for line in osz.read_text(errors="replace").splitlines():
        m = re.match(r"\s*(?:DAV|RMM|CG|SDA):\s*(\d+)", line)
        if m:
            steps = int(m.group(1))
        elif re.match(r"\s*\d+\s+F=", line):
            last, steps = steps, 0
    last = last or steps
    return last < nelm, f"{last} electronic steps (NELM {nelm})"


def label_from_outcar(path: Path) -> Atoms:
    a = ase.io.read(path, format="vasp-out", index=-1)
    res = a.calc.results
    out = Atoms(numbers=a.numbers, positions=a.positions, cell=a.cell, pbc=a.pbc)
    out.info["REF_energy"] = float(res.get("free_energy", res["energy"]))
    out.info["REF_energy_sigma0"] = float(res["energy"])
    out.arrays["REF_forces"] = np.asarray(res["forces"])
    if "stress" in res:
        from ase.stress import voigt_6_to_full_3x3_stress
        out.info["REF_stress"] = voigt_6_to_full_3x3_stress(np.asarray(res["stress"])).ravel()
    return out


def import_tflow(calc_dir: Path, wanted: dict) -> tuple[list, list]:
    """wanted: mflow_id -> candidate Atoms. Returns (accepted frames, rejects)."""
    job_map = json.loads((calc_dir / "job_map.json").read_text(encoding="utf-8"))
    accepted, rejects, seen = [], [], set()
    for name, entry in job_map.items():
        mid = str((entry.get("info") or {}).get("mflow_id", ""))
        if mid not in wanted:
            continue
        seen.add(mid)
        job = Path(entry.get("path") or calc_dir / name)
        if not job.is_absolute():
            job = calc_dir / job
        static = job / "static"

        def reject(reason):
            rejects.append({"mflow_id": mid, "dir": str(job), "reason": reason})

        sentinel = job / ".htcflow_result.json"
        status = json.loads(sentinel.read_text()).get("status") if sentinel.exists() else None
        if status != "done":
            reject(f"TFlow status {status or 'missing'}")
            continue
        # an applied error fix is the more informative reason, so it is checked before OUTCAR
        if list(static.glob("error_job_*")):
            reject("an error fix ran (static/error_job_*)")
            continue
        kp = static / "KPOINTS"
        if kp.exists() and "Modified by VaspErrorFixer" in kp.read_text(errors="replace"):
            reject("KPOINTS modified by the error fixer")
            continue
        if not (static / "OUTCAR").exists():
            reject("no static/OUTCAR")
            continue
        ok, msg = scf_converged(static)
        if ok is False:
            reject(f"SCF not converged: {msg}")
            continue
        try:
            frame = label_from_outcar(static / "OUTCAR")
        except Exception as exc:
            reject(f"OUTCAR unreadable: {type(exc).__name__}: {exc}")
            continue
        if geometry_key(frame) != geometry_key(wanted[mid]):
            reject("structure changed during the calculation (error fix?)")
            continue
        frame.info.update({k: v for k, v in wanted[mid].info.items() if k.startswith("mflow_") or k == "config_type"})
        frame.info["dft_scf"] = msg
        accepted.append(frame)
    for mid in wanted:
        if mid not in seen:
            rejects.append({"mflow_id": mid, "dir": None, "reason": "not in TFlow job_map.json (not prepared?)"})
    return accepted, rejects


def import_labelled_xyz(path: Path, wanted: dict) -> tuple[list, list]:
    from ..data import find_ref, keys, read_dataset
    from ..train.check import stress_key
    frames = read_dataset(str(path))
    ref = find_ref(frames, exclude="mflow_")
    accepted, rejects, seen = [], [], set()
    for a in frames:
        mid = str(a.info.get("mflow_id", ""))
        if mid not in wanted:
            continue
        seen.add(mid)
        ekey, fkey = keys(ref or "")
        if ekey not in a.info or fkey not in a.arrays:
            rejects.append({"mflow_id": mid, "dir": None, "reason": "no reference energy/forces"})
            continue
        out = Atoms(numbers=a.numbers, positions=a.positions, cell=a.cell, pbc=a.pbc)
        out.info["REF_energy"] = float(a.info[ekey])
        out.arrays["REF_forces"] = np.asarray(a.arrays[fkey])
        if stress_key(ref) in a.info:
            out.info["REF_stress"] = a.info[stress_key(ref)]
        out.info.update({k: v for k, v in wanted[mid].info.items() if k.startswith("mflow_") or k == "config_type"})
        accepted.append(out)
    rejects += [{"mflow_id": m, "dir": None, "reason": "missing from the labelled file"} for m in wanted if m not in seen]
    return accepted, rejects


def show_rejects(rejects: list):
    if not rejects:
        return
    reasons = {}
    for r in rejects:
        key = r["reason"].split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    LOG(table("rejected labels", ["reason", "frames"], sorted(reasons.items(), key=lambda kv: -kv[1]),
              styles=["key"]))
