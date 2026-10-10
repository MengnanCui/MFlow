"""Active learning: selection maths, TFlow import rules, and a 2-round campaign with EMT as the 'DFT'."""

import json
import shutil
from argparse import Namespace

import ase.io
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from conftest import labelled
from mflow.al.select import fps
from mflow.al.tflow import geometry_key, import_tflow, scf_converged


def test_fps_picks_spread_points_and_respects_reference():
    X = np.array([[0.0], [0.1], [5.0], [10.0], [10.1]])
    assert sorted(fps(X, 3)) == [0, 2, 3] or sorted(fps(X, 3)) == [0, 2, 4]
    picks = fps(X, 1, X_ref=np.array([[0.0], [10.0]]))
    assert picks == [2]                                       # farthest from what is already labelled


def test_scf_convergence_from_oszicar(tmp_path):
    (tmp_path / "INCAR").write_text("NELM = 5\n")
    osz = "".join(f"DAV:   {i}   -0.1E+02   0.1E-01\n" for i in range(1, 6)) + "   1 F= -.10E+02 E0= -.10E+02\n"
    (tmp_path / "OSZICAR").write_text(osz)
    assert scf_converged(tmp_path)[0] is False
    (tmp_path / "OSZICAR").write_text(osz.replace("DAV:   5", "DAV:   x").replace("DAV:   x   -0.1E+02   0.1E-01\n", ""))
    assert scf_converged(tmp_path)[0] is True


def fake_tflow(tmp_path, cands, status="done", error_fix=False):
    calc = tmp_path / "calc"
    job_map = {}
    for i, a in enumerate(cands):
        d = calc / f"job{i}"
        (d / "static").mkdir(parents=True)
        (d / ".htcflow_result.json").write_text(json.dumps({"status": status if i == 0 else "done"}))
        if error_fix and i == 1:
            (d / "static" / "error_job_1").mkdir()
        job_map[f"job{i}"] = {"path": str(d), "info": {"mflow_id": a.info["mflow_id"]}}
    (calc / "job_map.json").write_text(json.dumps(job_map))
    return calc


def test_tflow_import_rejections(tmp_path):
    cands = []
    for i in range(3):
        a = bulk("Cu", "fcc", a=3.6, cubic=True)
        a.info["mflow_id"] = f"r00_c{i:05d}"
        cands.append(a)
    calc = fake_tflow(tmp_path, cands, status="failed", error_fix=True)
    wanted = {a.info["mflow_id"]: a for a in cands}
    accepted, rejects = import_tflow(calc, wanted)
    reasons = {r["mflow_id"]: r["reason"] for r in rejects}
    assert accepted == []
    assert "status failed" in reasons["r00_c00000"]
    assert "error fix" in reasons["r00_c00001"]
    assert "OUTCAR" in reasons["r00_c00002"]


def test_geometry_key_is_order_independent_and_detects_change():
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    b = a[[3, 1, 2, 0]]
    assert geometry_key(a) == geometry_key(b)
    c = a.copy()
    c.set_cell(c.cell * 1.05, scale_atoms=True)
    assert geometry_key(a) != geometry_key(c)


def emt_label(src, dst):
    out = []
    for a in ase.io.read(src, ":"):
        a.calc = EMT()
        a.info["REF_energy"] = a.get_potential_energy()
        a.arrays["REF_forces"] = a.get_forces()
        a.calc = None
        out.append(a)
    ase.io.write(dst, out)


pytest.importorskip("mace")


@pytest.mark.skipif(not shutil.which("mace_run_train"), reason="mace_run_train not on PATH")
def test_two_round_campaign(tmp_path, monkeypatch):
    from mflow.al.campaign import Campaign, cmd_al_init
    from mflow.al.commands import cmd_al_import, cmd_al_run, cmd_al_status
    from mflow.train.run import train_one

    monkeypatch.chdir(tmp_path)
    ase.io.write("data.xyz", labelled(30, seed=1))
    ase.io.write("test.xyz", labelled(6, seed=2))
    ase.io.write("system.xyz", labelled(3, seed=3)[:2])
    base = Namespace(input="data.xyz", index=":", name="base", outdir="base", test=None, valid=None, ref_prefix=None,
                     group=None, mode="scratch", foundation=None, head=None, pt=None, epochs=2, batch=8,
                     dtype="float64", device="cpu", cueq="off", seed=1, dry=False, resume=False, force=True,
                     no_plot=True, set=["num_channels=8", "max_L=0", "r_max=4.0"], committee=1, bagging=None)
    assert train_one(base) == 0
    assert cmd_al_init(Namespace(name="camp", system="system.xyz", data="data.xyz", test="test.xyz",
                                 foundation=str(tmp_path / "base" / "base.model"), mode="finetune",
                                 committee=2, engine="ase", force=False)) == 0
    with open("camp/campaign.yaml", "a") as fh:
        fh.write("train: {committee: 2, epochs: 1, dtype: float64, device: cpu}\nseed: {n_select: 4}\n"
                 "explore:\n  md: {temps: [300], steps: 40, dump: 10}\n  perturb: {n: 3}\n"
                 "select: {max_per_round: 4}\n")
    run = Namespace(name="camp", lmp=None, force=False)
    assert cmd_al_run(run) == 0                                    # round 0 → export
    emt_label("camp/round_00/export/candidates.extxyz", "lab0.xyz")
    assert cmd_al_import(Namespace(name="camp", round=None, tflow=None, labelled="lab0.xyz")) == 0
    assert cmd_al_run(run) == 0                                    # train → explore → select → export
    camp = Campaign("camp")
    assert camp.state["round"] == 1 and len(camp.committee()) == 2
    steps = [x["step"] for x in camp.ledger()]
    assert steps[:6] == ["init", "explore", "select", "export", "import", "train"]
    assert cmd_al_status(Namespace(name="camp")) == 0
    assert (tmp_path / "camp" / "status.json").exists()


def test_minhop_explorer_finds_minima_of_a_cluster(tmp_path):
    from ase import Atoms
    from mflow.al.explore import explore_minhop
    rng, pos = np.random.default_rng(0), []
    while len(pos) < 8:                                       # random, loosely packed Cu8: not a minimum
        p = rng.uniform(0, 5, 3)
        if all(np.linalg.norm(p - q) > 2.3 for q in pos):
            pos.append(p)
    a = Atoms("Cu8", positions=pos)
    a.center(vacuum=6.0)
    out = explore_minhop([a], {"steps": 4, "T0": 1000.0, "fmax": 0.05}, "unused", tmp_path, calc=EMT())
    minima = [x for x in out if x.info["mflow_source"] == "minhop_min"]
    assert len(minima) >= 2                                   # the start minimum and at least one new one
    energies = sorted({round(x.info["mflow_E_explore"], 3) for x in minima})
    assert len(energies) >= 2
    assert any(x.info["mflow_source"] == "minhop_md" for x in out)
    assert all(x.calc is None for x in out)
    assert (tmp_path / "minhop_0" / "hop.log").exists()      # ASE's files stay out of the caller's cwd
