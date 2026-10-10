import json
from argparse import Namespace

import ase.io
import pytest
from ase.build import bulk

from mflow.lmp.inputs import MDParams, md_input, read_back, species_for, write_data
from mflow.lmp.run import cmd_md


def test_species_order_and_unknown_elements():
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    a.numbers[0] = 13
    assert species_for([a], ["Cu", "Al", "O"]) == ["Al", "Cu"]
    with pytest.raises(ValueError):
        species_for([a], ["Cu"])


def test_data_roundtrip_keeps_order(tmp_path):
    a = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    a.numbers[::3] = 13
    a.rattle(0.05, seed=1)
    write_data(a, tmp_path / "data.lmp", ["Al", "Cu"])
    b = read_back(tmp_path / "data.lmp", ["Al", "Cu"])
    assert b.get_chemical_symbols() == a.get_chemical_symbols()
    assert abs(b.get_volume() - a.get_volume()) < 1e-6


def test_input_template():
    p = MDParams(ensemble="npt", temp=500, press=10, dt=2, steps=1000, species=["Al", "Cu"],
                 pair_style="mliap unified m.pt 0")
    text = md_input(p)
    assert "pair_coeff      * * Al Cu" in text
    assert "fix             md all npt temp 500 500 0.2 iso 10 10 2" in text
    assert "timestep        0.002" in text and "run             1000 upto" in text
    assert "dump_modify     traj element Al Cu sort id" in text
    assert "velocity" not in md_input(p, resume_from="restart.a")


def lj_deploy(tmp_path):
    d = tmp_path / "deploy"
    d.mkdir()
    (d / "deploy.json").write_text(json.dumps({
        "elements": ["Cu"], "r_max": 6.0, "dtype": "float64", "kokkos": False, "exported": "lj",
        "pair_style": "lj/cut 6.0", "pair_coeff": "* * 0.4093 2.338", "pair_extra": ["pair_modify shift yes"],
        "reference": {"type": "lj", "epsilon": 0.4093, "sigma": 2.338, "rc": 6.0, "dtype": "float64"}}))
    return d


def md_args(tmp_path, deploy, lmp, **kw):
    base = dict(input=str(tmp_path / "start.xyz"), index="-1", deploy=str(deploy), ensemble="nvt", temp=300.0,
                temp_end=None, press=1.0, dt=1.0, steps=600, tdamp=None, pdamp=None, seed=1, dump=50,
                thermo=20, restart=300, ngpu=1, lmp=lmp, outdir=str(tmp_path / "md"), skip=0.2,
                nocheck=False, dry=False, resume=False, force=False, no_plot=True)
    base.update(kw)
    return Namespace(**base)


def test_md_pipeline_with_lj(tmp_path, lmp_exe):
    """inputs -> skiprun -> run-0 consistency (rotation, units, virial sign) -> run -> gates."""
    a = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((3, 3, 3))
    a.rattle(0.05, seed=2)
    ase.io.write(tmp_path / "start.xyz", a)
    deploy = lj_deploy(tmp_path)
    assert cmd_md(md_args(tmp_path, deploy, lmp_exe)) == 0
    report = json.loads((tmp_path / "md" / "md_report.json").read_text())
    checks = {c["name"]: c for c in report["checks"]}
    assert checks["consistency: energy"]["value"] < 1e-8
    assert checks["consistency: forces"]["value"] < 1e-8
    assert checks["consistency: pressure"]["status"] == "pass"
    assert report["steps_done"] == 600

    assert cmd_md(md_args(tmp_path, deploy, lmp_exe, steps=900, resume=True)) == 0
    report = json.loads((tmp_path / "md" / "md_report.json").read_text())
    assert report["steps_done"] == 900


def test_md_blow_up_is_caught(tmp_path, lmp_exe):
    a = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((3, 3, 3))
    ase.io.write(tmp_path / "start.xyz", a)
    code = cmd_md(md_args(tmp_path, lj_deploy(tmp_path), lmp_exe, ensemble="nve", dt=30.0, steps=2000))
    assert code == 2
    report = json.loads((tmp_path / "md" / "md_report.json").read_text())
    assert report["status"] == "fail"
