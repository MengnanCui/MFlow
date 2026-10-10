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
                nocheck=False, dry=False, resume=False, force=False, no_plot=True, protocol="single",
                thermostat="nh", baro="iso", melt_temp=None, fmax=0.01)
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


def test_thermostat_barostat_and_protocol_inputs():
    from mflow.lmp.inputs import stages
    kw = dict(temp=500, press=10, dt=2, steps=1000, species=["Cu"], pair_style="lj/cut 6.0")
    text = md_input(MDParams(ensemble="npt", thermostat="csvr", baro="aniso", **kw))
    assert "fix             md all nph aniso 10 10 2" in text and "temp/csvr 500 500 0.2" in text
    text = md_input(MDParams(ensemble="nvt", thermostat="langevin", **kw))
    assert "fix             md all nve" in text and "langevin 500 500 0.2" in text and "zero yes" in text
    p = MDParams(protocol="transport", **kw)
    st = stages(p)
    assert [s.ensemble for s in st] == ["npt", "nvt", "nve"] and st[-1].end == 1000
    text = md_input(p)
    assert "change_box      all x scale ${s} y scale ${s} z scale ${s} remap" in text
    assert text.count("run             ") == 3
    resumed = md_input(p, resume_from="restart.a", resume_step=400)  # inside nvt: npt stage dropped
    assert "change_box" not in resumed and "run             300 upto" not in resumed
    assert "run             1000 upto" in resumed
    text = md_input(MDParams(protocol="minimize", baro="tri", **kw))
    assert "suffix          off" in text and "box/relax tri 10" in text and "jump            SELF relax_loop" in text
    q = stages(MDParams(protocol="quench", temp=1500, temp_end=300, steps=1000))
    assert (q[-1].temp, q[-1].temp_end) == (1500, 300)


def test_protocols_run_in_lammps(tmp_path, lmp_exe):
    """transport (npt → average volume → nvt → nve), minimize (box/relax loop) and melt, with LJ copper."""
    a = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((3, 3, 3))
    a.rattle(0.02, seed=3)
    ase.io.write(tmp_path / "start.xyz", a)
    deploy = lj_deploy(tmp_path)

    out = tmp_path / "transport"
    cmd_md(md_args(tmp_path, deploy, lmp_exe, protocol="transport", thermostat="csvr", steps=1500, dump=10,
                   outdir=str(out)))
    rep = json.loads((out / "md_report.json").read_text())
    names = {c["name"] for c in rep["checks"]}
    assert rep["production_stage"] == "nve" and "NVE energy drift" in names
    assert "MFLOW volume -> average" in (out / "log.lammps").read_text()
    assert rep["block_averages"]["Temp"]["se"] > 0

    out = tmp_path / "relax"
    assert cmd_md(md_args(tmp_path, deploy, lmp_exe, protocol="minimize", baro="iso", press=0.0, steps=2000,
                          fmax=1e-4, outdir=str(out))) == 0
    rep = json.loads((out / "md_report.json").read_text())
    relaxed = ase.io.read(out / "relaxed.xyz")
    assert rep["fmax_eV_A"] < 2e-4 and abs(rep["press_bar"]) < 500
    assert relaxed.get_volume() != pytest.approx(a.get_volume(), rel=1e-4)    # LJ a0 differs from 3.61 Å



def test_melt_protocol_with_eam(tmp_path, lmp_exe):
    """Coexistence mechanics with a Sutton–Chen Cu EAM: groups, frozen half, q6 slab profile, verdict."""
    from conftest import sutton_chen_cu
    pot = sutton_chen_cu(tmp_path / "Cu_SC.eam.alloy")
    d = tmp_path / "deploy"
    d.mkdir()
    (d / "deploy.json").write_text(json.dumps({
        "elements": ["Cu"], "r_max": 7.0, "dtype": "float64", "kokkos": False, "exported": "eam",
        "pair_style": "eam/alloy", "pair_coeff": f"* * {pot} {{species}}",
        "reference": {"type": "eam", "file": str(pot), "dtype": "float64"}}))
    ase.io.write(tmp_path / "start.xyz", bulk("Cu", "fcc", a=3.61, cubic=True).repeat((3, 3, 8)))
    out = tmp_path / "melt"
    cmd_md(md_args(tmp_path, d, lmp_exe, protocol="melt", temp=950, melt_temp=2000, steps=8000,
                   outdir=str(out)))                                   # this EAM melts at ~950 K
    rep = json.loads((out / "md_report.json").read_text())
    checks = {c["name"]: c for c in rep["checks"]}
    assert checks["consistency: energy"]["status"] == "pass"
    assert rep["production_stage"] == "coexistence" and "melt: phases" in checks
    assert (out / "phase_profile.dat").exists()
    assert rep["bonds_crystal_start"] > 6 > 3 > rep["bonds_liquid_start"]      # one crystal, one liquid half
    log = (out / "log.lammps").read_text()
    assert "atoms in group liquid" in log and "fix             freeze solid setforce" in log
