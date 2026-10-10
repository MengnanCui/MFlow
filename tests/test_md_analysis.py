import numpy as np
from ase.build import bulk

from mflow.analysis import gates
from mflow.analysis.thermo import merged, parse_log
from mflow.analysis.traj import diffusion, msd, rdf
from mflow.report import Report

LOG = """LAMMPS (22 Jul 2025)
   Step          Time           Temp          PotEng         TotEng
         0   0              300            -10.0          -9.0
        10   0.01           310            -10.1          -9.0
Loop time of 1.0 on 1 procs for 10 steps with 4 atoms
Performance: 12.5 ns/day, 1.92 hours/ns
Dangerous builds = 0
   Step          Time           Temp          PotEng         TotEng
        10   0.01           310            -10.1          -9.0
WARNING: something odd (src/foo.cpp:1)
        20   0.02           305            -10.2          -9.0
Loop time of 1.0 on 1 procs for 10 steps with 4 atoms
Total wall time: 0:00:02
"""


def test_parse_log_two_runs(tmp_path):
    path = tmp_path / "log.lammps"
    path.write_text(LOG)
    log = parse_log(path)
    assert len(log["runs"]) == 2 and log["finished"] and log["ns_per_day"] == [12.5]
    assert log["dangerous_builds"] == [0] and log["atoms"] == [4, 4]
    th = merged(log)
    assert th["Step"].tolist() == [0, 10, 20]             # repeated step 10 kept once


def test_nve_drift_gate():
    t = np.linspace(0, 10, 101)
    for drift, status in ((1e-5, "pass"), (1e-3, "warn"), (1e-2, "fail")):
        rep = Report("md")
        th = {"Time": t, "TotEng": -100 + drift * 10 * t}       # 10 atoms
        gates.gate_energy(th, "nve", 10, rep, 0.0)
        assert rep.checks[0]["status"] == status, drift


def test_temperature_gate():
    t = np.linspace(0, 10, 200)
    rep = Report("md")
    gates.gate_temperature({"Time": t, "Temp": 300 + np.sin(t)}, 300, "nvt", rep, 0.2, False)
    assert rep.status == "pass"
    rep = Report("md")
    gates.gate_temperature({"Time": t, "Temp": np.full_like(t, 400.0)}, 300, "nvt", rep, 0.2, False)
    assert rep.failed


def test_rdf_first_peak_fcc():
    a = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((4, 4, 4))
    g = rdf([a], rmax=6.0, nbins=300)
    assert abs(g["r"][np.argmax(g["Cu-Cu"])] - 3.6 / np.sqrt(2)) < 0.05


def test_msd_and_diffusion_of_ballistic_drift():
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    frames = []
    for k in range(40):                                    # random walk with known D
        b = a.copy()
        b.positions += np.sqrt(k) * 0.1 * np.array([1.0, 0.0, 0.0])
        frames.append(b)
    curve = msd(frames, dt_ps=1.0)
    assert curve["Cu"][0] == 0 and np.all(np.diff(curve["Cu"]) >= 0)
    assert diffusion(curve)["Cu"] > 0
