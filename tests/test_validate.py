"""mflow validate: dimer shape metrics, EOS, phonons and defects with EMT."""

from argparse import Namespace

import numpy as np
import pytest
import ase.io
from ase.build import bulk

from mflow.validate import cmd_validate, dimer_metrics, parse_ref, tick_labels


def morse(r, D=2.0, a=1.5, r0=2.3):
    x = np.exp(-a * (r - r0))
    return D * (x ** 2 - 2 * x), 2 * a * D * (x ** 2 - x)       # E, F = -dE/dr


def test_dimer_metrics_physical_curve():
    r = np.linspace(1.0, 6.0, 501)                         # 0.01 Å, as MLIP Arena
    E, F = morse(r)
    m = dimer_metrics(r, E, F)
    assert m["force_flips"] == 1 and m["energy_diff_flips"] == 1
    assert m["repulsion_violations"] == 0 and m["E_short"] > 0
    assert abs(m["r_eq"] - 2.3) < 0.03 and abs(m["well_depth"] - 2.0) < 0.02
    assert m["tortuosity"] == pytest.approx(1.0, abs=1e-6)
    assert m["spearman_repulsion"] == pytest.approx(-1.0) and m["spearman_attraction"] == pytest.approx(1.0)
    assert m["conservation_deviation"] < 0.05 and m["energy_jump"] < 0.01


def test_dimer_metrics_catch_a_hole_in_the_wall():
    r = np.linspace(1.0, 6.0, 501)                         # 0.01 Å, as MLIP Arena
    E, F = morse(r)
    hole = r < 1.4                                             # the wall collapses at short range
    E, F = E.copy(), F.copy()
    E[hole], F[hole] = -5.0 + (r[hole] - 1.0), -1.0
    m = dimer_metrics(r, E, F)
    assert m["E_short"] <= m["E_tail"]                         # what `validate dimer` fails on
    assert m["energy_diff_flips"] > 1 and m["energy_jump"] > 0.05


def test_parse_ref_and_tick_labels():
    assert parse_ref("V0=11.8, B0=140") == {"V0": 11.8, "B0": 140.0}
    labels = ["G", "X", "U", "K", "G", "L"]
    assert tick_labels(labels, [True, False, True, True]) == ["G", "X", "U|K", "G", "L"]


def args(kind, tmp_path, **kw):
    base = dict(kind=kind, model="emt", head=None, dtype="float64", input=None, index="-1", elements=None,
                rmin=0.9, rmax=None, dr=0.02, npts=11, strain=0.04, relax=True, supercell=None, mesh=6,
                displacement=0.01, min_width=8.0, surface=None, layers=4, ref=None,
                outdir=str(tmp_path), no_plot=False)
    base.update(kw)
    return Namespace(**base)


@pytest.fixture
def cu(tmp_path):
    path = tmp_path / "cu.xyz"
    ase.io.write(path, bulk("Cu", "fcc", a=3.59, cubic=True))
    return str(path)


def test_validate_eos_and_defect_with_emt(tmp_path, cu):
    import json
    assert cmd_validate(args("eos", tmp_path, input=cu, ref="B0=134")) == 0
    rep = json.loads((tmp_path / "cu_eos.json").read_text())
    assert rep["status"] == "pass"
    assert 11.0 < rep["eos"]["V0_A3_atom"] < 12.2 and 110 < rep["eos"]["B0_GPa"] < 160
    assert (tmp_path / "cu_eos.png").exists()
    assert cmd_validate(args("defect", tmp_path, input=cu, surface="111")) == 0
    rep = json.loads((tmp_path / "cu_defect.json").read_text())
    assert 0.8 < rep["defect"]["vacancy"]["Cu"]["E_formation_eV"] < 1.6
    assert rep["defect"]["surface"]["111"]["gamma_J_m2"] > 0


def test_validate_dimer_writes_every_pair(tmp_path):
    import json
    cmd_validate(args("dimer", tmp_path, elements="Cu,Al"))
    rep = json.loads((tmp_path / "dimer_CuAl.json").read_text())
    assert set(rep["dimer"]["pairs"]) == {"Cu-Cu", "Cu-Al", "Al-Al"}
    assert all(p["force_flips"] >= 1 for p in rep["dimer"]["pairs"].values())


def test_validate_phonon_with_emt(tmp_path, cu):
    pytest.importorskip("phonopy")
    import json
    assert cmd_validate(args("phonon", tmp_path, input=cu, supercell="2")) == 0
    rep = json.loads((tmp_path / "cu_phonon.json").read_text())
    assert rep["phonon"]["n_imaginary_qpoints"] == 0 and 6 < rep["phonon"]["max_THz"] < 10
