# -*- coding: utf-8 -*-
"""Small synthetic data shared by the tests: EMT-labelled Cu/Al cells, no DFT, no GPU."""

import shutil

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT


def labelled(n=30, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        kind = ["Cu", "Al", "CuAl"][i % 3]
        if kind == "CuAl":
            a = bulk("Cu", "fcc", a=3.8, cubic=True).repeat((2, 1, 1))
            a.numbers[::2] = 13
        else:
            a = bulk(kind, "fcc", a={"Cu": 3.6, "Al": 4.05}[kind], cubic=True).repeat((2, 1, 1))
        a.set_cell(a.cell * (1 + rng.normal(0, 0.02)), scale_atoms=True)
        a.rattle(0.08, seed=seed * 1000 + i)
        a.calc = EMT()
        a.info["REF_energy"] = a.get_potential_energy()
        a.arrays["REF_forces"] = a.get_forces()
        a.info["REF_stress"] = a.get_stress(voigt=False).ravel()
        a.info["config_type"] = kind
        a.calc = None
        frames.append(a)
    return frames


@pytest.fixture
def frames():
    return labelled()


@pytest.fixture
def lmp_exe():
    exe = shutil.which("lmp")
    if not exe:
        pytest.skip("no lmp executable on PATH")
    return exe


def sutton_chen_cu(path, cutoff: float = 7.0):
    """
    Sutton–Chen EAM for Cu (eps 12.382 meV, a 3.61 Å, c 39.432, n 9, m 6; Phil. Mag. Lett. 61, 139 (1990))
    written as a LAMMPS eam/alloy setfl file. Pair term shifted to zero at the cutoff. A metal that melts
    like one (T_m ≈ 950 K with this 7 Å cutoff, from MFlow's own coexistence runs), for testing
    protocols without a MACE model.
    """
    import numpy as np
    eps, a, c, n, m = 0.012382, 3.61, 39.432, 9, 6
    nr, nrho = 2000, 2000
    dr = cutoff / (nr - 1)
    r = np.arange(nr) * dr
    r[0] = 1e-3
    rho = (a / r) ** m - (a / cutoff) ** m
    rho[r > cutoff] = 0
    phi = eps * ((a / r) ** n - (a / cutoff) ** n)
    rho_max = 400.0                                  # fcc Cu at a0 is ~150; compressed MD stays below
    drho = rho_max / (nrho - 1)
    F = -eps * c * np.sqrt(np.arange(nrho) * drho)
    rho[0], phi[0] = rho[1], phi[1]
    lines = ["Sutton-Chen Cu (MFlow tests)", "eps 12.382 meV a 3.61 c 39.432 n 9 m 6", "",
             "1 Cu", f"{nrho} {drho:.12e} {nr} {dr:.12e} {cutoff}", "29 63.546 3.61 fcc"]
    fmt = lambda v: "\n".join(f"{x:.12e}" for x in v)
    path.write_text("\n".join(lines) + "\n" + fmt(F) + "\n" + fmt(rho) + "\n" + fmt(r * phi) + "\n")
    return path
