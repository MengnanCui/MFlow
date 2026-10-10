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
