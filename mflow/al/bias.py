# -*- coding: utf-8 -*-
"""
Uncertainty-biased MD (after Zaverkin et al., npj Comput. Mater. 10, 83 (2024); Kulichenko et al.,
Nat. Comput. Sci. 3, 230 (2023)): the committee mean minus kappa times the energy spread,

    E_b = <E> - kappa * sigma_E,          sigma_E = sqrt(mean_m (E_m - <E>)^2)
    F_b = <F> - (kappa / sigma_E) * mean_m[(E_m - <E>) (F_m - <F>)]

(the second term is +kappa * d sigma_E / dx, exact for any committee). The dynamics drifts
towards configurations the members disagree on. kappa is per structure (sigma_E grows with the
number of atoms); start around 0.5 and check how many frames end up 'failed'.
"""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class BiasedCommitteeCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, calcs: list, kappa: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        if len(calcs) < 2:
            raise ValueError("uncertainty-biased MD needs at least two committee members")
        self.calcs = list(calcs)
        self.kappa = float(kappa)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        E, F = [], []
        for c in self.calcs:
            work = self.atoms.copy()
            work.calc = c
            E.append(work.get_potential_energy())
            F.append(work.get_forces())
        E, F = np.asarray(E), np.asarray(F)
        dE, dF = E - E.mean(), F - F.mean(0)
        sigma = float(np.sqrt((dE ** 2).mean()))
        energy = float(E.mean() - self.kappa * sigma)
        forces = F.mean(0)
        if sigma > 1e-10:
            forces = forces - self.kappa / sigma * np.einsum("m,mij->ij", dE, dF) / len(E)
        self.results = {"energy": energy, "free_energy": energy, "forces": forces,
                        "sigma_E": sigma, "energy_unbiased": float(E.mean())}
