# -*- coding: utf-8 -*-
"""
Quality gates for an MD run. Each gate adds one pass/warn/fail check to a Report.

Thresholds (also in docs/conventions.html):
    NVE drift         |slope| of E_total/atom vs time   pass < 5e-4, warn < 5e-3 eV/atom/ps (LAMBench)
    temperature       production mean vs target         pass < 5 %,  warn < 15 %
    temperature trend linear change over production      pass < 5 % of the target
    volume drift      NPT, change over production        pass < 2 %,  warn < 5 %
    closest pair      any frame                          fail < 0.6 Å
Force accuracy does not predict MD stability (Fu et al., arXiv 2210.07237), so these run on
every trajectory, not just when training errors look bad.
"""

from __future__ import annotations

import numpy as np

from ..report import Report

DRIFT_PASS, DRIFT_WARN = 5e-4, 5e-3      # eV/atom/ps
T_PASS, T_WARN = 0.05, 0.15
T_CHANGE = 0.05                           # trend over the production window, fraction of target
V_PASS, V_WARN = 0.02, 0.05
D_MIN = 0.6                               # Å


def production(x: np.ndarray, skip: float) -> slice:
    return slice(int(len(x) * skip), None)


def slope(t: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(t, y, 1)[0]) if len(t) >= 3 and np.ptp(t) > 0 else 0.0


def gate_health(log: dict, rep: Report):
    rep.check("LAMMPS errors", "fail" if log["errors"] else "pass",
              log["errors"][0] if log["errors"] else "none", value=log["errors"][:5],
              suggest="LAMMPS stopped with an ERROR: read the first ERROR line in log.lammps")
    db = max(log["dangerous_builds"] or [0])
    rep.check("dangerous builds", "warn" if db else "pass", f"{db}",
              suggest="dangerous neighbour-list builds: atoms moved too far between rebuilds — "
                      "smaller timestep or larger skin")
    rep.check("lost atoms", "fail" if log["lost_atoms"] else "pass",
              "atoms lost" if log["lost_atoms"] else "none", suggest="atoms left the box: the run blew up")
    rep.check("runaway halt", "fail" if log["halted"] else "pass",
              "fix halt stopped the run (temperature runaway)" if log["halted"] else "not triggered",
              suggest="the temperature ran away: the potential is unstable here (extrapolation?) or dt too big")
    rep.check("finished", "pass" if log["finished"] else "warn",
              "completed" if log["finished"] else "log has no 'Total wall time' — still running or killed")


def gate_energy(th: dict, ensemble: str, natoms: int, rep: Report, skip: float):
    if "Time" not in th or len(th["Time"]) < 3:
        rep.check("thermo data", "fail", "fewer than 3 thermo rows")
        return {}
    t = th["Time"]
    sl = production(t, skip)
    out = {}
    if ensemble == "nve":
        drift = slope(t[sl], th["TotEng"][sl] / natoms)
        status = "pass" if abs(drift) < DRIFT_PASS else ("warn" if abs(drift) < DRIFT_WARN else "fail")
        rep.check("NVE energy drift", status, f"{drift:+.2e} eV/atom/ps (pass < {DRIFT_PASS:g})",
                  value=drift, suggest="energy drifts in NVE: lower the timestep; if it persists the "
                                       "potential is not smooth / conservative enough (fp32 export?)")
        out["nve_drift_eV_atom_ps"] = drift
        fluct = float(np.std(th["TotEng"][sl]) / natoms)
        out["nve_fluct_eV_atom"] = fluct
    return out


def gate_temperature(th: dict, target_end: float, ensemble: str, rep: Report, skip: float, ramp: bool):
    if "Temp" not in th:
        return {}
    t, T = th["Time"], th["Temp"]
    sl = production(T, skip)
    mean, std = float(np.mean(T[sl])), float(np.std(T[sl]))
    out = {"T_mean": mean, "T_std": std}
    if ensemble in ("nvt", "npt") and not ramp:
        dev = abs(mean - target_end) / max(target_end, 1e-9)
        status = "pass" if dev < T_PASS else ("warn" if dev < T_WARN else "fail")
        rep.check("temperature", status, f"{mean:.1f} ± {std:.1f} K vs {target_end:g} K ({dev * 100:.1f} %)",
                  suggest="mean temperature off target: run longer, check tdamp, or look for instability")
        span = float(t[sl][-1] - t[sl][0]) if len(t[sl]) > 1 else 0.0
        change = slope(t[sl], T[sl]) * span / max(target_end, 1e-9)
        rep.check("temperature drift", "pass" if abs(change) < T_CHANGE else "warn",
                  f"{change * 100:+.2f} % over the production window",
                  suggest="temperature still drifting: equilibrate longer")
    return out


def gate_volume(th: dict, ensemble: str, rep: Report, skip: float):
    if ensemble != "npt" or "Volume" not in th:
        return {}
    V = th["Volume"]
    sl = production(V, skip)
    vp = V[sl]
    change = float((vp[-1] - vp[0]) / vp[0]) if len(vp) > 1 else 0.0
    status = "pass" if abs(change) < V_PASS else ("warn" if abs(change) < V_WARN else "fail")
    rep.check("volume drift (NPT)", status, f"{change * 100:+.2f} % over production",
              suggest="volume still changing: equilibrate longer; large changes can mean a phase change "
                      "or an unstable potential")
    out = {"V_mean": float(np.mean(vp)), "V_std": float(np.std(vp))}
    if "Density" in th:
        out["density_mean"] = float(np.mean(th["Density"][sl]))
    return out


def gate_geometry(dmin: np.ndarray, rep: Report):
    if dmin is None or not len(dmin):
        return {}
    bad = np.where(dmin < D_MIN)[0]
    rep.check("closest pair (trajectory)", "fail" if len(bad) else "pass",
              f"min {dmin.min():.2f} Å" + (f" · {len(bad)} frames below {D_MIN} Å" if len(bad) else ""),
              suggest="atoms collapsed onto each other: the potential has a hole there — add such "
                      "configurations to training (active learning)")
    return {"dmin_min": float(dmin.min()), "dmin_first_bad_frame": int(bad[0]) if len(bad) else None}
