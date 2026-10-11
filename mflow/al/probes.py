# -*- coding: utf-8 -*-
"""
Property probes — the L2 "properties stable between rounds" criterion.

After every training round each model (every committee member, or the single model) computes
the probes listed in campaign.yaml, reusing `mflow validate`:

    probes:
      - {kind: eos, index: 0}                     # V0, B0 of data/system.xyz frame 0
      - {kind: dimer, elements: [Cu]}             # r_eq, well depth, wall intact, per pair
      - {kind: phonon, index: 0, supercell: [3, 3, 3]}   # highest frequency, no imaginary modes

Each value is compared with the previous probed round (member mean) and across members; the
tolerances are al.probe_tol in mflow/thresholds.yaml. A property that still moves between rounds
means the dataset still changes what the model predicts — not converged, whatever the errors say.
"""

from __future__ import annotations

import json

import ase.io
import numpy as np

from ..log import LOG, table
from ..report import Report

# metric suffix -> (tolerance key, relative?)
TOL = {"V0": ("eos_V0_rel", True), "B0": ("eos_B0_rel", True), "r_eq": ("dimer_rmin_A", False),
       "depth": ("dimer_Emin_eV", False), "max_THz": ("phonon_max_THz", False)}


def probe_values(calc, system: list, probes: list) -> dict:
    """{'eos0:V0': 11.8, 'dimer:Cu-Cu:r_eq': 2.2, 'dimer:Cu-Cu:wall': True, ...} for one model."""
    from ..validate import run_dimer, run_eos, run_phonon
    rep = Report("probe")                                     # pass/warn/fail of validate is not used here
    out = {}
    for p in probes:
        kind = p["kind"]
        idx = int(p.get("index", 0))
        if kind == "eos":
            r = run_eos(calc, system[idx].copy(), float(p.get("strain", 0.04)), int(p.get("npts", 11)),
                        bool(p.get("relax", False)), rep, {})
            out[f"eos{idx}:V0"] = r["V0_A3_atom"]
            out[f"eos{idx}:B0"] = r["B0_GPa"]
        elif kind == "dimer":
            elements = p.get("elements") or sorted(set(system[idx].get_chemical_symbols()))
            r = run_dimer(calc, list(elements), float(p.get("rmin", 0.9)), p.get("rmax"), float(p.get("dr", 0.01)),
                          rep)
            for pair, m in r["pairs"].items():
                out[f"dimer:{pair}:r_eq"] = m["r_eq"]
                out[f"dimer:{pair}:depth"] = m["well_depth"]
                out[f"dimer:{pair}:wall"] = bool(m["E_short"] > m["E_tail"] and not m["repulsion_violations"])
        elif kind == "phonon":
            r = run_phonon(calc, system[idx].copy(), list(p.get("supercell", [2, 2, 2])), int(p.get("mesh", 8)),
                           float(p.get("displacement", 0.01)), rep)
            if r:
                out[f"phonon{idx}:max_THz"] = r["max_THz"]
                out[f"phonon{idx}:stable"] = r["n_imaginary_qpoints"] == 0
        else:
            raise ValueError(f"unknown probe kind {kind!r} (eos, dimer, phonon)")
    return out


def _tol(key: str, tol: dict) -> tuple[float, bool] | None:
    name, rel = TOL.get(key.rsplit(":", 1)[-1], (None, False))
    return (float(tol[name]), rel) if name else None


def within(a, b, key: str, tol: dict) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) and bool(b)
    t = _tol(key, tol)
    if t is None:
        return True
    lim, rel = t
    return abs(a - b) <= lim * (abs(b) if rel else 1.0)


def mean_values(per_model: dict) -> dict:
    keys = next(iter(per_model.values())).keys()
    out = {}
    for k in keys:
        vals = [v[k] for v in per_model.values() if k in v]
        out[k] = all(vals) if isinstance(vals[0], bool) else float(np.mean(vals))
    return out


def compare(prev: dict, cur: dict, tol: dict) -> list:
    rows = []
    for k, v in cur.items():
        if k in prev:
            rows.append({"metric": k, "previous": prev[k], "current": v, "ok": within(v, prev[k], k, tol)})
    return rows


def spread(per_model: dict, tol: dict) -> tuple[bool | None, str]:
    if len(per_model) < 2:
        return None, "single model"
    bad = []
    for k in next(iter(per_model.values())):
        vals = [v[k] for v in per_model.values() if k in v]
        if isinstance(vals[0], bool):
            if not all(vals):
                bad.append(f"{k} false for some member")
            continue
        ref = float(np.mean(vals))
        if not all(within(x, ref, k, tol) for x in vals):
            bad.append(f"{k} {min(vals):.4g}–{max(vals):.4g}")
    return (not bad), ("; ".join(bad) if bad else f"{len(per_model)} members agree")


def step_probe(camp, r: int, calc_factory=None) -> dict:
    """Probe every model of round r; compare with the previous probed round; log to the ledger."""
    from ..thresholds import section
    from ..validate import calculator
    probes = camp.settings.get("probes") or []
    if not probes:
        LOG("[warn]no probes in campaign.yaml — nothing to do[/warn]")
        return {}
    LOG.rule(f"round {r} · property probes")
    tol = section("al")["probe_tol"]
    system = ase.io.read(camp.data / "system.xyz", ":")
    models = camp.committee() or [camp.settings["foundation"]]
    factory = calc_factory or (lambda m: calculator(m, None))
    per_model = {}
    for m in models:
        LOG(f"[dim]probing {m}[/dim]")
        per_model[str(m)] = probe_values(factory(m), system, probes)
    cur = mean_values(per_model)
    prev_rec = next((x for x in reversed(camp.ledger()) if x["step"] == "probe" and x["round"] < r), None)
    rows = compare(prev_rec["values"], cur, tol) if prev_rec else []
    stable = all(x["ok"] for x in rows) if rows else None
    spread_ok, spread_summary = spread(per_model, tol)
    LOG(table(f"round {r} · probes (member mean) vs round {prev_rec['round'] if prev_rec else '—'}",
              ["metric", "previous", "current", "within tolerance"],
              [(k, _fmt(prev_rec["values"].get(k)) if prev_rec else "—", _fmt(v),
                next((("[ok]yes[/ok]" if x["ok"] else "[warn]no[/warn]") for x in rows if x["metric"] == k), "—"))
               for k, v in cur.items()], styles=["key"]))
    if spread_ok is not None:
        LOG(("[ok]members agree[/ok]" if spread_ok else "[warn]members disagree[/warn]") + f": {spread_summary}")
    out = {"round": r, "models": per_model, "values": cur, "comparison": rows, "stable": stable,
           "spread_ok": spread_ok, "spread_summary": spread_summary, "tolerance": tol}
    path = camp.round_dir(r, "probes") / "probes.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    camp.log("probe", r, values=cur, stable=stable, spread_ok=spread_ok, spread_summary=spread_summary,
             changed=[x["metric"] for x in rows if not x["ok"]])
    return out


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return f"{v:.4g}"
