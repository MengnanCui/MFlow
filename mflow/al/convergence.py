# -*- coding: utf-8 -*-
"""
When to stop an active-learning campaign — three levels, all read from ledger.jsonl.

    L0  trust window      adaptive-lower (dpgen2): lo follows the explored deviations until it is
                          stable for `consecutive` rounds, then it is frozen (state trust_frozen_lo)
    L1  per round/region  accurate >= accurate_min and failed <= failed_max for `consecutive` rounds;
                          FPS leaves < max(selected_left_min, frac × explored) frames;
                          first-seen errors (absolute targets, and <= fresh_vs_test × test RMSE);
                          candidate fraction (v0.2 rule); every exploration MD finished
    L2  campaign          property probes stable between rounds and across members; coverage
                          (descriptor OOD / QUESTS dH > 0 fraction <= ood_max)
    budget                stop.max_rounds / stop.max_dft — reached before convergence means
                          "budget exhausted, NOT converged", never "converged"

A criterion without data yet is n/a (met = None): it neither passes nor blocks, except the v0.2
core rules (candidate fraction, first-seen E and F), which must have data. Values marked [lit]
come from the literature, [sug] are MFlow's suggested defaults (mflow/thresholds.yaml, section al).
"""

from __future__ import annotations

import numpy as np

CORE = {"candidate fraction", "first-seen energy", "first-seen forces"}


def _th() -> dict:
    from ..thresholds import section
    return section("al")


def _last(records: list, n: int) -> list:
    return records[-n:] if len(records) >= n else []


def selects(led: list, region: str | None = None) -> list:
    """select records that classified frames (round 0 has no uncertainty and does not count)."""
    out = [x for x in led if x["step"] == "select" and x.get("n_accurate") is not None]
    if region is not None:
        out = [x for x in out if x.get("region", "all") == region]
    return out


# ============================================================================ #
#  L0 — adaptive-lower trust window
# ============================================================================ #

def adaptive_lo(score: np.ndarray, hi: float, numb: int, rate: float) -> float:
    """dpgen2 adaptive-lower: lo such that max(numb, rate × N) frames below hi become candidates."""
    score = np.asarray(score, float)
    below = np.sort(score[score < hi])[::-1]
    if not len(below):
        return float(hi)
    k = max(int(numb), int(rate * len(score)), 1)
    lo = below[k - 1] if len(below) > k else below[-1]
    return float(min(max(lo, 1e-6), hi))


def lo_is_stable(history: list[float], tol: float, n: int) -> bool:
    """The last n changes of lo are all below tol (needs n + 1 values)."""
    if len(history) < n + 1:
        return False
    tail = np.asarray(history[-(n + 1):], float)
    return bool(np.all(np.abs(np.diff(tail)) < tol))


# ============================================================================ #
#  L1 — per round (and per region)
# ============================================================================ #

def l1_rows(camp, led: list, region: str | None = None) -> list:
    th, st = _th(), camp.settings["stop"]
    n = int(th["consecutive"])
    rows = []
    sel = selects(led, region)
    adaptive_open = (camp.settings["select"].get("trust") == "adaptive-lower"
                     and camp.state.get("trust_frozen_lo") is None)

    # accurate / failed ratios, `consecutive` rounds in a row (an adaptive window that is still moving
    # makes the ratios a matter of construction, so those rounds do not count)
    tail = _last([x for x in sel if x.get("trust_mode") != "adaptive-lower"], n)
    if adaptive_open:
        rows.append(("L1", "accurate ratio", False, "adaptive-lower window not frozen yet", "lit"))
    elif not tail:
        rows.append(("L1", "accurate ratio", None, f"needs {n} classified round(s), have {len(sel)}", "lit"))
    else:
        acc = [x["n_accurate"] / max(x["n_explored"], 1) for x in tail]
        rows.append(("L1", "accurate ratio", all(a >= th["accurate_min"] for a in acc),
                     " → ".join(f"{a:.1%}" for a in acc) + f"  (≥ {th['accurate_min']:.0%}, {n} rounds)", "lit"))
    if tail:
        fail = [x["n_failed"] / max(x["n_explored"], 1) for x in tail]
        rows.append(("L1", "failed ratio", all(f <= th["failed_max"] for f in fail),
                     " → ".join(f"{f:.2%}" for f in fail) + f"  (≤ {th['failed_max']:.1%})", "sug"))
        last = tail[-1]
        left_max = max(th["selected_left_min"], th["selected_left_frac"] * last["n_explored"])
        rows.append(("L1", "frames left after FPS", last.get("n_selected", 0) < left_max,
                     f"{last.get('n_selected', 0)} (< {left_max:.0f})", "sug"))

    # v0.2 core rule
    frac = sel[-1]["candidate_fraction"] if sel else None
    rows.append(("L1", "candidate fraction", frac is not None and frac < st["candidate_fraction"],
                 f"{frac:.1%} (< {st['candidate_fraction']:.0%})" if frac is not None else "no classified round yet",
                 "v0.2"))
    rows += fresh_rows(camp, led, th, st)
    rows.append(md_row(led, th, region))
    return rows


def fresh_rows(camp, led: list, th: dict, st: dict) -> list:
    imp = [x for x in led if x["step"] == "import" and x.get("fresh_frame_error")]
    ff = imp[-1]["fresh_frame_error"] if imp else None
    trained = ff is not None and ff["model"] == "committee"           # not the foundation model
    rows = [("L1", "first-seen energy", trained and ff["energy_RMSE"] < st["dft_energy_meV_atom"],
             f"{ff['energy_RMSE']:.1f} meV/atom (< {st['dft_energy_meV_atom']})" if ff else "no labels imported yet",
             "v0.2"),
            ("L1", "first-seen forces", trained and ff["force_RMSE"] < st["dft_force_meV_A"],
             f"{ff['force_RMSE']:.0f} meV/Å (< {st['dft_force_meV_A']})" if ff else "no labels imported yet", "v0.2")]
    test = None
    if trained:
        tr = [x for x in led if x["step"] == "train" and x["round"] == imp[-1]["round"]
              and x.get("test_force_RMSE") is not None]
        test = tr[-1]["test_force_RMSE"] if tr else None
    if test:
        ratio = ff["force_RMSE"] / test
        rows.append(("L1", "first-seen / test forces", ratio <= th["fresh_vs_test"],
                     f"{ratio:.2f} (≤ {th['fresh_vs_test']}; test {test:.0f} meV/Å)", "sug"))
    else:
        rows.append(("L1", "first-seen / test forces", None, "no test RMSE of the exploring model", "sug"))
    return rows


def md_row(led: list, th: dict, region: str | None = None) -> tuple:
    ex = [x for x in led if x["step"] == "explore" and (x.get("counts") or {}).get("md_runs")]
    if region is not None:
        ex = [x for x in ex if x.get("region", "all") == region]
    if not ex:
        return ("L1", "exploration MD finished", None, "no MD exploration yet", "sug")
    c = ex[-1]["counts"]
    frac = c.get("md_failed", 0) / c["md_runs"]
    return ("L1", "exploration MD finished", frac <= th["md_failed_max"],
            f"{c['md_runs'] - c.get('md_failed', 0)}/{c['md_runs']} runs complete", "sug")


def l1_passed(camp, led: list, region: str | None = None) -> bool:
    rows = l1_rows(camp, led, region)
    return all(ok is not False and (ok is True or name not in CORE) for _, name, ok, _, _ in rows)


# ============================================================================ #
#  L2 — campaign
# ============================================================================ #

def l2_rows(camp, led: list) -> list:
    th = _th()
    n = int(th["consecutive"])
    rows = []
    probes = [x for x in led if x["step"] == "probe"]
    if not camp.settings.get("probes"):
        rows.append(("L2", "properties stable between rounds", None, "no probes in campaign.yaml", "sug"))
    else:
        tail = _last([x for x in probes if x.get("stable") is not None], n)
        rows.append(("L2", "properties stable between rounds", all(x["stable"] for x in tail) if tail else None,
                     (" → ".join("yes" if x["stable"] else "no" for x in tail) + f"  ({n} rounds)") if tail
                     else f"needs {n + 1} probed rounds", "sug"))
        spread = [x for x in probes if x.get("spread_ok") is not None]
        rows.append(("L2", "properties agree across members", spread[-1]["spread_ok"] if spread else None,
                     spread[-1].get("spread_summary", "") if spread else "single model or not probed", "sug"))
    sel = selects(led) or [x for x in led if x["step"] == "select"]
    cov = next((x for x in reversed(sel) if x.get("ood_fraction") is not None or x.get("dH_pos_fraction") is not None),
               None)
    if cov is None:
        rows.append(("L2", "coverage (out-of-distribution)", None, "not measured (select.coverage)", "sug"))
    else:
        val = cov.get("dH_pos_fraction") if cov.get("dH_pos_fraction") is not None else cov["ood_fraction"]
        what = "QUESTS dH > 0" if cov.get("dH_pos_fraction") is not None else "descriptor OOD"
        rows.append(("L2", "coverage (out-of-distribution)", val <= th["ood_max"],
                     f"{what}: {val:.1%} of explored frames (≤ {th['ood_max']:.0%})", "sug"))
    return rows


# ============================================================================ #
#  regions and budget
# ============================================================================ #

def regions(camp) -> list[dict]:
    md = camp.settings["explore"]["md"]
    regs = md.get("regions") or [{"name": "all"}]
    out = []
    for i, r in enumerate(regs):
        r = dict(r)
        r.setdefault("name", f"region{i}")
        out.append(r)
    return out


def current_region(camp) -> dict:
    regs = regions(camp)
    return regs[min(int(camp.state.get("region_index", 0)), len(regs) - 1)]


def budget(camp, led: list) -> tuple[bool, list]:
    st = camp.settings["stop"]
    rows, hit = [], False
    rounds = max([x["round"] for x in led if x["step"] == "select"], default=-1) + 1
    dft = sum(int(x.get("accepted") or 0) for x in led if x["step"] == "import")
    for key, used, unit in (("max_rounds", rounds, "rounds"), ("max_dft", dft, "DFT labels")):
        cap = st.get(key)
        if cap:
            over = used >= int(cap)
            hit |= over
            rows.append(("budget", key, not over, f"{used}/{cap} {unit}", "user"))
    return hit, rows


# ============================================================================ #
#  verdict
# ============================================================================ #

def evaluate(camp) -> dict:
    """{'verdict': converged | continue | budget_exhausted, 'rows': [...], 'region': name, 'regions': {...}}"""
    led = camp.ledger()
    regs = regions(camp)
    reg = current_region(camp)
    per_region = {r["name"]: l1_passed(camp, led, r["name"] if len(regs) > 1 else None) for r in regs}
    last_region = int(camp.state.get("region_index", 0)) >= len(regs) - 1
    rows = l1_rows(camp, led, reg["name"] if len(regs) > 1 else None)
    if len(regs) > 1:
        rows.insert(0, ("L1", "regions passed", last_region and all(per_region.values()),
                        ", ".join(f"{k} {'✓' if v else '·'}" for k, v in per_region.items())
                        + f"  (exploring {reg['name']})", "lit"))
    rows += l2_rows(camp, led)
    hit, brows = budget(camp, led)
    rows += brows
    l1 = [r for r in rows if r[0] == "L1"]
    l1_ok = all(ok is not False and (ok is True or name not in CORE) for _, name, ok, _, _ in l1)
    l2_ok = all(ok is not False for lvl, _, ok, _, _ in rows if lvl == "L2")
    converged = l1_ok and l2_ok and (len(regs) == 1 or (last_region and all(per_region.values())))
    verdict = "converged" if converged else ("budget_exhausted" if hit else "continue")
    return {"verdict": verdict, "rows": rows, "region": reg["name"], "regions": per_region}


def stop_criteria(camp) -> tuple[bool, list]:
    """v0.2 interface: (converged?, [(name, met, value)])."""
    ev = evaluate(camp)
    return ev["verdict"] == "converged", [(f"{lvl} {name}", ok, val) for lvl, name, ok, val, _ in ev["rows"]]
