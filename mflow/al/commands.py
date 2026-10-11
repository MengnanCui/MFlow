# -*- coding: utf-8 -*-
"""
mflow al <step> — one command per step, `mflow al run` chains them up to the next human gate.

Round r (r = 0 is the seed round, explored with the foundation model):
    explore   ->  select  ->  export      ...TFlow labels...      import  ->  train (round r+1)
State lives in state.json (round, stage, committee, dataset); every step appends to ledger.jsonl.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import ase.io

from .. import __version__
from ..log import LOG, num, table
from ..report import Report
from .campaign import Campaign
from .convergence import current_region, evaluate, l1_passed, regions, stop_criteria  # noqa: F401

ORDER = ["init", "explored", "selected", "exported", "labelled", "trained"]


def load(args) -> Campaign:
    camp = Campaign(args.name)
    if not (camp.root / "campaign.yaml").exists():
        raise SystemExit(f"no campaign at {camp.root} — run `mflow al init` first")
    return camp


def current_round(camp: Campaign, args) -> int:
    return int(args.round) if getattr(args, "round", None) is not None else int(camp.state.get("round", 0))


# ============================================================================ #
#  steps
# ============================================================================ #

def step_explore(camp, r, lmp=None):
    from .explore import run_explorers
    LOG.rule(f"round {r} · explore (model: {Path(camp.explorer_model()).name})")
    frames, counts = run_explorers(camp, r, lmp)
    camp.log("explore", r, model=camp.explorer_model(), counts=counts, n=len(frames), region=counts.get("region"),
             settings=camp.settings["explore"])
    camp.save_state(round=r, stage="explored")
    return frames


def step_select(camp, r, frames=None):
    from .select import run_selection
    LOG.rule(f"round {r} · select")
    if frames is None:
        frames = ase.io.read(camp.round_dir(r, "explore") / "explored.xyz", ":")
    selected, stats = run_selection(camp, r, frames)
    camp.log("select", r, **stats, settings=camp.settings["select"])
    camp.save_state(round=r, stage="selected")
    return selected, stats


def step_export(camp, r, selected=None):
    from .tflow import export_round
    LOG.rule(f"round {r} · export to TFlow")
    if selected is None:
        path = camp.round_dir(r, "select") / "selected.xyz"
        selected = ase.io.read(path, ":") if path.exists() and path.stat().st_size else []
    if not selected:
        LOG(f"[warn]round {r}: nothing selected — nothing to export[/warn]")
        return None
    out = export_round(camp, r, selected)
    camp.log("export", r, n=len(selected), dir=str(out))
    camp.save_state(round=r, stage="exported")
    return out


def step_import(camp, r, tflow_dir=None, labelled=None) -> Report:
    from ..train.check import check_frames
    from .tflow import import_labelled_xyz, import_tflow, show_rejects

    LOG.rule(f"round {r} · import labels")
    rep = Report("al-import", round=r)
    cands = ase.io.read(camp.round_dir(r, "export") / "candidates.extxyz", ":")
    wanted = {str(a.info["mflow_id"]): a for a in cands}
    if tflow_dir:
        accepted, rejects = import_tflow(Path(tflow_dir).resolve(), wanted)
    else:
        accepted, rejects = import_labelled_xyz(Path(labelled), wanted)

    # the same rules as `mflow check`: broken labels never enter the dataset
    if accepted:
        sub = Report("check")
        flags = check_frames(accepted, "REF_", sub, None, has_test=True)
        bad = set(flags["nan"]) | set(flags["f_fail"]) | set(flags["close"])
        for i in sorted(bad):
            rejects.append({"mflow_id": accepted[i].info["mflow_id"], "dir": None,
                            "reason": "failed mflow check (NaN, |F| > fail limit, or atoms too close)"})
        accepted = [a for i, a in enumerate(accepted) if i not in bad]
    show_rejects(rejects)
    rep.check("labels", "pass" if accepted else "fail", f"{len(accepted)}/{len(wanted)} accepted · {len(rejects)} rejected",
              suggest="nothing usable came back from DFT: check the TFlow directory / report.jsonl")
    if len(rejects) > 0.2 * max(len(wanted), 1):
        rep.check("reject rate", "warn", f"{len(rejects) / len(wanted):.0%} rejected",
                  suggest="many labels rejected: look at the reasons (SCF? structure changed?) before training")
    (camp.round_dir(r, "export") / "rejects.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in rejects), encoding="utf-8")
    if not accepted:
        rep.show()
        return rep

    # fresh-frame error: the model that explored had never seen these frames — the honest test
    spot = fresh_frame_error(camp, accepted)
    rep.data["fresh_frame_error"] = spot
    if spot:
        LOG(f"fresh frames vs {spot['model']}: E RMSE [hi]{spot['energy_RMSE']:.2f}[/hi] meV/atom · "
            f"F RMSE [hi]{spot['force_RMSE']:.1f}[/hi] meV/Å (N={spot['n']})")

    camp.labelled(r).parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(camp.labelled(r), accepted, format="extxyz")
    path, n = camp.rebuild_dataset(r + 1)
    LOG(f"dataset v{r + 1:02d}: [key]{path}[/key] · {n} structures (initial + {r + 1} labelled round(s))")
    camp.log("import", r, accepted=len(accepted), rejected=len(rejects), dataset=str(path), n_dataset=n,
             fresh_frame_error=spot, reject_reasons=sorted({x["reason"] for x in rejects}))
    camp.save_state(round=r + 1, stage="labelled", dataset=str(path))
    rep.show()
    return rep


def fresh_frame_error(camp, frames: list) -> dict | None:
    from ..calc import compare, evaluate_committee, run_mace
    from ..model import pick_device
    models = camp.committee()
    work = [a.copy() for a in frames]
    try:
        if models:
            evaluate_committee(work, models, pick_device("auto"), "float64", "ff_", None)
            e0 = "none"
        else:
            run_mace(work, spec=camp.settings["foundation"], batch_size=16, device=pick_device("auto"),
                     dtype="float64", prefix="ff_", head=None)
            e0 = "fit"           # a foundation model sits on another DFT reference
        errors, _ = compare(work, "ff_", "REF_", e0)
    except Exception as exc:
        LOG(f"[warn]fresh-frame error skipped: {type(exc).__name__}: {exc}[/warn]")
        return None
    return {"model": "committee" if models else camp.settings["foundation"], "e0": e0, "n": len(work),
            "energy_RMSE": errors["energy"]["RMSE"], "force_RMSE": errors.get("forces", {}).get("RMSE", float("nan"))}


def step_train(camp, r, lmp=None) -> int:
    from ..train.run import train_committee, train_one
    LOG.rule(f"round {r} · train")
    s, state = camp.settings, camp.state
    t = s["train"]
    test = camp.data / "test.xyz"
    out = camp.round_dir(r, "train")
    n = max(1, int(t["committee"]))
    args = Namespace(input=state["dataset"], index=":", name=f"al_r{r:02d}", outdir=str(out), log=None,
                     test=str(test) if test.exists() else None, valid=None, ref_prefix="REF_", group=None,
                     mode=t["mode"], foundation=s["foundation"], head=None, pt=None, epochs=t["epochs"],
                     batch=t["batch"], dtype=t["dtype"], device=t["device"], cueq="auto", seed=123 + 10 * r,
                     set=list(t.get("set") or []), dry=False, resume=False, force=False, no_plot=True,
                     committee=n, bagging=t.get("bagging"), foundations=t.get("foundations"),
                     member_set=t.get("member_set"))
    if n == 1:
        LOG("[dim]committee: 1 — a single model; selection uses select.uncertainty (descriptor distance by "
            "default) instead of a committee spread[/dim]")
        code = train_one(args)
        model = out / f"{args.name}.model"
        models = [str(model)] if model.exists() else []
        report = out / f"{args.name}_train_report.json"
    else:
        code = train_committee(args)
        cj = out / "committee.json"
        models = json.loads(cj.read_text())["models"] if cj.exists() else []
        report = out / f"{args.name}_committee_report.json"
    data = json.loads(report.read_text()) if report.exists() else {}
    test_data = data.get("test") or (data.get("data") or {}).get("test") or {}
    cal = test_data.get("calibration")
    f_rmse = ((test_data.get("errors") or {}).get("forces") or {}).get("RMSE")
    camp.log("train", r, models=models, exit=code, dataset=state["dataset"],
             spearman=cal["spearman_atom"] if cal else None,
             sigma_ratio=cal["sigma_ratio"] if cal else None, collapsed=cal.get("collapsed") if cal else None,
             test_force_RMSE=f_rmse, settings=t)
    if cal and cal.get("collapsed"):
        LOG(f"[warn]committee collapsed (RMS error / RMS σ = {cal['sigma_ratio']:.1f}): its spread is much smaller "
            f"than the real error. Selection still works after recalibration if Spearman is good; otherwise use "
            f"train.foundations (heterogeneous members) or select.uncertainty: descriptor[/warn]")
    elif cal and cal.get("verdict") == "fail":
        LOG(f"[warn]{cal['verdict_message']}: selection by committee deviation is unreliable this round — "
            f"more members, larger bagging differences, or more epochs[/warn]")
    if len(models) < n:
        return 2                              # a member did not train: that is a real failure
    camp.save_state(round=r, stage="trained", models=models)
    return 0


# ============================================================================ #
#  status / stopping
# ============================================================================ #

def cmd_al_status(args) -> int:
    camp = load(args)
    st = camp.state
    LOG.rule(f"MFlow {__version__} · al status · {camp.root.name}")
    LOG.kv({"round": st.get("round"), "stage": st.get("stage"), "dataset": st.get("dataset") or "—",
            "committee": f"{len(camp.committee())} model(s)", "explorer": Path(str(camp.explorer_model())).name})
    led = camp.ledger()
    rounds = sorted({x["round"] for x in led})
    rows = []
    for r in rounds:
        get = lambda step: next((x for x in reversed(led) if x["round"] == r and x["step"] == step), {})
        ex, se, im, tr = get("explore"), get("select"), get("import"), get("train")
        ff = im.get("fresh_frame_error") or {}
        rows.append([r, ex.get("n", "—"),
                     f"{se['candidate_fraction']:.1%}" if se.get("candidate_fraction") is not None else "—",
                     se.get("n_selected", "—"), im.get("accepted", "—"), im.get("rejected", "—"),
                     num(ff["energy_RMSE"]) if ff else "—", num(ff["force_RMSE"]) if ff else "—",
                     f"{tr['spearman']:.2f}" if tr.get("spearman") is not None else "—"])
    LOG(table("ledger", ["round", "explored", "cand. frac", "selected", "labelled", "rejected",
                         "fresh E", "fresh F", "Spearman"], rows, styles=["key"]))
    LOG("[dim]fresh E/F: error (meV/atom, meV/Å) of the exploring model on the frames DFT labelled that round — "
        "it never saw them, so this is the honest accuracy estimate[/dim]")
    ev = evaluate(camp)
    mark = {True: "[ok]yes[/ok]", False: "[warn]no[/warn]", None: "[dim]n/a[/dim]"}
    LOG(table("convergence", ["level", "criterion", "met", "value", "source"],
              [(lvl, n, mark[ok], v, src) for lvl, n, ok, v, src in ev["rows"]], styles=["key", "key"]))
    LOG("[dim]source: lit = published rule, sug = MFlow default (mflow/thresholds.yaml al), v0.2 = earlier rule, "
        "user = your budget. n/a = no data yet: neither passes nor blocks.[/dim]")
    if ev["verdict"] == "converged":
        LOG("[ok]converged[/ok] — run production MD and check the md-report gates before calling it done")
    elif ev["verdict"] == "budget_exhausted":
        LOG("[err]budget exhausted — NOT converged[/err]: the rows marked no are what is still missing")
    else:
        LOG("[hi]continue[/hi]: next step is " + next_step(st))
    rep = Report("al-status", round=st.get("round"), stage=st.get("stage"), verdict=ev["verdict"],
                 region=ev["region"], regions=ev["regions"])
    for lvl, n, ok, v, src in ev["rows"]:
        rep.check(f"{lvl} {n}", "warn" if ok is False else "pass", v if ok is not None else f"n/a: {v}",
                  value=ok)
    rep.data["ledger_rows"] = rows
    rep.write(camp.root / "status.json")
    return 0


def next_step(state: dict) -> str:
    stage = state.get("stage", "init")
    return {"init": "explore (round 0)", "explored": "select", "selected": "export",
            "exported": "DFT in TFlow, then `mflow al import`", "labelled": "train",
            "trained": "explore"}.get(stage, "?")


# ============================================================================ #
#  commands
# ============================================================================ #

def cmd_al_explore(args) -> int:
    camp = load(args)
    step_explore(camp, current_round(camp, args), args.lmp)
    return 0


def cmd_al_select(args) -> int:
    camp = load(args)
    step_select(camp, current_round(camp, args))
    return 0


def cmd_al_export(args) -> int:
    camp = load(args)
    step_export(camp, current_round(camp, args))
    return 0


def cmd_al_import(args) -> int:
    camp = load(args)
    if not (args.tflow or args.labelled):
        LOG("[err]pass -tflow <calc_dir> or -in <labelled.xyz>[/err]")
        return 2
    r = int(args.round) if args.round is not None else int(camp.state.get("round", 0))
    rep = step_import(camp, r, args.tflow, args.labelled)
    rep.write(camp.round_dir(r, "export") / "import_report.json")
    return 0 if not rep.failed else 2


def cmd_al_probe(args) -> int:
    from .probes import step_probe
    camp = load(args)
    out = step_probe(camp, current_round(camp, args))
    return 0 if out else 2


def cmd_al_train(args) -> int:
    camp = load(args)
    return step_train(camp, current_round(camp, args))


def cmd_al_run(args) -> int:
    """Advance through the steps that need no human, stop at the DFT hand-off (or when done)."""
    camp = load(args)
    LOG.rule(f"MFlow {__version__} · al run · {camp.root.name}")
    for _ in range(8):                                  # at most one full cycle per call
        st = camp.state
        stage, r = st.get("stage", "init"), int(st.get("round", 0))
        if stage in ("init", "trained", "labelled") and not args.force:
            verdict = evaluate(camp)["verdict"]
            if verdict == "budget_exhausted":
                LOG("[err]budget exhausted (stop.max_rounds / stop.max_dft) — NOT converged[/err]: see "
                    "`mflow al status` for what is missing; raise the budget in campaign.yaml or -force")
                return 0
        if stage in ("init", "trained"):
            step_explore(camp, r, args.lmp)
        elif stage == "explored":
            _, stats = step_select(camp, r)
            if not stats.get("n_selected"):
                done, _ = stop_criteria(camp)
                LOG("[warn]no structure needs DFT this round.[/warn] "
                    + ("The stopping criteria are met — see `mflow al status`." if done else
                       "The stopping criteria are NOT met (see `mflow al status`): explore harder — more steps, "
                       "higher explore.md.temps or widen, minhop — or lower select.lo, then `mflow al explore` again."))
                camp.save_state(stage="trained")
                return 0
        elif stage == "selected":
            step_export(camp, r)
        elif stage == "exported":
            LOG(f"[hi]waiting for DFT[/hi]: label round {r} in TFlow, then "
                f"`mflow al import -name {args.name} -tflow <calc_dir>`")
            return 0
        elif stage == "labelled":
            done, _ = stop_criteria(camp)
            if done and not args.force:
                LOG("[ok]stopping criteria met[/ok] — see `mflow al status`; -force to keep going")
                return 0
            if step_train(camp, r) != 0:
                LOG("[err]training failed — fix it and rerun `mflow al train`[/err]")
                return 2
            if camp.settings.get("probes"):
                from .probes import step_probe
                step_probe(camp, r)
        else:
            LOG(f"[err]unknown stage {stage}[/err]")
            return 2
    return 0
