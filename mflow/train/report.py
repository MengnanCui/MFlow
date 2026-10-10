# -*- coding: utf-8 -*-
"""
mflow train-report — what did the run learn, and is the model good enough?

Reads what mace_run_train leaves behind (config.yaml, results/<name>_run-<seed>_train.txt,
the final .model) and evaluates the test set with MFlow's own calc/compare, so the numbers
use exactly the same definitions as `mflow calc`.

The results file is one JSON object per line:
    {"mode": "eval", "epoch": 3, "head": "Default", "loss": ..., "rmse_e_per_atom": ..., "rmse_f": ...}
    {"mode": "opt",  "epoch": 3, "loss": ...}                     (one per optimiser step)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .. import __version__
from ..data import group_of, read_dataset
from ..log import LOG, num, table
from ..report import Report
from ..style import COLORS, LINESTYLES, MARKERS, plt
from ..thresholds import section

_T = section("train")    # mflow/thresholds.yaml
GRADES = [(name, _T[f"grade_{name}"]["energy_meV_atom"], _T[f"grade_{name}"]["force_meV_A"])
          for name in ("excellent", "good")]                 # E meV/atom, F meV/Å
OVERFIT_RATIO = _T["overfit_ratio"]                         # final valid loss vs best valid loss
PT_HEAD = "pt_head"      # the replay head of multihead fine-tuning


def load_config(run_dir: Path) -> dict:
    import yaml
    path = run_dir / "config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def results_file(run_dir: Path, cfg: dict) -> Path | None:
    name, seed = cfg.get("name"), cfg.get("seed")
    exact = run_dir / "results" / f"{name}_run-{seed}_train.txt"
    if exact.exists():
        return exact
    found = sorted((run_dir / "results").glob("*_train.txt"))
    return found[-1] if found else None


def parse_results(path: Path) -> dict:
    """-> {"eval": {head: [rows sorted by epoch]}, "opt": {epoch: mean loss}}; tolerant to junk lines."""
    evals, opt = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("mode") == "eval":
            evals.setdefault(row.get("head", "Default"), []).append(row)
        elif row.get("mode") == "opt" and row.get("epoch") is not None:
            opt.setdefault(int(row["epoch"]), []).append(float(row.get("loss", math.nan)))
    for rows in evals.values():
        rows.sort(key=lambda r: -1 if r.get("epoch") is None else r["epoch"])
    return {"eval": evals, "opt": {e: float(np.mean(v)) for e, v in sorted(opt.items())}}


def main_head(evals: dict) -> str | None:
    heads = [h for h in evals if h != PT_HEAD]
    return heads[0] if heads else (next(iter(evals), None))


def curve(rows: list, key: str, scale: float = 1.0):
    epochs = np.array([-1 if r.get("epoch") is None else r["epoch"] for r in rows], float)
    values = np.array([float(r.get(key, math.nan)) * scale for r in rows])
    return epochs, values


def analyse_curves(parsed: dict, rep: Report) -> dict:
    """Divergence, over-fitting and how far training moved the validation error."""
    head = main_head(parsed["eval"])
    if head is None:
        rep.check("training curve", "fail", "no evaluation rows in the results file",
                  suggest="the run stopped before the first evaluation: check mace_train.out")
        return {}
    rows = parsed["eval"][head]
    ep, loss = curve(rows, "loss")
    _, rmse_e = curve(rows, "rmse_e_per_atom", 1000)
    _, rmse_f = curve(rows, "rmse_f", 1000)
    finite = np.isfinite(loss)
    if not finite.all():
        rep.check("loss finite", "fail", f"NaN/inf validation loss at epoch {int(ep[~finite][0])}",
                  suggest="training diverged: lower lr, check the data with `mflow check`, use float64")
    else:
        rep.check("loss finite", "pass", "no NaN / inf")

    trained = ep >= 0
    if not trained.any():
        rep.check("epochs", "fail", "no completed epoch")
        return {"head": head}
    best = int(np.nanargmin(np.where(trained, loss, np.inf)))
    last = len(rows) - 1
    ratio = loss[last] / loss[best] if loss[best] > 0 else 1.0
    rep.check("over-fitting", "warn" if ratio > OVERFIT_RATIO else "pass",
              f"final valid loss {ratio:.2f}× the best (epoch {int(ep[best])})",
              suggest="validation loss rose after its minimum: fewer epochs, lower lr, or more data")
    initial = rows[0] if rows[0].get("epoch") is None else None
    gain = {}
    if initial is not None:
        gain = {"E": rmse_e[0] / rmse_e[last] if rmse_e[last] else math.nan,
                "F": rmse_f[0] / rmse_f[last] if rmse_f[last] else math.nan}
        status = "warn" if gain["F"] < 1.05 else "pass"
        rep.check("improvement", status, f"valid RMSE F {rmse_f[0]:.0f} → {rmse_f[last]:.0f} meV/Å "
                  f"(×{gain['F']:.1f}) · E {rmse_e[0]:.1f} → {rmse_e[last]:.1f} meV/atom",
                  suggest="training barely moved the validation error: lr too low, too few epochs, "
                          "or the foundation model was already as good as the data allows")
    opt = parsed["opt"]
    if opt and np.isfinite(loss[last]):
        train_last = opt.get(int(ep[last]), math.nan)
        rep.data["train_loss_last"] = train_last
    return {"head": head, "best_epoch": int(ep[best]), "last_epoch": int(ep[last]),
            "valid": {"rmse_e_meV_atom": float(rmse_e[last]), "rmse_f_meV_A": float(rmse_f[last]),
                      "loss": float(loss[last])},
            "best": {"rmse_e_meV_atom": float(rmse_e[best]), "rmse_f_meV_A": float(rmse_f[best]),
                     "loss": float(loss[best])},
            "gain": gain}


def plot_curves(parsed: dict, head: str, outfile: str, best_epoch: int | None = None):
    rows = parsed["eval"][head]
    ep, loss = curve(rows, "loss")
    _, rmse_e = curve(rows, "rmse_e_per_atom", 1000)
    _, rmse_f = curve(rows, "rmse_f", 1000)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    opt = parsed["opt"]
    if opt:
        axes[0].plot(list(opt), list(opt.values()), color=COLORS[1], ls=LINESTYLES[1], label="train (mean/epoch)")
    axes[0].plot(ep, loss, color=COLORS[0], marker=MARKERS[0], markersize=6, markerfacecolor="none",
                 label=f"valid ({head})")
    axes[0].set(xlabel="Epoch", ylabel="Loss", yscale="log")
    axes[0].legend()
    for ax, y, label, color in ((axes[1], rmse_e, "Valid energy RMSE (meV/atom)", COLORS[0]),
                                (axes[2], rmse_f, r"Valid force RMSE (meV/$\mathrm{\AA}$)", COLORS[1])):
        ax.plot(ep, y, color=color, marker=MARKERS[0], markersize=6, markerfacecolor="none")
        ax.set(xlabel="Epoch", ylabel=label, yscale="log")
    for ax in axes:
        if best_epoch is not None:
            ax.axvline(best_epoch, color="black", ls=LINESTYLES[2], lw=1.5)
        ax.tick_params(which="both", direction="in")
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    LOG(f"figure [key]{outfile}[/key]")


def grade(e_rmse: float, f_rmse: float) -> str:
    for name, e_max, f_max in GRADES:
        if e_rmse < e_max and f_rmse < f_max:
            return name
    return "needs work"


def evaluate_test(model: Path, test_file: str, head: str | None, group_key: str | None, device: str,
                  dtype: str, rep: Report, outdir: Path, stem: str, plot: bool = True) -> dict | None:
    """The test set through MFlow's calc path: same error definitions as `mflow calc`."""
    from ..calc import compare, make_plot, report_errors, report_groups, report_worst, run_mace
    from ..data import find_ref
    import ase.io

    frames = read_dataset(test_file)
    ref = find_ref(frames, exclude="mlp_")
    if ref is None:
        rep.check("test set", "fail", f"no reference labels in {test_file}")
        return None
    frames = run_mace(frames, spec=str(model), batch_size=16, device=device, dtype=dtype, prefix="mlp_",
                      head=head)
    errors, series = compare(frames, "mlp_", ref, "none")   # same DFT reference, E0s live in the model
    report_errors(errors, "mlp_", ref)
    groups = group_of(frames, group_key) if group_key else None
    by_group = report_groups(groups, series, group_key) if groups is not None else None
    report_worst(frames, series, groups)
    ase.io.write(outdir / f"{stem}_test_mace.xyz", frames, format="extxyz")
    if plot:
        make_plot(frames, "mlp_", ref, errors, series, str(outdir / f"{stem}_test_summary.png"), groups, by_group)

    e_rmse = errors["energy"]["RMSE"]
    f_rmse = errors.get("forces", {}).get("RMSE", math.nan)
    g = grade(e_rmse, f_rmse)
    rep.check("test accuracy", "pass" if g != "needs work" else "warn",
              f"{g}: E {e_rmse:.2f} meV/atom · F {f_rmse:.1f} meV/Å (test, N={len(frames)})",
              suggest="test error above the 'good' band (10 meV/atom, 50 meV/Å): add data where the "
                      "worst groups are, or train longer")
    if by_group:
        worst = max(by_group, key=lambda k: by_group[k]["E_RMSE"])
        best = min(by_group, key=lambda k: by_group[k]["E_RMSE"])
        if by_group[worst]["E_RMSE"] > 3 * max(by_group[best]["E_RMSE"], 1e-9) and len(by_group) > 1:
            rep.suggest(f"group '{worst}' has {by_group[worst]['E_RMSE']:.1f} meV/atom, "
                        f"{by_group[worst]['E_RMSE'] / max(by_group[best]['E_RMSE'], 1e-9):.0f}× the best "
                        f"group: add more {worst} configurations")
    return {"errors": errors, "by_group": by_group, "grade": g, "n": len(frames)}


def model_file(run_dir: Path, cfg: dict) -> Path | None:
    name = cfg.get("name", "")
    for cand in (run_dir / f"{name}.model", *sorted(run_dir.glob("*.model"))):
        if cand.exists() and not cand.name.endswith("_compiled.model"):
            return cand
    return None


def analyse_run(run_dir: Path, rep: Report, test_file: str | None = None, group_key: str | None = None,
                device: str = "auto", plot: bool = True) -> dict:
    """Everything `train-report` does; `train` calls it right after mace_run_train returns."""
    from ..model import pick_device
    cfg = load_config(run_dir)
    name = cfg.get("name", run_dir.name)
    meta_file = run_dir / "mflow_train.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    out = {"run_dir": str(run_dir.resolve()), "name": name, "mode": meta.get("_mflow_mode"),
           "foundation": cfg.get("foundation_model")}

    path = results_file(run_dir, cfg)
    if path is None:
        rep.check("results file", "fail", f"no results/*_train.txt in {run_dir}")
        return out
    parsed = parse_results(path)
    curves = analyse_curves(parsed, rep)
    out["curves"] = curves
    if curves.get("head") and plot and curves.get("best_epoch") is not None:
        plot_curves(parsed, curves["head"], str(run_dir / f"{name}_learning_curve.png"), curves.get("best_epoch"))
    if curves.get("valid"):
        v = curves["valid"]
        LOG(table("validation (last epoch)", ["", "value", "unit"],
                  [("energy RMSE", num(v["rmse_e_meV_atom"]), "meV/atom"),
                   ("force RMSE", num(v["rmse_f_meV_A"]), "meV/Å"),
                   ("best epoch", curves["best_epoch"], ""), ("last epoch", curves["last_epoch"], "")],
                  styles=["key", "hi", "dim"]))

    model = model_file(run_dir, cfg)
    rep.check("model file", "pass" if model else "fail", str(model) if model else "no .model written",
              suggest="no final model: the run did not finish — look at mace_train.out")
    out["model"] = str(model) if model else None
    test_file = test_file or cfg.get("test_file")
    if model and test_file:
        LOG.rule("test set")
        heads = _model_heads(model)
        head = curves.get("head") if len(heads) > 1 else None
        out["test"] = evaluate_test(model, test_file, head, group_key, pick_device(device),
                                    cfg.get("default_dtype", "float64"), rep, run_dir, name, plot)
        out["heads"] = heads
        if len(heads) > 1:
            rep.suggest(f"multi-head model ({', '.join(heads)}): export with `mflow deploy -head {head}`")
    elif model:
        rep.check("test set", "warn", "no test set given", suggest="pass -test to measure the final accuracy")
    return out


def _model_heads(model: Path) -> list:
    import torch
    try:
        net = torch.load(str(model), map_location="cpu", weights_only=False)
    except TypeError:
        net = torch.load(str(model), map_location="cpu")
    return list(getattr(net, "heads", []) or [])


def cmd_train_report(args) -> int:
    run_dir = Path(args.dir)
    LOG.rule(f"MFlow {__version__} · train-report")
    LOG.kv({"run": str(run_dir), "test": args.test or "from config", "group by": args.group or "—"})
    rep = Report("train-report")
    rep.data.update(analyse_run(run_dir, rep, args.test, args.group, args.device, not args.no_plot))
    LOG.rule("verdict")
    rep.show()
    rep.write(run_dir / f"{rep.data.get('name', run_dir.name)}_train_report.json")
    return 0 if not rep.failed else 2
