# -*- coding: utf-8 -*-
"""
mflow train — check the data, write config.yaml, run mace_run_train and watch it.

The run directory is self-contained and can be re-analysed or resumed later:

    <outdir>/
      config.yaml            exactly what mace_run_train received
      data/train.xyz ...     the datasets, labels renamed to REF_energy / REF_forces / REF_stress
      mace_train.out         stdout + stderr of mace_run_train
      results/ logs/ checkpoints/ <name>.model     written by MACE
      <name>_train_report.json / _learning_curve.png / _test_summary.png
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import ase.io
from rich.progress import Progress

from .. import __version__
from ..data import find_ref, keys, read_dataset
from ..log import LOG, MONITOR, elapsed, progress_columns
from ..model import pick_device, versions
from ..report import Report
from . import config as C
from .check import is_isolated_atom, run_check, stress_key
from .report import analyse_run, results_file

RISING_EVALS = 5        # consecutive validation-loss increases that trigger a warning
POLL_S = 2.0


def normalise(frames: list, ref: str) -> list:
    """Copy with the reference labels renamed to the keys the config names (REF_*)."""
    out = []
    ekey, fkey = keys(ref)
    skey = stress_key(ref)
    for a in frames:
        b = a.copy()
        b.calc = None
        b.info = {k: v for k, v in a.info.items() if k not in (ekey, skey)}
        b.arrays = {k: v for k, v in a.arrays.items() if k != fkey}
        if ekey in a.info:
            b.info["REF_energy"] = float(a.info[ekey])
        if fkey in a.arrays:
            b.arrays["REF_forces"] = a.arrays[fkey]
        if skey in a.info:
            b.info["REF_stress"] = a.info[skey]
        out.append(b)
    return out


def cueq_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("cuequivariance_torch") is not None


class Watcher:
    """Follows results/<name>_run-<seed>_train.txt while mace_run_train writes it."""

    def __init__(self, run_dir: Path, cfg: dict):
        self.run_dir, self.cfg = run_dir, cfg
        self.path = None
        self.pos = 0
        self.rows: list[dict] = []
        self.rising = 0
        self.warned_rising = self.warned_mem = False

    def poll(self) -> list[dict]:
        if self.path is None:
            self.path = results_file(self.run_dir, self.cfg)
            if self.path is None:
                return []
        with open(self.path, encoding="utf-8") as fh:
            fh.seek(self.pos)
            chunk = fh.read()
        end = chunk.rfind("\n")
        if end < 0:
            return []
        self.pos += len(chunk[:end + 1].encode("utf-8"))
        new = []
        for line in chunk[:end].splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("mode") == "eval" and row.get("head") != "pt_head":
                new.append(row)
        self.rows += new
        return new


def monitor(proc, watcher: Watcher, rep: Report, epochs: int) -> bool:
    """Progress bar + live rules. Returns False if the run was stopped by MFlow."""
    last_loss = math.inf
    with Progress(*progress_columns("epochs"), console=LOG.term, transient=True) as progress:
        task = progress.add_task("training", total=epochs)
        while True:
            done = proc.poll() is not None
            for row in watcher.poll():
                loss = float(row.get("loss", math.nan))
                epoch = row.get("epoch")
                e = float(row.get("rmse_e_per_atom", math.nan)) * 1000
                f = float(row.get("rmse_f", math.nan)) * 1000
                label = "initial" if epoch is None else f"epoch {epoch}"
                LOG(f"[dim]{label:>10}[/dim]  loss {loss:10.5f}  RMSE E {e:8.2f} meV/atom  F {f:8.1f} meV/Å")
                if epoch is not None:
                    progress.update(task, completed=epoch + 1,
                                    description=f"training · F {f:.0f} meV/Å")
                if not math.isfinite(loss):
                    rep.check("live: loss finite", "fail", f"NaN/inf validation loss at {label} — stopped",
                              suggest="training diverged: lower lr, check data, or use float64")
                    proc.terminate()
                    return False
                watcher.rising = watcher.rising + 1 if loss > last_loss else 0
                last_loss = loss
                if watcher.rising >= RISING_EVALS and not watcher.warned_rising:
                    watcher.warned_rising = True
                    rep.check("live: valid loss rising", "warn",
                              f"validation loss rose {RISING_EVALS} evaluations in a row (at {label})",
                              suggest="validation loss keeps rising: lr too high or over-fitting")
                    LOG(f"[warn]validation loss rose {RISING_EVALS} times in a row[/warn]")
            now = MONITOR.history.get("gpu_mem", [])
            if now and MONITOR.gpu_total and now[-1] > 0.9 * MONITOR.gpu_total and not watcher.warned_mem:
                watcher.warned_mem = True
                rep.check("live: GPU memory", "warn", f"{now[-1]:.1f}/{MONITOR.gpu_total:.0f} GB",
                          suggest="GPU memory above 90%: lower -batch")
            if done:
                return True
            time.sleep(POLL_S)


def cmd_train(args) -> int:
    t0 = time.perf_counter()
    run_dir = Path(args.outdir or args.name).resolve()
    device = pick_device(args.device)
    foundation = C.resolve_foundation(args.foundation) if args.mode != "scratch" else None

    LOG.rule(f"MFlow {__version__} · train")
    LOG.kv({"input": args.input, "test": args.test or "—", "valid": args.valid or "10% of train",
            "mode": args.mode, "foundation": foundation or "—", "head": args.head or "—",
            "device": device, "dtype": args.dtype, "batch": args.batch,
            "epochs": args.epochs or C.MODE_DEFAULTS[args.mode]["max_num_epochs"], "run dir": str(run_dir)})
    LOG(f"[dim]{versions()}[/dim]")
    rep = Report("train", run_dir=str(run_dir))

    # -- 1. pre-flight: the same rules as `mflow check`, without the zero-shot pass ---- #
    LOG.rule("pre-flight check")
    run_dir.mkdir(parents=True, exist_ok=True)
    frames = read_dataset(args.input, args.index)
    check_args = Namespace(input=args.input, index=args.index, ref_prefix=args.ref_prefix, group=args.group,
                           test=args.test, split=None, model=None, head=None, device=args.device,
                           dtype=args.dtype, outdir=str(run_dir), seed=args.seed, no_plot=True)
    pre = run_check(check_args, frames)
    pre.show("pre-flight")
    for c in pre.checks:
        if c["status"] != "pass":
            rep.check(f"pre-flight: {c['name']}", c["status"], c["message"])
    if pre.failed and not args.force:
        LOG("[err]the dataset failed the pre-flight check — fix it (see `mflow check`) or pass -force[/err]")
        rep.suggestions += pre.suggestions
        rep.write(run_dir / f"{args.name}_train_report.json")
        return 2
    ref = pre.data["ref_prefix"]

    # -- 2. datasets with normalised keys -------------------------------------------- #
    data_dir = run_dir / "data"
    data_dir.mkdir(exist_ok=True)
    files = {}
    for role, path in (("train", args.input), ("valid", args.valid), ("test", args.test)):
        if not path:
            continue
        part = frames if role == "train" else read_dataset(path)
        part_ref = ref if role == "train" else (find_ref(part, exclude="zs_") or ref)
        files[role] = str(data_dir / f"{role}.xyz")
        ase.io.write(files[role], normalise(part, part_ref), format="extxyz")
    has_stress = bool(pre.data.get("has_stress"))
    e0s = None
    isolated = {int(a.numbers[0]) for a in frames if is_isolated_atom(a)}
    if isolated and isolated >= {int(z) for a in frames for z in a.numbers}:
        e0s = "isolated"
        LOG("[dim]E0s: isolated atoms found for every element — using them[/dim]")

    # -- 3. config -------------------------------------------------------------------- #
    cueq = {"on": True, "off": False}.get(args.cueq, device == "cuda" and cueq_available())
    extra = C.parse_set(args.set)
    if args.resume:
        extra["restart_latest"] = True
    try:
        cfg = C.build_config(name=args.name, mode=args.mode, train_file=files["train"],
                             valid_file=files.get("valid"), test_file=files.get("test"),
                             foundation=foundation, head=args.head, pt_file=args.pt, e0s=e0s,
                             has_stress=has_stress, device=device, dtype=args.dtype, cueq=cueq,
                             seed=args.seed, batch=args.batch, epochs=args.epochs,
                             work_dir=str(run_dir), extra=extra)
    except ValueError as exc:
        rep.check("config", "fail", str(exc))
        rep.show()
        rep.write(run_dir / f"{args.name}_train_report.json")
        return 2
    allowed = C.mace_arg_names()
    unknown = C.validate(cfg, allowed)
    if unknown:
        rep.check("config keys", "fail", f"mace_run_train does not know: {', '.join(unknown)}",
                  suggest="remove or rename the -set keys mace_run_train rejects (see mace_run_train --help)")
        rep.show()
        rep.write(run_dir / f"{args.name}_train_report.json")
        return 2
    rep.check("config keys", "pass" if allowed else "warn",
              "all keys known to mace_run_train" if allowed else "mace not importable — keys not validated")
    if cfg.get("E0s") == "isolated":
        cfg.pop("E0s")               # mace reads the IsolatedAtom structures itself
    cfg_path = C.write_yaml(cfg, run_dir / "config.yaml",
                            header=f"# written by MFlow {__version__} · mode {args.mode}\n"
                                   f"# run: mace_run_train --config config.yaml")
    meta = {"_mflow_mode": args.mode, "_mflow_ref_prefix": ref, "_mflow_input": str(Path(args.input).resolve())}
    (run_dir / "mflow_train.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    exe = shutil.which("mace_run_train") or str(Path(sys.executable).with_name("mace_run_train"))
    command = [exe, "--config", str(cfg_path)]
    LOG(f"config [key]{cfg_path}[/key]")
    LOG(f"[dim]{' '.join(command)}[/dim]")
    rep.data.update({"config": cfg, "command": command, "mode": args.mode})
    if args.dry:
        rep.show()
        rep.write(run_dir / f"{args.name}_train_report.json")
        LOG("[ok]dry run[/ok] — config written, mace_run_train not started")
        return 0

    # -- 4. run + watch --------------------------------------------------------------- #
    LOG.rule("mace_run_train")
    out_path = run_dir / "mace_train.out"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t_run = time.perf_counter()
    with open(out_path, "a", encoding="utf-8") as out:
        proc = subprocess.Popen(command, cwd=run_dir, stdout=out, stderr=subprocess.STDOUT, env=env)
        try:
            finished = monitor(proc, Watcher(run_dir, cfg), rep, int(cfg["max_num_epochs"]))
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise
    code = proc.wait()
    MONITOR.report()
    rep.data["train_walltime_s"] = round(time.perf_counter() - t_run, 1)
    rep.data["resources"] = MONITOR.summary()
    if code != 0 and finished:
        tail = out_path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:]
        rep.check("mace_run_train", "fail", f"exit code {code}", value=tail,
                  suggest=f"mace_run_train failed — the last lines are in the report and {out_path}")
        LOG("[err]mace_run_train failed:[/err]\n" + "\n".join(tail[-15:]))
    elif finished:
        rep.check("mace_run_train", "pass", f"finished in {elapsed(rep.data['train_walltime_s'])}")

    # -- 5. analysis ------------------------------------------------------------------ #
    LOG.rule("analysis")
    rep.data.update(analyse_run(run_dir, rep, None, args.group, args.device, not args.no_plot))
    LOG.rule("verdict")
    rep.show()
    rep.write(run_dir / f"{args.name}_train_report.json")
    LOG(f"total [hi]{elapsed(time.perf_counter() - t0)}[/hi]")
    return 0 if not rep.failed else 2

