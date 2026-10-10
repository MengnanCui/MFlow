# -*- coding: utf-8 -*-
"""Campaign settings, state and the append-only ledger."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import ase.io

from .. import __version__
from ..log import LOG

DEFAULTS = {
    "foundation": "mpa",          # explores round 0 and is the starting point of every fine-tune
    "train": {"mode": "finetune", "committee": 4, "bagging": None, "epochs": None, "batch": 8,
              "dtype": "float32", "device": "auto", "set": []},
    "seed": {"n_select": 100},    # round 0: diversity only (no committee yet)
    "explore": {
        "md": {"enabled": True, "engine": "ase", "temps": [300, 600, 900], "press": [1.0], "steps": 2000,
               "dump": 20, "dt": 1.0, "deploy": None, "widen": 1.25},
        "perturb": {"enabled": True, "n": 20, "rattle": 0.05, "strain": 0.03, "vacancies": 0},
        "minhop": {"enabled": False, "steps": 20, "T0": 1000.0, "fmax": 0.05},
    },
    "select": {"trust": "fixed", "lo": None, "hi": None, "recalibrate": True, "dedupe": "fps",
               "max_per_round": None},
    "stop": {"candidate_fraction": None, "dft_energy_meV_atom": None, "dft_force_meV_A": None},
}


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Campaign:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    # -- paths --------------------------------------------------------------- #
    @property
    def data(self) -> Path:
        return self.root / "data"

    def round_dir(self, r: int, part: str = "") -> Path:
        d = self.root / f"round_{r:02d}" / part
        d.mkdir(parents=True, exist_ok=True)
        return d

    def labelled(self, r: int) -> Path:
        return self.data / "labelled" / f"round_{r:02d}.xyz"

    def train_file(self, version: int) -> Path:
        return self.data / f"train_v{version:02d}.xyz"

    # -- settings / state ------------------------------------------------------- #
    @property
    def settings(self) -> dict:
        import yaml
        from ..thresholds import section
        al = section("al")
        defaults = deep_merge(DEFAULTS, {
            "select": {"lo": al["trust_lo_eV_A"], "hi": al["trust_hi_eV_A"], "max_per_round": al["max_per_round"]},
            "stop": {"candidate_fraction": al["stop_candidate_fraction"],
                     "dft_energy_meV_atom": al["stop_dft_energy_meV_atom"],
                     "dft_force_meV_A": al["stop_dft_force_meV_A"]}})
        user = yaml.safe_load((self.root / "campaign.yaml").read_text(encoding="utf-8")) or {}
        return deep_merge(defaults, user)

    @property
    def state(self) -> dict:
        path = self.root / "state.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def save_state(self, **changes) -> dict:
        state = {**self.state, **changes, "updated": datetime.now().isoformat(timespec="seconds")}
        (self.root / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        return state

    # -- ledger ------------------------------------------------------------------ #
    def log(self, step: str, round_: int, **data):
        rec = {"time": datetime.now().isoformat(timespec="seconds"), "mflow": __version__, "step": step,
               "round": round_, **data}
        with open(self.root / "ledger.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def ledger(self) -> list[dict]:
        path = self.root / "ledger.jsonl"
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    # -- models ------------------------------------------------------------------ #
    def committee(self) -> list[str]:
        return list(self.state.get("models") or [])

    def explorer_model(self) -> str:
        """Member 0 of the current committee, or the foundation model before any training."""
        models = self.committee()
        return models[0] if models else self.settings["foundation"]

    # -- dataset ------------------------------------------------------------------- #
    def rebuild_dataset(self, version: int) -> tuple[Path, int]:
        """initial + every labelled round, rewritten from scratch (no appending)."""
        frames = []
        if (self.data / "initial.xyz").exists():
            frames += ase.io.read(self.data / "initial.xyz", ":")
        for path in sorted((self.data / "labelled").glob("round_*.xyz")):
            frames += ase.io.read(path, ":")
        out = self.train_file(version)
        ase.io.write(out, frames, format="extxyz")
        return out, len(frames)


def cmd_al_init(args) -> int:
    root = Path(args.name).resolve()
    if (root / "campaign.yaml").exists() and not args.force:
        LOG(f"[err]{root}/campaign.yaml exists[/err] — pick another -name or pass -force")
        return 2
    (root / "data" / "labelled").mkdir(parents=True, exist_ok=True)
    system = ase.io.read(args.system, ":")
    ase.io.write(root / "data" / "system.xyz", system, format="extxyz")
    for src, dst in ((args.data, "initial.xyz"), (args.test, "test.xyz")):
        if src:
            shutil.copy(src, root / "data" / dst)
    import yaml
    user = {"foundation": args.foundation,
            "train": {"mode": args.mode, "committee": args.committee},
            "explore": {"md": {"engine": args.engine}}}
    text = ("# MFlow active-learning campaign. Every key not listed here takes the default from\n"
            "# mflow/al/campaign.py (DEFAULTS) and mflow/thresholds.yaml (section al).\n"
            "# Edit between rounds; each step records the settings it used in ledger.jsonl.\n")
    (root / "campaign.yaml").write_text(text + yaml.safe_dump(user, sort_keys=False), encoding="utf-8")
    camp = Campaign(root)
    camp.save_state(round=0, stage="init", models=[], dataset=None)
    camp.log("init", 0, system=str(Path(args.system).resolve()), n_system=len(system),
             initial=bool(args.data), test=bool(args.test), settings=camp.settings)
    LOG(f"[ok]campaign[/ok] [key]{root}[/key] · {len(system)} start structure(s)"
        + (f" · initial data {args.data}" if args.data else "") + (f" · test {args.test}" if args.test else ""))
    LOG("next: [hi]mflow al run -name " + args.name + "[/hi]  (or step by step: explore → select → export)")
    return 0
