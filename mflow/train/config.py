# -*- coding: utf-8 -*-
"""
mace_run_train configuration — one YAML file per run, built from a training mode.

Recommended values follow the MACE fine-tuning guidance
(https://mace-docs.readthedocs.io/en/latest/guide/finetuning_guidance.html):

    mode        lr     ema_decay  clip_grad   notes
    finetune    1e-3   0.999      1.0         naive fine-tuning, best for one system
    multihead   1e-4   0.9999     1.0         replay of pretraining data, keeps transferability
    lora        1e-2   0.99       10.0        low-rank adapters on a frozen foundation
    scratch     1e-2   0.99       10.0        new model, stage-two (SWA) at the end

Energy / force weights stay constant during fine-tuning (a force-then-energy schedule can
destabilise it), so stage two is only switched on for scratch training. Values the
guidance leaves open (epochs, batch, stress weight, r_max) get conservative defaults and
are the knobs to tune later.
"""

from __future__ import annotations

from pathlib import Path

from ..model import MODEL_ALIASES

MODES = ("finetune", "multihead", "lora", "scratch")

# foundation names mace_run_train understands, plus MFlow's short aliases
FOUNDATION_ALIASES = {**MODEL_ALIASES, "mh1": "mh-1", "mh0": "mh-0",
                      "matpes": "mace-matpes-pbe-0", "matpes-r2scan": "mace-matpes-r2scan-0"}
MULTIHEAD_FOUNDATIONS = {"mh-0": "omat_pbe", "mh-1": "omat_pbe"}   # default head to start from

MODE_DEFAULTS = {
    "finetune":  {"lr": 1e-3, "ema": True, "ema_decay": 0.999, "clip_grad": 1.0, "weight_decay": 0.0,
                  "max_num_epochs": 50, "energy_weight": 1.0, "forces_weight": 10.0,
                  "multiheads_finetuning": False},
    "multihead": {"lr": 1e-4, "ema": True, "ema_decay": 0.9999, "clip_grad": 1.0, "weight_decay": 0.0,
                  "max_num_epochs": 20, "energy_weight": 1.0, "forces_weight": 10.0,
                  "multiheads_finetuning": True, "num_samples_pt": 10000,
                  "subselect_pt": "fps", "filter_type_pt": "combinations"},
    "lora":      {"lr": 1e-2, "ema": True, "ema_decay": 0.99, "clip_grad": 10.0, "weight_decay": 0.0,
                  "max_num_epochs": 50, "energy_weight": 1.0, "forces_weight": 10.0,
                  "multiheads_finetuning": False, "lora": True, "lora_rank": 8},
    "scratch":   {"lr": 1e-2, "ema": True, "ema_decay": 0.99, "clip_grad": 10.0,
                  "max_num_epochs": 500, "energy_weight": 1.0, "forces_weight": 100.0,
                  "model": "MACE", "num_channels": 128, "max_L": 1, "r_max": 5.0,
                  "swa": True},
}


def resolve_foundation(spec: str | None) -> str | None:
    """Alias -> mace name; a path stays a path (absolute, since the run has its own cwd)."""
    if not spec or spec == "none":
        return None
    path = Path(spec).expanduser()
    if path.exists():
        return str(path.resolve())
    return FOUNDATION_ALIASES.get(spec.lower(), spec)


def mace_arg_names() -> set[str] | None:
    """Every option mace_run_train accepts in the installed version (None if mace is missing)."""
    try:
        from mace.tools.arg_parser import build_default_arg_parser
    except Exception:
        return None
    names = set()
    for action in build_default_arg_parser()._actions:
        names.add(action.dest)
        names.update(s.lstrip("-") for s in action.option_strings)
    return names


def parse_set(pairs: list[str] | None) -> dict:
    """-set key=value overrides; values are parsed as YAML scalars (1e-3, true, [1, 8])."""
    import yaml
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"-set expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        parsed = yaml.safe_load(value)
        if isinstance(parsed, str):          # YAML 1.1 reads 5e-4 (no dot) as a string
            try:
                parsed = float(parsed)
            except ValueError:
                pass
        out[key.strip()] = parsed
    return out


def build_config(*, name: str, mode: str, train_file: str, valid_file: str | None, test_file: str | None,
                 foundation: str | None, head: str | None, pt_file: str | None, e0s: str | None,
                 has_stress: bool, device: str, dtype: str, cueq: bool, seed: int, batch: int,
                 epochs: int | None, work_dir: str, extra: dict | None = None) -> dict:
    """The full option dict; written as YAML and passed with --config."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode != "scratch" and not foundation:
        raise ValueError(f"mode {mode} needs a foundation model (-foundation)")

    cfg = {"name": name, "seed": seed, "work_dir": work_dir, "device": device, "default_dtype": dtype,
           "train_file": train_file, "energy_key": "REF_energy", "forces_key": "REF_forces",
           "stress_key": "REF_stress", "batch_size": batch, "valid_batch_size": batch,
           "eval_interval": 1, "patience": 20, "save_cpu": True, "error_table": "PerAtomRMSE"}
    if valid_file:
        cfg["valid_file"] = valid_file
    else:
        cfg["valid_fraction"] = 0.1
    if test_file:
        cfg["test_file"] = test_file

    cfg.update(MODE_DEFAULTS[mode])
    if epochs:
        cfg["max_num_epochs"] = epochs
    if mode == "scratch" and "start_swa" not in (extra or {}):
        cfg["start_swa"] = max(1, int(cfg["max_num_epochs"] * 0.75))

    if foundation and mode != "scratch":
        cfg["foundation_model"] = foundation
        if head or foundation in MULTIHEAD_FOUNDATIONS:
            cfg["foundation_head"] = head or MULTIHEAD_FOUNDATIONS[foundation]
        if mode == "multihead":
            cfg["pt_train_file"] = pt_file or "mp"

    # E0s: isolated atoms in the training file win; otherwise estimate from the foundation model
    if e0s:
        cfg["E0s"] = e0s
    elif mode == "scratch":
        cfg["E0s"] = "average"
    else:
        cfg["E0s"] = "estimated"

    cfg["stress_weight"] = 1.0 if has_stress else 0.0
    if has_stress:
        cfg["compute_stress"] = True
    if cueq:
        cfg["enable_cueq"] = True
    cfg.update(extra or {})
    return cfg


def validate(cfg: dict, allowed: set[str] | None) -> list[str]:
    """Keys mace_run_train would reject — caught here, not three minutes into a run."""
    if allowed is None:
        return []
    return sorted(k for k in cfg if k not in allowed)


def write_yaml(cfg: dict, path: Path, header: str = "") -> Path:
    import yaml
    text = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    path.write_text((header + "\n" if header else "") + text, encoding="utf-8")
    return path
