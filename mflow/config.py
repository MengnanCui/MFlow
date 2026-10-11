# -*- coding: utf-8 -*-
"""
User settings that would otherwise be typed on every command.

    mflow config                         show every key, its value and where it comes from
    mflow config set model ~/models/cu.model
    mflow config set lmp /opt/lammps/build/lmp
    mflow config unset model

Lookup order for a key: environment variable MFLOW_<KEY> > config file > built-in default.
The file is $MFLOW_CONFIG, else $XDG_CONFIG_HOME/mflow/config.yaml, else ~/.config/mflow/config.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path

KEYS = {
    "model": "default MACE model for calc / relax / check / validate: a .model file or a foundation "
             "name (mpa, omat, medium, ...)",
    "lmp": "LAMMPS executable built with KOKKOS + ML-IAP + PYTHON",
    "skills_dir": "where `mflow skills install` puts the skills [~/.agents/skills]",
}
# the model this project was developed with; used only if it exists on this machine
LEGACY_MODEL = "/home/cmn01/software/mace-main032026/models/mace-mpa-0-medium.model"
FALLBACK_MODEL = "mpa"                     # MACE-MPA-0 medium, downloaded by mace on first use


def config_path() -> Path:
    if os.environ.get("MFLOW_CONFIG"):
        return Path(os.environ["MFLOW_CONFIG"]).expanduser()
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "mflow" / "config.yaml"


def load() -> dict:
    path = config_path()
    if not path.exists():
        return {}
    import yaml
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def lookup(key: str) -> tuple[str | None, str]:
    """(value, source) with source one of env / file / default / unset."""
    env = os.environ.get(f"MFLOW_{key.upper()}")
    if env:
        return env, f"env MFLOW_{key.upper()}"
    data = load()
    if data.get(key) not in (None, ""):
        return str(data[key]), "config file"
    if key == "model":
        if Path(LEGACY_MODEL).is_file():
            return LEGACY_MODEL, "default (local file)"
        return FALLBACK_MODEL, "default (downloaded on first use)"
    if key == "skills_dir":
        return str(Path.home() / ".agents" / "skills"), "default"
    return None, "unset"


def get(key: str) -> str | None:
    return lookup(key)[0]


def default_model() -> str:
    return get("model") or FALLBACK_MODEL


def save(data: dict) -> Path:
    import yaml
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# MFlow user settings — edit freely or use `mflow config set/unset`\n"
    path.write_text(text + yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
    return path


def cmd_config(args) -> int:
    from .log import LOG, table
    action = args.action or "show"
    if action == "path":
        print(config_path())
        return 0
    if action in ("set", "unset"):
        if args.key not in KEYS:
            LOG(f"[err]unknown key {args.key!r}[/err] — keys: {', '.join(KEYS)}")
            return 2
        data = load()
        if action == "set":
            if args.value is None:
                LOG("[err]mflow config set <key> <value>[/err]")
                return 2
            value = args.value
            if args.key in ("model", "lmp", "skills_dir") and ("/" in value or value.startswith("~")):
                value = str(Path(value).expanduser().resolve())
                if args.key != "skills_dir" and not Path(value).exists():
                    LOG(f"[warn]{value} does not exist (saved anyway)[/warn]")
            data[args.key] = value
        else:
            data.pop(args.key, None)
        path = save(data)
        LOG(f"[ok]{action}[/ok] {args.key}" + (f" = {data[args.key]}" if action == "set" else "") + f" · {path}")
        return 0
    rows = []
    for key, doc in KEYS.items():
        value, source = lookup(key)
        rows.append((key, value or "—", source, doc))
    t = table(f"mflow config · {config_path()}", ["key", "value", "from", "meaning"], rows,
              styles=["key", "hi", "dim", "dim"])
    for col in t.columns:
        col.justify, col.no_wrap, col.overflow = "left", False, "fold"
    t.columns[0].min_width = 10
    LOG(t)
    LOG("[dim]change: mflow config set <key> <value> · environment variables MFLOW_<KEY> win over the file[/dim]")
    return 0
