# -*- coding: utf-8 -*-
"""
mflow skills — the Agent Skills shipped with MFlow, and putting them where agents look.

    mflow skills                      list them, and whether they are installed
    mflow skills install              symlink every skill into ~/.agents/skills (pi, and other agents
                                      that read ~/.agents/skills)
    mflow skills install -dest ~/.pi/agent/skills -copy
    mflow skills path                 print the directory that holds the shipped skills

Symlinks follow the installed MFlow, so an upgrade updates the skills too; -copy freezes them.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from .log import LOG, table


def source_dir() -> Path:
    """mflow/_skills inside a wheel install, else <repo>/skills for a source checkout / editable install."""
    here = Path(__file__).resolve().parent
    for cand in (here / "_skills", here.parent / "skills"):
        if cand.is_dir() and any(cand.glob("*/SKILL.md")):
            return cand
    raise FileNotFoundError("the MFlow skills are not installed with this copy of mflow")


def shipped() -> list[Path]:
    return sorted(p.parent for p in source_dir().glob("*/SKILL.md"))


def description(skill: Path) -> str:
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"^description:\s*(.+)$", text, re.M)
    return m.group(1).strip() if m else ""


def destination(dest: str | None = None) -> Path:
    from .config import get
    return Path(dest or get("skills_dir")).expanduser()


def installed_state(dest: str | None = None) -> tuple[Path, dict]:
    target = destination(dest)
    try:
        skills = shipped()
    except FileNotFoundError:
        return target, {"total": 0, "installed": 0}
    ok = sum((target / s.name / "SKILL.md").exists() for s in skills)
    return target, {"total": len(skills), "installed": ok}


def install(dest: Path, copy: bool = False, force: bool = False) -> list[tuple[str, str]]:
    dest.mkdir(parents=True, exist_ok=True)
    done = []
    for skill in shipped():
        target = dest / skill.name
        if target.is_symlink() or target.exists():
            same = target.is_symlink() and target.resolve() == skill.resolve()
            if same and not copy:
                done.append((skill.name, "already linked"))
                continue
            if not force:
                done.append((skill.name, "exists — kept (use -force to replace)"))
                continue
            if target.is_symlink() or target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target)
        if copy:
            shutil.copytree(skill, target)
            done.append((skill.name, "copied"))
        else:
            target.symlink_to(skill.resolve(), target_is_directory=True)
            done.append((skill.name, "linked"))
    return done


def cmd_skills(args) -> int:
    action = args.action or "list"
    try:
        src = source_dir()
    except FileNotFoundError as exc:
        LOG(f"[err]{exc}[/err]")
        return 2
    if action == "path":
        print(src)
        return 0
    dest = destination(args.dest)
    if action == "install":
        rows = install(dest, copy=args.copy, force=args.force)
        LOG(table(f"skills → {dest}", ["skill", "result"], rows, styles=["key"]))
        LOG("agents that read this directory (pi: ~/.agents/skills or ~/.pi/agent/skills) pick them up on "
            "their next start; other locations: -dest DIR, or mflow config set skills_dir DIR")
        return 0
    rows = []
    for skill in shipped():
        there = (dest / skill.name / "SKILL.md").exists()
        rows.append((skill.name, "[ok]yes[/ok]" if there else "[dim]no[/dim]", description(skill)[:90] + "…"))
    t = table(f"MFlow skills · source {src} · installed in {dest}?", ["skill", "installed", "what for"], rows,
              styles=["key"])
    t.columns[2].no_wrap = False
    LOG(t)
    if not all("yes" in r[1] for r in rows):
        LOG("install: [hi]mflow skills install[/hi]")
    return 0
