# -*- coding: utf-8 -*-
"""
One machine-readable verdict per command.

Every check is (name, status, message, value) with status pass | warn | fail; the report's
own status is the worst of them. Suggestions are plain sentences an agent (or a person)
can act on. The json is the contract — the terminal output is only for reading.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from . import __version__
from .log import LOG, table

RANK = {"pass": 0, "warn": 1, "fail": 2}
STYLE = {"pass": "ok", "warn": "warn", "fail": "err"}


class Report:
    def __init__(self, command: str, **meta):
        self.command = command
        self.meta = meta
        self.checks: list[dict] = []
        self.suggestions: list[str] = []
        self.data: dict = {}

    # -- building ------------------------------------------------------------ #
    def check(self, name: str, status: str, message: str = "", value=None, suggest: str | None = None):
        assert status in RANK, status
        self.checks.append({"name": name, "status": status, "message": message, "value": value})
        if suggest and status != "pass" and suggest not in self.suggestions:
            self.suggestions.append(suggest)
        return status

    def suggest(self, text: str):
        if text not in self.suggestions:
            self.suggestions.append(text)

    @property
    def status(self) -> str:
        return max((c["status"] for c in self.checks), key=RANK.get, default="pass")

    @property
    def failed(self) -> bool:
        return self.status == "fail"

    # -- output -------------------------------------------------------------- #
    def show(self, title: str | None = None):
        if self.checks:
            LOG(table(title or f"checks  {self.command}", ["check", "status", "detail"],
                      [(c["name"], f"[{STYLE[c['status']]}]{c['status']}[/]", c["message"])
                       for c in self.checks], styles=["key"]))
        for s in self.suggestions:
            LOG(f"[warn]→[/warn] {s}")
        LOG(f"verdict [{STYLE[self.status]}]{self.status.upper()}[/] · "
            f"{sum(c['status'] == 'pass' for c in self.checks)} pass · "
            f"{sum(c['status'] == 'warn' for c in self.checks)} warn · "
            f"{sum(c['status'] == 'fail' for c in self.checks)} fail")

    def to_dict(self) -> dict:
        return {"mflow": __version__, "command": self.command,
                "time": datetime.now().isoformat(timespec="seconds"),
                "status": self.status, "checks": self.checks, "suggestions": self.suggestions,
                **self.meta, **self.data}

    def write(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=_jsonable),
                        encoding="utf-8")
        LOG(f"json [key]{path}[/key]")
        return path


def _jsonable(value):
    """numpy scalars / arrays and paths in a report should never crash the write."""
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:          # pragma: no cover
        pass
    return str(value)
