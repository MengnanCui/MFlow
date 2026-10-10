# -*- coding: utf-8 -*-
"""
log.lammps parsing — thermo blocks plus the warnings that matter for quality gates.

A thermo block starts at the header line beginning with "Step" and ends at "Loop time"
(or at the end of the file for a run still going / killed). Several runs (resume, multi-stage)
give several blocks; `merged` stitches them into one table, dropping repeated steps.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def parse_log(path: str | Path) -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    runs, cols, rows = [], None, []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Step ") or stripped == "Step":
            cols, rows = stripped.split(), []
            continue
        if cols is not None:
            if stripped.startswith("Loop time") or stripped.startswith("ERROR") or not stripped:
                if rows:
                    runs.append({"columns": cols, "data": np.array(rows, float)})
                cols, rows = None, []
                continue
            parts = stripped.split()
            if len(parts) == len(cols):
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    pass              # WARNING lines interleaved with thermo output
    if cols is not None and rows:
        runs.append({"columns": cols, "data": np.array(rows, float)})

    perf = re.findall(r"Performance:\s*([\d.eE+-]+)\s*ns/day", text)
    return {
        "runs": runs,
        "errors": [l.strip() for l in text.splitlines() if l.lstrip().startswith("ERROR")],
        "warnings": sorted({l.strip() for l in text.splitlines() if l.lstrip().startswith("WARNING")}),
        "dangerous_builds": [int(x) for x in re.findall(r"Dangerous builds\s*=\s*(\d+)", text)],
        "lost_atoms": bool(re.search(r"Lost atoms", text)),
        "halted": bool(re.search(r"Fix halt condition", text)),
        "ns_per_day": [float(x) for x in perf],
        "atoms": [int(x) for x in re.findall(r"for \d+ steps with (\d+) atoms", text)],
        "finished": "Total wall time" in text,
    }


def merged(log: dict) -> dict[str, np.ndarray]:
    """One column dict over every run that shares the first run's columns; later steps win."""
    if not log["runs"]:
        return {}
    cols = log["runs"][0]["columns"]
    blocks = [r["data"] for r in log["runs"] if r["columns"] == cols]
    data = np.vstack(blocks)
    step = data[:, cols.index("Step")]
    _, last = np.unique(step[::-1], return_index=True)
    keep = np.sort(len(step) - 1 - last)
    return {c: data[keep, i] for i, c in enumerate(cols)}
