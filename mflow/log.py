# -*- coding: utf-8 -*-
"""Terminal + log-file output, tables, and the live resource monitor."""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, ProgressColumn,
                          TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

GIB = 1024 ** 3   # memory is reported in GiB, the unit nvidia-smi and free agree on


class Log:
    """Terminal (colour) + log file (plain) in one call. Time stamps only on rules."""

    THEME = Theme({"key": "cyan", "ok": "green", "warn": "yellow", "err": "bold red", "hi": "bold"})

    def __init__(self):
        self.term = Console(theme=self.THEME, highlight=False)
        self.file = self.fh = None

    def open(self, path: str):
        self.fh = open(path, "w", encoding="utf-8")
        self.file = Console(file=self.fh, theme=self.THEME, no_color=True, width=110, highlight=False)

    def __call__(self, *args, **kwargs):
        self.term.print(*args, **kwargs)
        if self.file:
            self.file.print(*args, **kwargs)
            self.fh.flush()

    def rule(self, text: str):
        """Section marker — this is where the time stamp goes."""
        stamp = datetime.now().strftime("%H:%M:%S")
        self.term.rule(f"[dim]{stamp}[/dim] [hi]{text}[/hi]", align="left", style="cyan")
        if self.file:
            self.file.rule(f"{stamp} {text}", align="left")
            self.fh.flush()

    def kv(self, items: dict, cols: int = 2):
        """Dense key/value block: `cols` key-value pairs per row, no borders."""
        table = Table.grid(padding=(0, 2))
        for _ in range(cols):
            table.add_column(style="key", justify="right")
            table.add_column(max_width=46)
        pairs = [(str(k), str(v)) for k, v in items.items()]
        pairs += [("", "")] * (-len(pairs) % cols)
        for i in range(0, len(pairs), cols):
            table.add_row(*[x for pair in pairs[i:i + cols] for x in pair])
        self(table)


LOG = Log()


def table(title: str, columns: list, rows: list, styles: list | None = None) -> Table:
    t = Table(title=title, title_justify="left", title_style="bold", header_style="bold cyan",
              show_edge=False, box=None, pad_edge=False, padding=(0, 2))
    styles = list(styles or []) + [None] * len(columns)
    for i, col in enumerate(columns):
        t.add_column(col, justify="left" if i == 0 else "right", style=styles[i], no_wrap=True)
    for row in rows:
        t.add_row(*[str(x) for x in row])
    return t


def elapsed(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def num(value: float, digits: int = 2) -> str:
    """Fixed point while it stays readable, scientific once it would blow up the column."""
    if value is None:
        return "—"
    return f"{value:.{digits}f}" if abs(value) < 1e5 else f"{value:.3g}"


# ============================================================================ #
#  resource monitor — what the run is actually using, live on the progress bar
# ============================================================================ #

class Monitor:
    """
    GPU / CPU / RAM sampling with a fallback chain, so a missing package never
    costs more than the reading itself. Numbers are whole-device: with several
    workers on one card that is exactly what you need to size -nproc, but it is
    not "how much MFlow uses".
    """

    def __init__(self):
        self.gpu, self.cpu, self.ram = None, None, None   # backend names, None = unavailable
        self.history = {"gpu_util": [], "gpu_mem": [], "cpu": [], "ram": []}
        self.gpu_total = self.ram_total = 0.0             # GB
        self._nvml = self._psutil = None
        self._cpu_prev = None
        if not os.environ.get("MFLOW_NO_MONITOR"):        # escape hatch if a backend misbehaves
            self._setup()

    # -- backends ----------------------------------------------------------- #
    def _setup(self):
        try:                                              # nvml needs no CUDA context
            import pynvml
            pynvml.nvmlInit()
            self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
            self.gpu = "pynvml"
            self.gpu_total = pynvml.nvmlDeviceGetMemoryInfo(self._nvml[1]).total / GIB
        except Exception:
            if shutil.which("nvidia-smi"):
                self.gpu = "nvidia-smi"
        try:
            import psutil
            self._psutil = psutil
            psutil.cpu_percent(interval=None)             # prime the delta
            self.cpu = self.ram = "psutil"
            self.ram_total = psutil.virtual_memory().total / GIB
        except Exception:
            if Path("/proc/stat").exists():
                self.cpu, self._cpu_prev = "/proc/stat", self._proc_cpu_raw()
            if Path("/proc/meminfo").exists():
                self.ram = "/proc/meminfo"

    @staticmethod
    def _proc_cpu_raw():
        fields = [float(x) for x in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
        return sum(fields), sum(fields) - fields[3] - (fields[4] if len(fields) > 4 else 0.0)

    def _read_gpu(self) -> tuple:
        if self.gpu == "pynvml":
            pynvml, handle = self._nvml
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu), memory.used / GIB
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        util, used, total = (float(x) for x in out.splitlines()[0].split(","))   # MiB
        self.gpu_total = total / 1024
        return util, used / 1024

    def _read_cpu(self) -> float:
        if self.cpu == "psutil":
            return self._psutil.cpu_percent(interval=None) * (os.cpu_count() or 1) / 100
        total, busy = self._proc_cpu_raw()
        prev_total, prev_busy = self._cpu_prev
        self._cpu_prev = (total, busy)
        span = total - prev_total
        return (busy - prev_busy) / span * (os.cpu_count() or 1) if span > 0 else 0.0

    def _read_ram(self) -> float:
        if self.ram == "psutil":
            memory = self._psutil.virtual_memory()
            return (memory.total - memory.available) / GIB
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines()[:5]:
            key, _, value = line.partition(":")
            info[key] = float(value.split()[0]) / 1024 ** 2                # kB -> GiB
        self.ram_total = info.get("MemTotal", 0.0)
        return info.get("MemTotal", 0.0) - info.get("MemAvailable", 0.0)

    # -- sampling ----------------------------------------------------------- #
    def sample(self) -> dict:
        """One reading. Any backend that throws is switched off for the rest of the run."""
        now = {}
        for name, reader, keys_ in (("gpu", self._read_gpu, ("gpu_util", "gpu_mem")),
                                    ("cpu", self._read_cpu, ("cpu",)),
                                    ("ram", self._read_ram, ("ram",))):
            if getattr(self, name) is None:
                continue
            try:
                values = reader()
                values = values if isinstance(values, tuple) else (values,)
                for key, value in zip(keys_, values):
                    now[key], _ = value, self.history[key].append(value)
            except Exception:
                setattr(self, name, None)                 # never let monitoring break a run
        return now

    # -- rendering ---------------------------------------------------------- #
    @staticmethod
    def _band(fraction: float, low_is_bad: bool = True) -> str:
        """Idle GPU means raise -nproc; nearly full memory means lower it."""
        if low_is_bad:
            return "ok" if fraction >= 0.7 else ("warn" if fraction >= 0.3 else "err")
        return "err" if fraction >= 0.9 else ("warn" if fraction >= 0.75 else "ok")

    def line(self) -> str:
        """
        Compact markup for the progress bar. Parts are dropped from the right when the
        terminal is too narrow, so GPU — the number you tune -nproc by — survives longest.
        """
        now, parts = self.sample(), []                    # (plain text, markup)
        if "gpu_util" in now:
            util, mem = now["gpu_util"], now.get("gpu_mem", 0.0)
            plain = f"GPU {util:3.0f}% {mem:.1f}/{self.gpu_total:.0f}GB"
            parts.append((plain, f"[{self._band(util / 100)}]GPU {util:3.0f}%[/] "
                                 f"[{self._band(mem / self.gpu_total if self.gpu_total else 0, False)}]"
                                 f"{mem:.1f}/{self.gpu_total:.0f}GB[/]"))
        if "cpu" in now:
            cores = os.cpu_count() or 1
            style = "warn" if now["cpu"] > 0.95 * cores else "dim"   # busy is fine, oversubscribed is not
            plain = f"CPU {now['cpu'] * 100:.0f}%"
            parts.append((plain, f"[{style}]{plain}[/]"))
        if "ram" in now:
            plain = f"RAM {now['ram']:.1f}/{self.ram_total:.0f}GB"
            parts.append((plain, f"[{self._band(now['ram'] / self.ram_total if self.ram_total else 0, False)}]"
                                 f"{plain}[/]"))

        budget = LOG.term.width - 60                      # what the bar itself needs
        while parts and 2 + sum(len(p) for p, _ in parts) + 3 * (len(parts) - 1) > budget:
            parts.pop()
        return "[dim]│[/dim] " + " [dim]·[/dim] ".join(m for _, m in parts) if parts else ""

    def summary(self) -> dict:
        """min / mean / max per metric, no time series kept."""
        out = {}
        for key, values in self.history.items():
            if values:
                out[key] = {"mean": round(float(np.mean(values)), 2),
                            "peak": round(float(np.max(values)), 2)}
        out["backends"] = {"gpu": self.gpu, "cpu": self.cpu, "ram": self.ram}
        if self.gpu_total:
            out["gpu_total_gb"] = round(self.gpu_total, 1)
        if self.ram_total:
            out["ram_total_gb"] = round(self.ram_total, 1)
        return out

    def report(self):
        data, parts = self.summary(), []
        if "gpu_util" in data:
            parts.append(f"GPU util mean {data['gpu_util']['mean']:.0f}% peak {data['gpu_util']['peak']:.0f}%")
        if "gpu_mem" in data:
            parts.append(f"GPU mem peak {data['gpu_mem']['peak']:.1f}/{self.gpu_total:.1f} GB")
        if "cpu" in data:
            parts.append(f"CPU mean {data['cpu']['mean'] * 100:.0f}% of {(os.cpu_count() or 1) * 100}%")
        if "ram" in data:
            parts.append(f"RAM peak {data['ram']['peak']:.1f}/{self.ram_total:.1f} GB")
        if parts:
            LOG(f"[dim]resources ·[/dim] " + " [dim]·[/dim] ".join(parts))


MONITOR = Monitor()


class ResourceColumn(ProgressColumn):
    """Live resource readout appended to the bar. rich re-renders it at most once a second."""

    max_refresh = 1.0

    def render(self, task) -> Text:
        return Text.from_markup(MONITOR.line(), style="none")


def progress_columns(unit: str) -> tuple:
    return ("[progress.description]{task.description}", BarColumn(), MofNCompleteColumn(), unit,
            TimeElapsedColumn(), "eta", TimeRemainingColumn(), ResourceColumn())

