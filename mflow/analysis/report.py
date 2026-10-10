# -*- coding: utf-8 -*-
"""
mflow md-report — thermodynamics, structure, dynamics and the quality gates of one MD run.

Reads an `mflow md` directory (md.json, log.lammps, traj.lammpstrj); a plain LAMMPS run
works too if you pass -ensemble / -temp yourself.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import ase.io

from .. import __version__
from ..log import LOG, num, table
from ..report import Report
from ..style import COLORS, LINESTYLES, plt
from . import gates
from .thermo import merged, parse_log
from .traj import diffusion, min_distances, msd, rdf, rdf_difference, read_dump


def analyse_md(run_dir: Path, rep: Report, *, ensemble: str | None = None, temp: float | None = None,
               skip: float = 0.2, stride: int = 1, ref: str | None = None, plot: bool = True) -> dict:
    meta_file = run_dir / "md.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    params = meta.get("params", {})
    ensemble = ensemble or params.get("ensemble", "nvt")
    temp_end = temp or params.get("temp_end") or params.get("temp", 300.0)
    ramp = params.get("temp") is not None and abs(params.get("temp", temp_end) - temp_end) > 1e-9
    out = {"run_dir": str(run_dir.resolve()), "ensemble": ensemble}

    log_file = run_dir / "log.lammps"
    if not log_file.exists():
        rep.check("log.lammps", "fail", f"not found in {run_dir}")
        return out
    log = parse_log(log_file)
    gates.gate_health(log, rep)
    th = merged(log)
    natoms = log["atoms"][-1] if log["atoms"] else meta.get("natoms", 1)
    out["natoms"] = natoms
    out["ns_per_day"] = log["ns_per_day"][-1] if log["ns_per_day"] else None
    out.update(gates.gate_energy(th, ensemble, natoms, rep, skip))
    out.update(gates.gate_temperature(th, temp_end, ensemble, rep, skip, ramp))
    out.update(gates.gate_volume(th, ensemble, rep, skip))
    if th:
        sl = gates.production(th["Step"], skip)
        LOG(table(f"thermo  (production: last {100 * (1 - skip):.0f} %)", ["", "mean", "std", "unit"],
                  [(name, num(np.mean(th[col][sl]), 4), num(np.std(th[col][sl]), 4), unit)
                   for col, name, unit in (("Temp", "temperature", "K"), ("Press", "pressure", "bar"),
                                           ("PotEng", "potential energy", "eV"), ("TotEng", "total energy", "eV"),
                                           ("Volume", "volume", "Å³"), ("Density", "density", "g/cm³"))
                   if col in th], styles=["key", None, None, "dim"]))
        out["steps_done"] = int(th["Step"][-1])

    # -- trajectory --------------------------------------------------------------- #
    dump = run_dir / "traj.lammpstrj"
    rdf_data = msd_data = None
    if dump.exists():
        frames = read_dump(dump, stride, params.get("species"))
        prod = frames[int(len(frames) * skip):] or frames
        LOG(f"trajectory [key]{dump.name}[/key] · {len(frames)} frames read (stride {stride}) · "
            f"{len(prod)} in production")
        out.update(gates.gate_geometry(min_distances(frames), rep))
        if any(frames[0].pbc):
            rdf_data = rdf(prod[:: max(1, len(prod) // 200)])
            out["rdf_first_peak"] = {k: float(rdf_data["r"][np.argmax(v)]) for k, v in rdf_data.items() if k != "r"}
            if ref:
                ref_frames = ase.io.read(ref, ":")
                diff = rdf_difference(rdf_data, rdf(ref_frames))
                out["rdf_mae_vs_ref"] = diff
                worst = max(diff.values()) if diff else 0.0
                rep.check("RDF vs reference", "pass" if worst < 0.1 else ("warn" if worst < 0.3 else "fail"),
                          " · ".join(f"{k} {v:.3f}" for k, v in diff.items()),
                          suggest="structure differs from the reference trajectory: check the potential "
                                  "on these configurations")
        dt_frame = params.get("dt", 1.0) * 1e-3 * params.get("dump", 1) * stride
        msd_data = msd(prod, dt_frame)
        if msd_data:
            out["diffusion_cm2_s"] = diffusion(msd_data)
            LOG(table("diffusion (MSD slope / 6)", ["element", "D (cm²/s)", "MSD at end (Å²)"],
                      [(el, f"{d:.3e}", num(msd_data[el][-1], 3)) for el, d in out["diffusion_cm2_s"].items()],
                      styles=["key"]))
    else:
        rep.check("trajectory", "warn", "no traj.lammpstrj — structural checks skipped")

    if plot and th:
        make_md_plot(th, ensemble, temp_end, rdf_data, msd_data, str(run_dir / "md_summary.png"), skip)
    return out


def make_md_plot(th, ensemble, temp, rdf_data, msd_data, outfile, skip):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    t = th["Time"]
    t0 = t[int(len(t) * skip)] if len(t) else 0
    panels = [("Temp", "Temperature (K)", temp if ensemble != "nve" else None),
              ("Press", "Pressure (bar)", None),
              ("TotEng" if ensemble == "nve" else "PotEng",
               "Total energy (eV)" if ensemble == "nve" else "Potential energy (eV)", None),
              ("Volume", r"Volume ($\mathrm{\AA}^3$)", None)]
    for ax, (col, label, target), color in zip(axes.ravel()[:4], panels, COLORS):
        if col in th:
            ax.plot(t, th[col], color=color, lw=1.2)
            if target is not None:
                ax.axhline(target, color="black", ls=LINESTYLES[1], lw=1.5)
            ax.axvspan(t[0], t0, color="grey", alpha=0.15, lw=0)
        ax.set(xlabel="Time (ps)", ylabel=label)
    ax = axes[1, 1]
    if rdf_data:
        for i, (k, g) in enumerate((k, v) for k, v in rdf_data.items() if k != "r"):
            ax.plot(rdf_data["r"], g, color=COLORS[i % len(COLORS)], label=k)
        ax.legend()
    ax.set(xlabel=r"r ($\mathrm{\AA}$)", ylabel="g(r)")
    ax = axes[1, 2]
    if msd_data:
        for i, (k, y) in enumerate((k, v) for k, v in msd_data.items() if k != "t_ps"):
            ax.plot(msd_data["t_ps"], y, color=COLORS[i % len(COLORS)], label=k)
        ax.legend()
    ax.set(xlabel="Lag time (ps)", ylabel=r"MSD ($\mathrm{\AA}^2$)")
    for ax in axes.ravel():
        ax.tick_params(which="both", direction="in")
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    LOG(f"figure [key]{outfile}[/key]")


def cmd_md_report(args) -> int:
    run_dir = Path(args.dir)
    LOG.rule(f"MFlow {__version__} · md-report")
    LOG.kv({"run": str(run_dir), "skip": f"{args.skip:.0%} equilibration", "stride": args.stride,
            "reference": args.ref or "—"})
    rep = Report("md-report")
    rep.data.update(analyse_md(run_dir, rep, ensemble=args.ensemble, temp=args.temp, skip=args.skip,
                               stride=args.stride, ref=args.ref, plot=not args.no_plot))
    LOG.rule("verdict")
    rep.show()
    rep.write(run_dir / "md_report.json")
    return 0 if not rep.failed else 2
