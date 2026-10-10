# -*- coding: utf-8 -*-
"""
mflow md-report — thermodynamics, structure, dynamics and the quality gates of one MD run.

Reads an `mflow md` directory (md.json, log.lammps, traj.lammpstrj); a plain LAMMPS run
works too if you pass -ensemble / -temp yourself.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import ase.io

from .. import __version__
from ..log import LOG, num, table
from ..report import Report
from ..style import COLORS, LINESTYLES, plt
from . import gates
from .thermo import merged, parse_log
from .stats import block_average
from .traj import (diffusion, diffusion_axes, dump_steps, min_distances, msd, msd_loglog_slope, rdf,
                   rdf_difference, read_dump, select_steps, vdos, yeh_hummer)


def production_window(th: dict, stages: list, skip: float) -> tuple[dict, float, dict | None]:
    """
    Thermo rows the gates look at. Single stage: everything, gates skip `skip` of it. A protocol:
    only the last stage (the earlier ones are its equilibration), nothing skipped inside it.
    """
    if not th or not stages or len(stages) == 1:
        return th, skip, (stages[-1] if stages else None)
    last = stages[-1]
    m = th["Step"] >= last["start"]
    return {k: v[m] for k, v in th.items()}, 0.0, last


def analyse_md(run_dir: Path, rep: Report, *, ensemble: str | None = None, temp: float | None = None,
               skip: float = 0.2, stride: int = 1, ref: str | None = None, plot: bool = True,
               viscosity: float | None = None) -> dict:
    meta_file = run_dir / "md.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    params = meta.get("params", {})
    stages = meta.get("stages") or []
    protocol = params.get("protocol", "single")
    out = {"run_dir": str(run_dir.resolve()), "protocol": protocol}

    log_file = run_dir / "log.lammps"
    if not log_file.exists():
        rep.check("log.lammps", "fail", f"not found in {run_dir}")
        return out
    log = parse_log(log_file)
    gates.gate_health(log, rep)
    th_all = merged(log)
    natoms = log["atoms"][-1] if log["atoms"] else meta.get("natoms", 1)
    out["natoms"] = natoms
    out["ns_per_day"] = log["ns_per_day"][-1] if log["ns_per_day"] else None
    if protocol == "minimize":
        out.update(analyse_minimize(run_dir, th_all, params, natoms, rep))
        return out

    th, skip_eff, last = production_window(th_all, stages, skip)
    ensemble = ensemble or (last["ensemble"] if last else params.get("ensemble", "nvt"))
    temp_end = temp or (last["temp_end"] if last else None) or params.get("temp_end") or params.get("temp", 300.0)
    ramp = bool(last and abs(last["temp"] - last["temp_end"]) > 1e-9) or \
        (not last and params.get("temp") is not None and abs(params.get("temp", temp_end) - temp_end) > 1e-9)
    out["ensemble"] = ensemble
    if last and len(stages) > 1:
        out["production_stage"] = last["name"]
        LOG(f"protocol [key]{protocol}[/key]: gates on the last stage '{last['name']}' ({last['ensemble']}, "
            f"steps {last['start']}–{last['end']}); earlier stages are its equilibration")
    out.update(gates.gate_energy(th, ensemble, natoms, rep, skip_eff))
    out.update(gates.gate_temperature(th, temp_end, ensemble, rep, skip_eff, ramp))
    out.update(gates.gate_volume(th, ensemble, rep, skip_eff))
    if th:
        sl = gates.production(th["Step"], skip_eff)
        cols = [(c, n, u) for c, n, u in (("Temp", "temperature", "K"), ("Press", "pressure", "bar"),
                                          ("PotEng", "potential energy", "eV"), ("TotEng", "total energy", "eV"),
                                          ("Volume", "volume", "Å³"), ("Density", "density", "g/cm³")) if c in th]
        blocks = {c: block_average(th[c][sl]) for c, _, _ in cols}
        out["block_averages"] = blocks
        LOG(table("thermo  (production; ± block-averaged standard error)", ["", "mean", "± SE", "std", "n_eff", "unit"],
                  [(name, num(blocks[c]["mean"], 4), num(blocks[c]["se"], 4), num(np.std(th[c][sl]), 4),
                    f"{blocks[c]['n_eff']:.0f}", unit) for c, name, unit in cols],
                  styles=["key", None, None, None, "dim", "dim"]))
        out["steps_done"] = int(th_all["Step"][-1])
        if not ramp:
            gates.gate_halves(th, sl, ensemble, rep, out)

    # -- trajectory --------------------------------------------------------------- #
    dump = run_dir / "traj.lammpstrj"
    rdf_data = msd_data = vd = None
    if dump.exists():
        frames = read_dump(dump, stride, params.get("species"))
        steps = dump_steps(dump)[::max(stride, 1)][:len(frames)]
        if last and len(stages) > 1:
            prod = select_steps(frames, steps, last["start"])
        else:
            uniq = select_steps(frames, steps, 0)
            prod = uniq[int(len(uniq) * skip):] or uniq
        LOG(f"trajectory [key]{dump.name}[/key] · {len(frames)} frames read (stride {stride}) · "
            f"{len(prod)} in production")
        out.update(gates.gate_geometry(min_distances(frames), rep))
        if any(frames[0].pbc) and prod:
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
        if not ramp and len(prod) >= 10:
            msd_data = msd(prod, dt_frame)
        if msd_data:
            out.update(analyse_diffusion(prod, msd_data, dt_frame, ensemble, temp_end, viscosity, rep))
        if len(prod) >= 16:
            vd = vdos(prod, dt_frame)
            if vd and vd["nyquist_THz"] < 20:
                LOG(f"[dim]VDOS skipped: frames {dt_frame * 1000:g} fs apart resolve only up to "
                    f"{vd['nyquist_THz']:.1f} THz — dump every ≤ 10 fs (-dump) for a vibrational spectrum[/dim]")
                vd = None
            elif vd:
                peak = float(vd["nu_THz"][np.argmax(vd["g"][1:]) + 1])
                out["vdos_peak_THz"] = peak
                np.savetxt(run_dir / "vdos.dat", np.column_stack([vd["nu_THz"], vd["g"]]),
                           header="frequency_THz  g(nu) (unit area)")
                LOG(f"VDOS: main peak {peak:.2f} THz → [key]{run_dir / 'vdos.dat'}[/key]")
    else:
        rep.check("trajectory", "warn", "no traj.lammpstrj — structural checks skipped")

    if protocol == "melt":
        out.update(analyse_melt(run_dir, th, rep))
    if plot and th_all:
        make_md_plot(th_all, ensemble, temp_end, rdf_data, msd_data, str(run_dir / "md_summary.png"),
                     skip, stages, vd)
    return out


def analyse_diffusion(prod, msd_data, dt_frame, ensemble, T, viscosity, rep) -> dict:
    out = {"diffusion_cm2_s": diffusion(msd_data)}
    slopes = msd_loglog_slope(msd_data)
    axes = diffusion_axes(prod, dt_frame)
    out["msd_loglog_slope"], out["diffusion_axes_cm2_s"] = slopes, axes
    rows = []
    for el, d in out["diffusion_cm2_s"].items():
        mobile = msd_data[el][-1] > 1.0                              # Å²: moved more than ~1 Å
        a = slopes.get(el)
        regime = ("not diffusing" if not mobile else "diffusive" if a is not None and 0.85 <= a <= 1.15
                  else "sub-diffusive" if a is not None and a < 0.85 else "ballistic / too short")
        rows.append((el, f"{d:.3e}", " ".join(f"{x:.2e}" for x in axes.get(el, [])), num(a, 2), regime))
        if mobile:
            rep.check(f"diffusive regime {el}", "pass" if regime == "diffusive" else "warn",
                      f"log-log MSD slope {num(a, 2)}",
                      suggest="MSD is not linear in time over the fit window: D is not reliable — run longer, "
                              "fit a later window, or average independent runs (Maginn et al., LiveCoMS 2019)")
    if ensemble not in ("nve",) and any(r[4] == "diffusive" for r in rows):
        rep.check("transport ensemble", "warn", f"D measured under {ensemble}",
                  suggest="thermostats perturb dynamics: measure D in NVE (-protocol transport)")
    LOG(table("diffusion", ["element", "D (cm²/s)", "D_xx D_yy D_zz", "MSD slope (log-log)", "regime"], rows,
              styles=["key"]))
    if viscosity:
        L = float(np.cbrt(prod[0].get_volume()))
        out["diffusion_yeh_hummer_cm2_s"] = {el: yeh_hummer(d, T, L, viscosity) for el, d in out["diffusion_cm2_s"].items()}
        LOG("Yeh–Hummer (cubic box, η = " + f"{viscosity:g} mPa·s): " +
            " · ".join(f"{el} {d:.3e}" for el, d in out["diffusion_yeh_hummer_cm2_s"].items()) + " cm²/s")
    return out


def analyse_melt(run_dir: Path, th: dict, rep: Report) -> dict:
    """
    Coexistence: T of the NPH stage, and which slabs are crystalline. Each slab's mean number of ten
    Wolde solid bonds (first-shell neighbours with q6(i)·q6(j) > 0.5: ~10-12 in a hot crystal, ~1-3 in a
    liquid) is compared with the midpoint between the most and least ordered slabs at the start of the
    NPH stage, atom-weighted. Structure, not energy: a uniformly hot crystal is not mistaken for 50 %
    liquid, and the cut does not assume where the crystal sits after the box changed.
    """
    out = {}
    if th and "Temp" in th and len(th["Temp"]) >= 16:
        out["T_coexistence"] = block_average(th["Temp"][len(th["Temp"]) // 2:])
    if th and "Volume" in th and len(th["Volume"]) > 2:
        dv = float(th["Volume"][-1] / th["Volume"][0] - 1)
        out["volume_change_nph"] = dv
        rep.check("melt: volume", "pass" if abs(dv) < 0.15 else "fail", f"{dv * 100:+.1f} % over the NPH stage",
                  suggest="the box keeps growing: the liquid is boiling (-melt_temp above the boiling point "
                          "or the critical point) — lower -melt_temp")
    prof = parse_chunks(run_dir / "phase_profile.dat", ncols=3)        # Ncount, pe, solid bonds per slab
    if len(prof) < 2:
        rep.check("melt: phases", "warn", "no phase_profile.dat — cannot tell which phases survived")
        return out
    first, final = prof[0], prof[-1]
    filled = first[:, 0] > 0.5 * np.median(first[:, 0])               # empty slabs (vacuum, gas) out
    q = first[filled, 2]
    q_solid, q_liquid = float(np.percentile(q, 90)), float(np.percentile(q, 10))
    out.update({"bonds_crystal_start": q_solid, "bonds_liquid_start": q_liquid})
    if q_solid - q_liquid < 3.0:                              # solid bonds: crystal ~10, liquid ~2
        rep.check("melt: phases", "fail", f"solid bonds per atom {q_liquid:.1f}–{q_solid:.1f} across slabs at the start of NPH: one phase only",
                  suggest="no crystal/liquid pair at the start of coexistence: if the count is high everywhere the "
                          "molten half recrystallised (raise -melt_temp); if low everywhere the crystal melted "
                          "too — -temp is above the superheating limit (lower it)")
        return out
    cut = 0.5 * (q_solid + q_liquid)
    frac = lambda b: float(b[b[:, 2] < cut, 0].sum() / max(b[:, 0].sum(), 1e-12))
    liquid = frac(final)
    out["liquid_fraction_start"] = frac(first)
    out["liquid_fraction_end"] = liquid
    Tm = out.get("T_coexistence", {})
    both = 0.15 <= liquid <= 0.85
    rep.check("melt: phases", "pass" if both else "warn",
              f"liquid fraction {liquid:.0%} at the end (cut {cut:.1f} solid bonds) · T {Tm.get('mean', float('nan')):.0f} ± "
              f"{Tm.get('se', float('nan')):.0f} K",
              suggest="one phase took over, so the final T is not T_m: " +
                      ("all liquid → restart with a lower -temp" if liquid > 0.85 else
                       "all crystal → restart with a higher -temp"))
    if both and Tm:
        out["T_melt_estimate"] = Tm["mean"]
        LOG(f"[ok]coexistence held[/ok]: T_m ≈ {Tm['mean']:.0f} ± {Tm['se']:.0f} K "
            f"(liquid fraction {liquid:.0%}); check finite-size effects with a larger box")
    return out


def parse_chunks(path: Path, ncols: int = 1) -> list[np.ndarray]:
    """fix ave/chunk output: per block, the last `ncols` columns of every chunk row."""
    if not path.exists():
        return []
    blocks, cur, left = [], [], 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if left == 0:
            if cur:
                blocks.append(np.array(cur))
            cur, left = [], int(parts[1])
            continue
        cur.append([float(x) for x in parts[-ncols:]])
        left -= 1
    if cur and left == 0:
        blocks.append(np.array(cur))
    return blocks


def analyse_minimize(run_dir: Path, th: dict, params: dict, natoms: int, rep: Report) -> dict:
    from ..lmp.inputs import read_back
    from .traj import read_dump as _rd
    out = {}
    iters = len(re.findall(r"MFLOW relax iteration", (run_dir / "log.lammps").read_text(errors="replace")))
    out["relax_iterations"] = iters
    data = run_dir / "relaxed.data"
    if not data.exists():
        rep.check("minimize", "fail", "no relaxed.data written", suggest="see log.lammps")
        return out
    species = params.get("species") or []
    final = read_back(data, species)
    start = read_back(run_dir / "data.lmp", species)
    fmax = None
    if (run_dir / "relaxed.dump").exists():
        fr = _rd(run_dir / "relaxed.dump", 1, species)[-1]
        fmax = float(np.abs(fr.get_forces()).max())
        final.calc = None
    dv = (final.get_volume() - start.get_volume()) / start.get_volume() if all(final.pbc) else 0.0
    out.update({"fmax_eV_A": fmax, "volume_change": float(dv),
                "pe_eV_atom": float(th["PotEng"][-1] / natoms) if th else None,
                "press_bar": float(th["Press"][-1]) if th and "Press" in th else None})
    target = params.get("fmax", 0.01)
    rep.check("minimize: forces", "pass" if fmax is not None and fmax <= 1.5 * target else "warn",
              f"max |F| {num(fmax, 4)} eV/Å (target {target:g})",
              suggest="the minimiser stopped above the force target: raise -steps or check log.lammps")
    if all(final.pbc) and th and "Press" in th:
        p_ok = abs(th["Press"][-1] - params.get("press", 1.0)) < 500
        rep.check("minimize: pressure", "pass" if p_ok else "warn",
                  f"{th['Press'][-1]:.0f} bar after {iters} box/relax cycle(s) · volume {dv * 100:+.2f} %",
                  suggest="the cell is not relaxed to the target pressure: more box/relax cycles or -steps")
    ase.io.write(run_dir / "relaxed.xyz", final, format="extxyz")
    LOG(f"relaxed structure [key]{run_dir / 'relaxed.xyz'}[/key] · volume {dv * 100:+.2f} % · "
        f"{iters} box/relax cycle(s)")
    return out


def make_md_plot(th, ensemble, temp, rdf_data, msd_data, outfile, skip, stages=None, vd=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    t = th["Time"]
    multi = stages and len(stages) > 1
    t0 = t[int(len(t) * skip)] if len(t) and not multi else None
    panels = [("Temp", "Temperature (K)", temp if ensemble not in ("nve", "nph") else None),
              ("Press", "Pressure (bar)", None),
              ("TotEng" if ensemble == "nve" else "PotEng",
               "Total energy (eV)" if ensemble == "nve" else "Potential energy (eV)", None),
              ("Volume", r"Volume ($\mathrm{\AA}^3$)", None)]
    dt_ps = (t[-1] - t[0]) / max(th["Step"][-1] - th["Step"][0], 1) if len(t) > 1 else 0.0
    for ax, (col, label, target), color in zip(axes.ravel()[:4], panels, COLORS):
        if col in th:
            ax.plot(t, th[col], color=color, lw=1.2)
            if target is not None:
                ax.axhline(target, color="black", ls=LINESTYLES[1], lw=1.5)
            if t0 is not None:
                ax.axvspan(t[0], t0, color="grey", alpha=0.15, lw=0)
            for st in (stages or [])[1:] if multi else []:
                ax.axvline(t[0] + st["start"] * dt_ps, color="grey", lw=0.8, ls=LINESTYLES[1])
        ax.set(xlabel="Time (ps)", ylabel=label)
    ax = axes[1, 1]
    if rdf_data:
        for i, (k, g) in enumerate((k, v) for k, v in rdf_data.items() if k != "r"):
            ax.plot(rdf_data["r"], g, color=COLORS[i % len(COLORS)], label=k)
        ax.legend()
    ax.set(xlabel=r"r ($\mathrm{\AA}$)", ylabel="g(r)")
    ax = axes[1, 2]
    if vd is not None:
        ax.plot(vd["nu_THz"], vd["g"], color=COLORS[0])
        ax.set(xlabel="Frequency (THz)", ylabel="VDOS (unit area)", xlim=(0, min(vd["nyquist_THz"], 40)))
    else:
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
                               stride=args.stride, ref=args.ref, plot=not args.no_plot,
                               viscosity=getattr(args, "viscosity", None)))
    LOG.rule("verdict")
    rep.show()
    rep.write(run_dir / "md_report.json")
    return 0 if not rep.failed else 2
