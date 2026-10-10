# -*- coding: utf-8 -*-
"""Command line: every sub-command is registered here."""

from __future__ import annotations

import argparse

from . import __version__
from .calc import cmd_calc, cmd_plot
from .log import LOG
from .model import DEFAULT_MODEL
from .relax import OPTIMIZERS, cmd_relax
from .train.config import MODES

EXAMPLES = """examples
  mflow calc -in data.xyz                           # default model, batch 32
  mflow calc -in data.xyz -model mpa -batch 64      # download a foundation model
  mflow calc -in data.xyz -model ./my.model -prefix mace_
  mflow calc -in data.xyz -ref dft_                 # compare with dft_energy/dft_forces
  mflow calc -in data.xyz -group config_type        # errors split by label
  mflow calc -in md.xyz -model a.model,b.model,c.model
  mflow plot -in data_mace.xyz -group formula       # re-plot, no recompute

  mflow relax -in data.xyz                          # relax positions + cell, fmax 0.05
  mflow relax -in data.xyz -cell none -fmax 0.02    # fixed cell, tighter
  mflow relax -in data.xyz -nproc 3 -dtype float32  # 3 workers sharing one gpu
  mflow relax -in data.xyz -resume                  # continue an interrupted run

  mflow check -in train.xyz -model mpa -split 0.1   # health + zero-shot + split
  mflow train -in train.xyz -test test.xyz -name cu # fine-tune MACE-MPA-0, watch it, report
  mflow train -in train.xyz -mode multihead -foundation mh1 -name cu -dry
  mflow train-report -dir cu                        # re-analyse a finished run

  mflow deploy -model cu/cu.model -head Default     # -> deploy/deploy.json (+ ML-IAP .pt)
  mflow lmp-check -deploy deploy                    # can this LAMMPS run it?
  mflow md -in start.xyz -deploy deploy -temp 600 -steps 50000 -outdir md600
  mflow md -in start.xyz -deploy deploy -ensemble npt -press 1 -outdir npt -dry
  mflow md-report -dir md600                        # gates, RDF, MSD, figure

  mflow validate dimer -model cu.model -elements Cu  # short-range wall, smoothness
  mflow validate eos -model cu.model -in cu.xyz -ref "V0=11.8,B0=140"
  mflow validate phonon -model cu.model -in cu.xyz -supercell 4
  mflow validate defect -model cu.model -in cu_conv.xyz -surface 111,100"""


AL_EXAMPLES = """examples
  mflow al init -name cu_al -system start.xyz -data dft.xyz -test test.xyz -foundation mpa
  mflow al run -name cu_al                    # round 0: explore with the foundation model, FPS, export
  #   ... label round_00/export/candidates.extxyz with TFlow (commands in round_00/export/README.txt) ...
  mflow al import -name cu_al -tflow /path/to/calc_dir
  mflow al run -name cu_al                    # train committee → explore → select → export (next round)
  mflow al status -name cu_al                 # ledger + stopping criteria"""


def _lazy(module: str, name: str):
    """The command is imported only when it runs, so `mflow -h` never pays for torch."""
    def run(args):
        import importlib
        return getattr(importlib.import_module(module), name)(args)
    return run


def build_parser() -> argparse.ArgumentParser:
    """Options are multi-letter on purpose: -in, -out, -model ... no single-letter clashes."""
    parser = argparse.ArgumentParser(prog="mflow", epilog=EXAMPLES,
                                     description=f"MFlow {__version__} — MACE training, evaluation and LAMMPS MD.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-version", action="version", version=f"MFlow {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-in", "--input", dest="input", required=True, metavar="FILE", help="input xyz/extxyz")
    common.add_argument("-index", default=":", help="ase slice, e.g. ':100' [:]")
    common.add_argument("-group", default=None, metavar="KEY",
                        help="split the results by an existing label: any info key "
                             "(config_type, step, ...) or the virtual keys formula / natoms. "
                             "A numeric key is cut into quartiles")
    common.add_argument("-outdir", default=".", help="output directory [.]")
    common.add_argument("-log", default="py.log", metavar="FILE", help="log file [py.log]")

    compare = argparse.ArgumentParser(add_help=False)
    compare.add_argument("-ref", dest="ref_prefix", default=None, metavar="PREFIX",
                         help="reference label prefix, e.g. dft_ [auto-detect]")
    compare.add_argument("-e0", default="fit", metavar="MODE",
                         help="how to align the atomic reference before comparing energies: "
                              "fit (per-element least squares) | mean (one shared shift) | "
                              "none | a json file {'Si': -0.12, ...} [fit]")

    mace_args = argparse.ArgumentParser(add_help=False)
    mace_args.add_argument("-model", default=DEFAULT_MODEL, metavar="PATH|NAME",
                           help="path to a .model file, or a foundation name to download "
                                "(small/medium/large/medium-mpa-0/medium-omat-0, aliases mpa/omat)\n"
                                f"[{DEFAULT_MODEL}]")
    mace_args.add_argument("-prefix", default="mpa0_", help="output label prefix [mpa0_]")
    mace_args.add_argument("-out", dest="output", default=None, metavar="FILE", help="output xyz")
    mace_args.add_argument("-device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="[auto]")
    mace_args.add_argument("-dtype", default="float64", choices=["float64", "float32"], help="[float64]")
    mace_args.add_argument("-head", default=None, help="head of a multi-head model")
    mace_args.add_argument("-overwrite", action="store_true",
                           help="reuse the prefix even when it already exists (default: fall back "
                                "to <prefix>c1_, <prefix>c2_, ... so nothing is overwritten)")
    mace_args.add_argument("-noplot", dest="no_plot", action="store_true", help="skip the figure")

    calc = sub.add_parser("calc", parents=[common, compare, mace_args],
                          help="run MACE on an xyz dataset",
                          epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    calc.add_argument("-batch", dest="batch_size", type=int, default=32, help="batch size [32]")
    calc.set_defaults(func=cmd_calc)

    relax = sub.add_parser("relax", parents=[common, mace_args],
                           help="optimise every structure with an ASE optimiser",
                           epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    relax.add_argument("-fmax", type=float, default=0.05,
                       help="convergence on the largest atomic force, eV/A [0.05]")
    relax.add_argument("-steps", type=int, default=500, help="max optimiser steps per structure [500]")
    relax.add_argument("-opt", default="fire", choices=list(OPTIMIZERS), help="ase optimiser [fire]")
    relax.add_argument("-cell", default="full", choices=["full", "none"],
                       help="full = relax the cell too (ase FrechetCellFilter), none = positions "
                            "only [full]. Non-periodic structures always fall back to positions only")
    relax.add_argument("-nproc", default="auto", metavar="N",
                       help="worker processes, each holding its own model "
                            "[auto: 1 on gpu, cores/4 on cpu]")
    relax.add_argument("-traj", default=None, metavar="FILE",
                       help="also write every intermediate frame to this file [off]")
    relax.add_argument("-resume", action="store_true",
                       help="skip structures already finished in <stem>_relax.part.xyz")
    relax.set_defaults(func=cmd_relax)

    check = sub.add_parser("check", parents=[common], help="dataset health check before training",
                           epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    check.add_argument("-test", default=None, metavar="FILE", help="test set, checked for labels and overlap")
    check.add_argument("-ref", dest="ref_prefix", default=None, metavar="PREFIX",
                       help="reference label prefix, e.g. REF_ [auto-detect]")
    check.add_argument("-model", default=None, metavar="PATH|NAME",
                       help="foundation model for a zero-shot pass (element coverage, error, E0 offset) [off]")
    check.add_argument("-head", default=None, help="head of a multi-head model")
    check.add_argument("-split", type=float, default=None, metavar="FRAC",
                       help="write <stem>_train.xyz / <stem>_test.xyz, stratified by group [off]")
    check.add_argument("-seed", type=int, default=0, help="split seed [0]")
    check.add_argument("-device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="[auto]")
    check.add_argument("-dtype", default="float64", choices=["float64", "float32"], help="[float64]")
    check.add_argument("-noplot", dest="no_plot", action="store_true", help="skip the figure")
    check.set_defaults(func=_lazy("mflow.train.check", "cmd_check"))

    train = sub.add_parser("train", help="check data, write config.yaml, run mace_run_train and watch it",
                           epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    train.add_argument("-in", "--input", dest="input", required=True, metavar="FILE", help="training xyz")
    train.add_argument("-index", default=":", help="ase slice of the training file [:]")
    train.add_argument("-name", required=True, help="run name (MACE file names, default run directory)")
    train.add_argument("-outdir", default=None, help="run directory [./<name>]")
    train.add_argument("-log", default="py.log", metavar="FILE", help="log file [py.log]")
    train.add_argument("-test", default=None, metavar="FILE", help="test set, evaluated at the end")
    train.add_argument("-valid", default=None, metavar="FILE", help="validation set [10%% of train]")
    train.add_argument("-ref", dest="ref_prefix", default=None, metavar="PREFIX", help="label prefix [auto]")
    train.add_argument("-group", default=None, metavar="KEY", help="split the test errors by this label")
    train.add_argument("-mode", default="finetune", choices=MODES, help="[finetune]")
    train.add_argument("-foundation", default="mpa", metavar="NAME|PATH",
                       help="model to start from: mpa / omat / mh1 / matpes / a .model file [mpa]")
    train.add_argument("-head", default=None, help="foundation head to start from (mh-1: omat_pbe)")
    train.add_argument("-pt", default=None, metavar="FILE|mp",
                       help="replay data for -mode multihead [mp = Materials Project, downloaded]")
    train.add_argument("-epochs", type=int, default=None, help="max epochs [mode default]")
    train.add_argument("-batch", type=int, default=8, help="batch size [8]")
    train.add_argument("-dtype", default="float32", choices=["float64", "float32"], help="[float32]")
    train.add_argument("-device", default="auto", choices=["auto", "cpu", "cuda"], help="[auto]")
    train.add_argument("-cueq", default="auto", choices=["auto", "on", "off"],
                       help="cuEquivariance kernels [auto: on with cuda when installed]")
    train.add_argument("-seed", type=int, default=123, help="[123]; committee members use seed, seed+1, ...")
    train.add_argument("-committee", type=int, default=1, metavar="N",
                       help="train N models with different seeds into <outdir>/m0..m{N-1} and measure how well "
                            "their spread tracks the test error [1]")
    train.add_argument("-bagging", type=float, default=None, metavar="FRAC",
                       help="committee members train on different random subsets of this size "
                            "[0.8 for fine-tuning, 1.0 for scratch]")
    train.add_argument("-set", nargs="*", default=None, metavar="KEY=VALUE",
                       help="extra mace_run_train options, checked against mace_run_train --help")
    train.add_argument("-dry", action="store_true", help="write config.yaml only, do not train")
    train.add_argument("-resume", action="store_true", help="continue from the latest checkpoint")
    train.add_argument("-force", action="store_true", help="train even if the pre-flight check fails")
    train.add_argument("-noplot", dest="no_plot", action="store_true", help="skip the figures")
    train.set_defaults(func=_lazy("mflow.train.run", "cmd_train"))

    report = sub.add_parser("train-report", help="learning curves, test-set accuracy and a verdict")
    report.add_argument("-dir", required=True, help="run directory written by `mflow train`")
    report.add_argument("-test", default=None, metavar="FILE", help="test set [the one in config.yaml]")
    report.add_argument("-group", default=None, metavar="KEY", help="split the test errors by this label")
    report.add_argument("-device", default="auto", choices=["auto", "cpu", "cuda"], help="[auto]")
    report.add_argument("-log", default="py.log", metavar="FILE", help="log file [py.log]")
    report.add_argument("-noplot", dest="no_plot", action="store_true", help="skip the figures")
    report.set_defaults(func=_lazy("mflow.train.report", "cmd_train_report"))

    deploy = sub.add_parser("deploy", help="export a model for LAMMPS (ML-IAP) and write deploy.json")
    deploy.add_argument("-model", required=True, metavar="PATH", help=".model file")
    deploy.add_argument("-head", default=None, help="head to export (required for multi-head models)")
    deploy.add_argument("-dtype", default="float32", choices=["float32", "float64"],
                        help="precision of the exported model [float32; float64 for checks]")
    deploy.add_argument("-outdir", default="deploy", help="[deploy]")
    deploy.add_argument("-log", default="py.log", metavar="FILE", help="log file [py.log]")
    deploy.set_defaults(func=_lazy("mflow.lmp.deploy", "cmd_deploy"))

    lmpcheck = sub.add_parser("lmp-check", help="check that LAMMPS can run MACE through ML-IAP + KOKKOS")
    lmpcheck.add_argument("-lmp", default=None, metavar="PATH", help="lmp executable [$MFLOW_LMP, PATH]")
    lmpcheck.add_argument("-deploy", default=None, metavar="DIR|JSON", help="also run the exported model")
    lmpcheck.add_argument("-outdir", default="lmp_check", help="[lmp_check]")
    lmpcheck.add_argument("-log", default="py.log", metavar="FILE", help="log file [py.log]")
    lmpcheck.set_defaults(func=_lazy("mflow.lmp.probe", "cmd_lmp_check"))

    md = sub.add_parser("md", help="molecular dynamics in LAMMPS with a deployed MACE model",
                        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    md.add_argument("-in", "--input", dest="input", required=True, metavar="FILE", help="start structure")
    md.add_argument("-index", default="-1", help="which structure of the file [-1]")
    md.add_argument("-deploy", required=True, metavar="DIR|JSON", help="written by `mflow deploy`")
    md.add_argument("-ensemble", default="nvt", choices=["nvt", "npt", "nve"], help="[nvt]")
    md.add_argument("-temp", type=float, default=300.0, help="K; start temperature for nve [300]")
    md.add_argument("-temp_end", type=float, default=None, help="K, linear ramp to this [= -temp]")
    md.add_argument("-press", type=float, default=1.0, help="bar, npt only [1]")
    md.add_argument("-dt", type=float, default=1.0, help="timestep, fs [1]")
    md.add_argument("-steps", type=int, default=100_000, help="[100000]")
    md.add_argument("-tdamp", type=float, default=None, help="thermostat damping, fs [100 dt]")
    md.add_argument("-pdamp", type=float, default=None, help="barostat damping, fs [1000 dt]")
    md.add_argument("-seed", type=int, default=42, help="velocity seed [42]")
    md.add_argument("-dump", type=int, default=100, help="dump every N steps [100]")
    md.add_argument("-thermo", type=int, default=100, help="thermo every N steps [100]")
    md.add_argument("-restart", type=int, default=10_000, help="restart file every N steps [10000]")
    md.add_argument("-ngpu", type=int, default=1, help="GPUs = MPI ranks (Kokkos domain decomposition) [1]")
    md.add_argument("-lmp", default=None, metavar="PATH", help="lmp executable [$MFLOW_LMP, PATH]")
    md.add_argument("-outdir", default="md", help="run directory [md]")
    md.add_argument("-skip", type=float, default=0.2, help="equilibration fraction left out of the gates [0.2]")
    md.add_argument("-log", default="py.log", metavar="FILE", help="log file [py.log]")
    md.add_argument("-nocheck", action="store_true", help="skip the run-0 consistency check")
    md.add_argument("-dry", action="store_true", help="write the inputs only")
    md.add_argument("-resume", action="store_true", help="continue from the newest restart file")
    md.add_argument("-force", action="store_true", help="run even if a check failed")
    md.add_argument("-noplot", dest="no_plot", action="store_true", help="skip the figure")
    md.set_defaults(func=_lazy("mflow.lmp.run", "cmd_md"))

    mdrep = sub.add_parser("md-report", help="quality gates, thermodynamics, RDF and MSD of an MD run")
    mdrep.add_argument("-dir", required=True, help="run directory (mflow md, or any dir with log.lammps)")
    mdrep.add_argument("-ensemble", default=None, choices=["nvt", "npt", "nve"], help="[from md.json]")
    mdrep.add_argument("-temp", type=float, default=None, help="target K [from md.json]")
    mdrep.add_argument("-skip", type=float, default=0.2, help="equilibration fraction [0.2]")
    mdrep.add_argument("-stride", type=int, default=1, help="read every N-th dump frame [1]")
    mdrep.add_argument("-ref", default=None, metavar="FILE", help="reference trajectory (AIMD) for the RDF")
    mdrep.add_argument("-log", default="py.log", metavar="FILE", help="log file [py.log]")
    mdrep.add_argument("-noplot", dest="no_plot", action="store_true", help="skip the figure")
    mdrep.set_defaults(func=_lazy("mflow.analysis.report", "cmd_md_report"))

    val = sub.add_parser("validate", help="static properties: dimer curves, EOS, phonons, defects")
    val.add_argument("kind", choices=["dimer", "eos", "phonon", "defect"])
    val.add_argument("-model", default=DEFAULT_MODEL, help=f"model file or foundation name; emt for a dry try [{DEFAULT_MODEL}]")
    val.add_argument("-head", default=None, help="head of a multi-head model")
    val.add_argument("-dtype", default="float64", choices=["float32", "float64"], help="[float64]")
    val.add_argument("-in", "--input", dest="input", default=None, metavar="FILE",
                     help="structure (relaxed crystal for eos / phonon / defect; elements for dimer)")
    val.add_argument("-index", default="-1", help="frame of -in [-1]")
    val.add_argument("-elements", default=None, help="dimer: elements, e.g. Cu,Al [from -in]")
    val.add_argument("-rmin", type=float, default=0.9, help="dimer: first distance, × covalent radius [0.9]")
    val.add_argument("-rmax", type=float, default=None, help="dimer: last distance in Å [3.1 × vdW radius]")
    val.add_argument("-dr", type=float, default=0.01, help="dimer: spacing in Å [0.01]")
    val.add_argument("-npts", type=int, default=11, help="eos: volumes [11]")
    val.add_argument("-strain", type=float, default=0.04, help="eos: ± linear strain [0.04]")
    val.add_argument("-relax", action=argparse.BooleanOptionalAction, default=True,
                     help="relax positions (fixed cell): eos at each volume, phonon before displacing [on]")
    val.add_argument("-supercell", default=None, help="phonon: e.g. 4 or 4,4,2 [widths ≥ 10 Å]")
    val.add_argument("-mesh", type=int, default=16, help="phonon: q-mesh per direction [16]")
    val.add_argument("-displacement", type=float, default=0.01, help="phonon: Å [0.01]")
    val.add_argument("-min_width", type=float, default=10.0, help="defect: supercell width in Å [10]")
    val.add_argument("-surface", default=None, help="defect: Miller indices of the given cell, e.g. 111,100")
    val.add_argument("-layers", type=int, default=6, help="defect: slab layers [6]")
    val.add_argument("-ref", default=None, help='DFT numbers to compare, e.g. "V0=11.8,B0=140,Ev_Cu=1.28,gamma_111=1.3"')
    val.add_argument("-outdir", default="validate", help="[validate]")
    val.add_argument("-noplot", dest="no_plot", action="store_true", help="skip the figure")
    val.add_argument("-log", default="py.log", metavar="FILE", help="log file [py.log]")
    val.set_defaults(func=_lazy("mflow.validate", "cmd_validate"))

    al = sub.add_parser("al", help="active learning: explore → select → TFlow → import → train",
                        epilog=AL_EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    alsub = al.add_subparsers(dest="al_command", required=True)
    al_name = argparse.ArgumentParser(add_help=False)
    al_name.add_argument("-name", required=True, help="campaign directory")
    al_name.add_argument("-log", default="py.log", metavar="FILE", help="log file [py.log]")
    al_round = argparse.ArgumentParser(add_help=False)
    al_round.add_argument("-round", type=int, default=None, help="[current round from state.json]")

    p_init = alsub.add_parser("init", parents=[al_name], help="create a campaign")
    p_init.add_argument("-system", required=True, metavar="FILE", help="start structure(s) for exploration")
    p_init.add_argument("-data", default=None, metavar="FILE", help="labelled data you already have")
    p_init.add_argument("-test", default=None, metavar="FILE", help="fixed test set")
    p_init.add_argument("-foundation", default="mpa", help="[mpa]")
    p_init.add_argument("-mode", default="finetune", choices=MODES, help="[finetune]")
    p_init.add_argument("-committee", type=int, default=4, help="[4]")
    p_init.add_argument("-engine", default="ase", choices=["ase", "lammps"], help="MD explorer engine [ase]")
    p_init.add_argument("-force", action="store_true", help="overwrite campaign.yaml")
    p_init.set_defaults(func=_lazy("mflow.al.campaign", "cmd_al_init"))
    for name, fn, hlp in (("explore", "cmd_al_explore", "run the explorers of campaign.yaml"),
                          ("select", "cmd_al_select", "committee scoring + trust window + FPS"),
                          ("export", "cmd_al_export", "write candidates + TFlow run script"),
                          ("train", "cmd_al_train", "train the committee on the current dataset")):
        sp = alsub.add_parser(name, parents=[al_name, al_round], help=hlp)
        if name == "explore":
            sp.add_argument("-lmp", default=None, metavar="PATH", help="lmp executable (engine lammps)")
        sp.set_defaults(func=_lazy("mflow.al.commands", fn))
    p_imp = alsub.add_parser("import", parents=[al_name, al_round], help="read DFT labels back")
    p_imp.add_argument("-tflow", default=None, metavar="CALC_DIR", help="TFlow calc dir (job_map.json)")
    p_imp.add_argument("-in", dest="labelled", default=None, metavar="FILE",
                       help="or: labelled extxyz carrying mflow_id")
    p_imp.set_defaults(func=_lazy("mflow.al.commands", "cmd_al_import"))
    p_st = alsub.add_parser("status", parents=[al_name], help="ledger + stopping criteria")
    p_st.set_defaults(func=_lazy("mflow.al.commands", "cmd_al_status"))
    p_run = alsub.add_parser("run", parents=[al_name], help="advance to the next DFT hand-off")
    p_run.add_argument("-lmp", default=None, metavar="PATH", help="lmp executable (engine lammps)")
    p_run.add_argument("-force", action="store_true", help="keep going even if the stopping criteria are met")
    p_run.set_defaults(func=_lazy("mflow.al.commands", "cmd_al_run"))

    plot = sub.add_parser("plot", parents=[common, compare], help="re-analyse an evaluated file")
    plot.add_argument("-pred", dest="pred_prefix", default="mpa0_", metavar="PREFIX",
                      help="prediction prefix [mpa0_]")
    plot.set_defaults(func=cmd_plot)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    LOG.open(args.log)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        LOG("[err]interrupted[/err]")
        return 130
    except Exception:
        LOG.term.print_exception(show_locals=False)
        if LOG.file:
            import traceback
            LOG.file.print(traceback.format_exc())
        return 1
