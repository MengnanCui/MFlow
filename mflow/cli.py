# -*- coding: utf-8 -*-
"""Command line: every sub-command is registered here."""

from __future__ import annotations

import argparse

from . import __version__
from .calc import cmd_calc, cmd_plot
from .log import LOG
from .model import DEFAULT_MODEL
from .relax import OPTIMIZERS, cmd_relax

EXAMPLES = """examples
  mflow calc -in data.xyz                                  # default model, batch 32
  mflow calc -in data.xyz -model mpa -batch 64             # download a foundation model
  mflow calc -in data.xyz -model ./my.model -prefix mace_
  mflow calc -in data.xyz -ref dft_                        # compare with dft_energy/dft_forces
  mflow calc -in data.xyz -group config_type               # errors split by label
  mflow plot -in data_mace.xyz -group formula              # re-plot, no recompute

  mflow relax -in data.xyz                                  # relax positions + cell, fmax 0.05
  mflow relax -in data.xyz -cell none -fmax 0.02  # fixed cell, tighter
  mflow relax -in data.xyz -nproc 3 -dtype float32              # 3 workers sharing one gpu
  mflow relax -in data.xyz -resume                          # continue an interrupted run"""


def build_parser() -> argparse.ArgumentParser:
    """Options are multi-letter on purpose: -in, -out, -model ... no single-letter clashes."""
    parser = argparse.ArgumentParser(prog="mflow", epilog=EXAMPLES,
                                     description=f"MFlow {__version__} — MACE energies and forces.",
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
