# Sources — mflow-md-setup

Everything in SKILL.md is paraphrased (often translated into Chinese) and adapted to MFlow's commands and
thresholds; no passages are copied. Numbers taken from a source are attributed below; numbers that are
MFlow's own (thresholds.yaml, tested in this repository) are not listed.

## Skill collections (read 2026-10-10 at the pinned commit)

| Source | What was used | License |
|---|---|---|
| [AtomisticSkills](https://github.com/learningmatter-mit/atomisticskills/tree/62574443f2a772e23dd5951f3712b65133b06715/skills) | mat-md-monitors (timestep by H/temperature, damping 100 dt, ramp from low T, stop conditions, recovery order) | MIT |
| [AtomisticSkills](https://github.com/learningmatter-mit/atomisticskills/tree/62574443f2a772e23dd5951f3712b65133b06715/skills) | mat-diffusion-analysis (≥ 10 Å cells, ≥ 20 mobile ions) | MIT |
| [CatMaster](https://github.com/q734738781/catmaster/tree/06b856814f980ca35f97fb01d7b970b5b18fd230) | skills/dynamics_worker/lammps-preparation (explicit type map, unwrapped coordinates), lammps-minimization/references/lammps_minimization_criteria.md (force targets, iteration cap ≠ converged), lammps-md-execution (finished ≠ equilibrated) | Apache-2.0 |
| [HeshamFS materials-simulation-skills](https://github.com/heshamfs/materials-simulation-skills/tree/fa1ce8de76b2ab7a856f81891525cfa227fadfe9/skills) | simulation-workflow/md-analysis-planner (checklist + pitfalls format) | Apache-2.0 |

## Literature and documentation

- LAMMPS documentation (fix nh, temp/csvr, langevin, box/relax, orientorder/atom, coord/atom), stable_22Jul2025 / stable_30Sep2026
- Basconi & Shirts, *J. Chem. Theory Comput.* 9, 2887 (2013) — thermostats and dynamics
- LAMBench — NVE drift tolerance 5e-4 eV/atom/ps

## Caveats

- AtomisticSkills' fine-tuning skill lists inconsistent defaults; its numbers are treated as rough heuristics.
- Committee-deviation thresholds are only comparable between tools that define the deviation the same way;
- Skill format follows agentskills.io/specification (name = directory, description ≤ 1024 characters).
