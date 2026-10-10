# Sources — mflow-property-protocols

Everything in SKILL.md is paraphrased (often translated into Chinese) and adapted to MFlow's commands and
thresholds; no passages are copied. Numbers taken from a source are attributed below; numbers that are
MFlow's own (thresholds.yaml, tested in this repository) are not listed.

## Skill collections (read 2026-10-10 at the pinned commit)

| Source | What was used | License |
|---|---|---|
| [AtomisticSkills](https://github.com/learningmatter-mit/atomisticskills/tree/62574443f2a772e23dd5951f3712b65133b06715/skills) | mat-melting-point (coexistence, ≥ 20 Å, phase fractions, final-T averaging), mat-diffusion-analysis (4–6 temperatures, stop at 5–10 % error, Arrhenius), mat-amorphization (melt far above T_m, 1–10 K/ps, final checks), ml-committee-uncertainty (late-time UQ on long runs) | MIT |
| [HeshamFS materials-simulation-skills](https://github.com/heshamfs/materials-simulation-skills/tree/fa1ce8de76b2ab7a856f81891525cfa227fadfe9/skills) | simulation-workflow/md-analysis-planner (diffusive-regime check, block averaging, half-vs-half equilibration) | Apache-2.0 |

## Literature and documentation

- Yeh & Hummer, *J. Phys. Chem. B* 108, 15873 (2004); Maginn et al., *LiveCoMS* 1, 6324 (2019) — finite-size correction, MSD best practice
- Flyvbjerg & Petersen, *J. Chem. Phys.* 91, 461 (1989) — block averaging
- ten Wolde, Ruiz-Montero, Frenkel, *J. Chem. Phys.* 104, 9932 (1996) — solid-bond order parameter
- LAMMPS documentation (fix nh, temp/csvr, langevin, box/relax, orientorder/atom, coord/atom), stable_22Jul2025 / stable_30Sep2026

## Caveats

- AtomisticSkills' fine-tuning skill lists inconsistent defaults; its numbers are treated as rough heuristics.
- Committee-deviation thresholds are only comparable between tools that define the deviation the same way;
- Skill format follows agentskills.io/specification (name = directory, description ≤ 1024 characters).
