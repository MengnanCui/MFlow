# Sources — mflow-active-learning-campaign

Everything in SKILL.md is paraphrased (often translated into Chinese) and adapted to MFlow's commands and
thresholds; no passages are copied. Numbers taken from a source are attributed below; numbers that are
MFlow's own (thresholds.yaml, tested in this repository) are not listed.

## Skill collections (read 2026-10-10 at the pinned commit)

| Source | What was used | License |
|---|---|---|
| [DP-GEN](https://github.com/deepmodeling/dpgen/tree/0b9acec837474d2a93fa010598472feb8f69f82a) | doc/run/overview-of-the-run-process.md, example-of-param.md, dpgen/generator/arginfo.py (trust levels, fp_accurate_threshold) | LGPL-3.0 (ideas only, no text copied) |
| [ai2-kit (TESLA)](https://github.com/chenggroup/ai2-kit/tree/507cd0f8db92587dd29270633b3c39d92fda0c3a) | skills/build-tesla/SKILL.md, doc/manual/model-deviation.md (MACE committee re-scoring of MD frames, per-trajectory caps) — deviation defined differently, numbers not transferred | MIT |
| [AtomisticSkills](https://github.com/learningmatter-mit/atomisticskills/tree/62574443f2a772e23dd5951f3712b65133b06715/skills) | ml-committee-uncertainty (≥ 3 members; rank by force disagreement) | MIT |
| [CatMaster](https://github.com/q734738781/catmaster/tree/06b856814f980ca35f97fb01d7b970b5b18fd230) | skills/ml_worker/active-learning-relabel-loop/SKILL.md (per-round book-keeping, never overwrite a round, rebuild the dataset, rankers are not calibrated acquisition) | Apache-2.0 |
| [DP-GEN](https://github.com/deepmodeling/dpgen/tree/0b9acec837474d2a93fa010598472feb8f69f82a) | monitoring.md (zero candidates: check thresholds, MD stability, dump frequency first) | LGPL-3.0 (ideas only, no text copied) |

## Literature and documentation

- Zeng et al., *Deep Potentials for Materials Science*, arXiv:2203.00393 — practical guide, Table I (trust windows, accurate-ratio rule, ~100 labels/iteration)
- Kulichenko et al., *Chem. Rev.* 124, 13681 (2024) — data generation and active learning
- Grasselli et al., *Digital Discovery* 4, 2654 (2025) — uncertainty, calibration, no formal stopping rule

## Caveats

- AtomisticSkills' fine-tuning skill lists inconsistent defaults; its numbers are treated as rough heuristics.
- Committee-deviation thresholds are only comparable between tools that define the deviation the same way;
  MFlow uses DP-GEN's definition (per atom sqrt(mean_m |F_m − ⟨F⟩|²), max over atoms).
