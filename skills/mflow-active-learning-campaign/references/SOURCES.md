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

- Wen et al., *Deep Potentials for Materials Science*, Mater. Futures 1, 022601 (2022), arXiv:2203.00393 — practical guide, Table I (trust windows, accurate-ratio rule, ~100 labels/iteration)
- Kulichenko et al., *Chem. Rev.* 124, 13681 (2024) — data generation and active learning
- Grasselli et al., *Digital Discovery* 4, 2654 (2025) — uncertainty, calibration, no formal stopping rule
- dpgen2 documentation and source (`report_adaptive_lower`, `report_trust_levels_max`), docs.deepmodeling.com/projects/dpgen2,
  read 2026-10-11 — adaptive-lower window (numb_candi 200, rate 0.01, level_f_hi 0.5, conv_tolerance 0.05, n_checked_steps 2)
- Schwalbe-Koda et al., *Nat. Commun.* 16, 4014 (2025), arXiv:2404.12367; github.com/dskoda/quests (BSD-3) — dH > 0 as an
  uncalibrated novelty test
- Varga-Umbrich et al., arXiv:2605.03964; Beck et al., *J. Chem. Phys.* 163, 234103 (2025), arXiv:2508.09907 — committees
  fine-tuned from one foundation model disagree too little (collapse)
- Zaverkin et al., *npj Comput. Mater.* 10, 83 (2024), arXiv:2312.01416 — uncertainty-biased MD
- Fu et al., arXiv:2210.07237 — stability, not force error, predicts simulation quality
- MFlow's own survey with the full reference list: docs/research-al-2026-10.md

## Caveats

- AtomisticSkills' fine-tuning skill lists inconsistent defaults; its numbers are treated as rough heuristics.
- Committee-deviation thresholds are only comparable between tools that define the deviation the same way;
  MFlow uses DP-GEN's definition (per atom sqrt(mean_m |F_m − ⟨F⟩|²), max over atoms).
