---
name: mflow-report-reading
description: How to read MFlow JSON reports (*_check.json, *_train_report.json, *_committee_report.json, md_report.json, lmp_env.json, deploy_report.json, validate *.json, al status.json / import_report.json / coverage.json / probes.json) and decide the next step from their pass / warn / fail checks. Use whenever an mflow command has finished and you need to tell the user what happened or what to do next.
---

# 读 MFlow 报告

每个报告都是同一结构：

```json
{"command": "...", "status": "pass|warn|fail",
 "checks": [{"name": "...", "status": "...", "message": "...", "value": ...}],
 "suggestions": ["..."], ...其余数据}
```

## 规则

1. **只看 JSON，不猜终端输出。** `status` 是所有 check 里最差的那个。
2. `fail`：停下。把 fail 的 `name` + `message` + 对应 `suggestions` 原样告诉用户，不要自行"修复"后重跑（除非用户已经同意）。
3. `warn`：继续可以，但要在总结里列出每条 warn 和建议。
4. 不要编造数字：误差、温度、漂移一律引用报告里的字段。

## 关键字段

| 报告 | 先看 |
|---|---|
| `*_check.json` | `flags`（坏帧的索引）、`zero_shot.errors`、`zero_shot.raw_offset_eV_per_atom`、`groups` |
| `*_train_report.json` | `curves.valid` / `curves.best_epoch`、`test.errors.energy.RMSE`（meV/atom）、`test.errors.forces.RMSE`（meV/Å）、`test.grade`、`test.by_group` |
| `deploy_report.json` / `deploy.json` | `head`、`dtype`、`elements`（= LAMMPS 原子类型顺序） |
| `lmp_env.json` | 每个 package / python 依赖是否通过 |
| `md_report.json` | `consistency`、`nve_drift_eV_atom_ps`、`block_averages`（均值 ± 分块误差）、`halves`、`dmin_min`、`diffusion_cm2_s` + `msd_loglog_slope`、`vdos_peak_THz`、`ns_per_day`；协议：`production_stage`、`T_melt_estimate` / `liquid_fraction_end`（melt）、`fmax_eV_A` / `volume_change`（minimize） |
| `*_committee_report.json` | `test.calibration.spearman_atom`、`sigma_ratio`（> 1 = 委员会过度自信；≥ 3 = `collapsed: true`，委员会塌缩）、`miscalibration_area`；`committee.json` 的 `members`（每个成员的 foundation 和额外选项） |
| `validate/*.json` | `dimer.pairs.<A-B>`（`repulsion_violations`、`force_flips`、`spearman_repulsion`）、`eos.B0_GPa` / `V0_A3_atom`、`phonon.min_THz` / `n_imaginary_qpoints`、`defect.vacancy` / `surface` |
| `<campaign>/status.json` | **`verdict`**：`converged` / `continue` / `budget_exhausted`（预算用完 ≠ 收敛，要明说"未收敛"）；`region` / `regions`（分区时哪个区通过了）；每个 check 的 `name` 以 L1 / L2 / budget 开头，`value` 为 `true` / `false` / `null`（null = n/a，数据不足，不阻塞）；`ledger_rows`。`round_NN/export/import_report.json` 里看被拒标注的原因 |
| `round_NN/select/coverage.json` | `ood_fraction`（描述子距离超阈值的比例）或 `dH_pos_fraction`（QUESTS，dH > 0 的比例）、`entropy_train` |
| `round_NN/probes/probes.json` | `values`（成员平均）、`comparison`（与上一轮：`previous` / `current` / `ok`）、`stable`、`spread_ok` / `spread_summary`（成员之间） |
| `ledger.jsonl` 的 `select` 记录 | `uncertainty`（committee / descriptor / quests / none）、`trust_mode`、`n_accurate/n_candidate/n_failed`、`regions`（按区分类）、`dev_median_wo_driver` |

## 精度分级（训练报告）

| grade | 能量 RMSE | 力 RMSE |
|---|---|---|
| excellent | < 5 meV/atom | < 20 meV/Å |
| good | < 10 meV/atom | < 50 meV/Å |
| needs work | 其他 | |

**注意**：力误差低不代表 MD 稳定 —— 训练报告好看之后，仍要跑 `mflow validate dimer` 和 `mflow md` 看闸门。
所有阈值在 `mflow/thresholds.yaml`（用户可用 `./mflow_thresholds.yaml` 覆盖）；引用阈值时说出处的键名，不要凭记忆。

## 总结模板

```
[命令] status=…
- 通过：…（一句话）
- 警告：<name>：<message> → <suggestion>
- 失败：<name>：<message> → <suggestion>
下一步：…
```
