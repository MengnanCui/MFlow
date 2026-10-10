---
name: mflow-report-reading
description: How to read MFlow JSON reports (*_check.json, *_train_report.json, md_report.json, lmp_env.json, deploy_report.json) and decide the next step from their pass / warn / fail checks. Use whenever an mflow command has finished and you need to tell the user what happened or what to do next.
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
| `md_report.json` | `consistency`、`nve_drift_eV_atom_ps`、`T_mean`、`V_mean`、`dmin_min`、`diffusion_cm2_s`、`ns_per_day` |

## 精度分级（训练报告）

| grade | 能量 RMSE | 力 RMSE |
|---|---|---|
| excellent | < 5 meV/atom | < 20 meV/Å |
| good | < 10 meV/atom | < 50 meV/Å |
| needs work | 其他 | |

**注意**：力误差低不代表 MD 稳定 —— 训练报告好看之后，仍要跑 `mflow md` 看闸门。

## 总结模板

```
[命令] status=…
- 通过：…（一句话）
- 警告：<name>：<message> → <suggestion>
- 失败：<name>：<message> → <suggestion>
下一步：…
```
