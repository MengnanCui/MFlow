---
name: mflow-train
description: Train or fine-tune a MACE interatomic potential with MFlow. Use when the user wants to check a DFT dataset, fine-tune MACE-MPA-0 / OMAT / MH-1, train from scratch, or judge a finished training run. Covers `mflow check`, `mflow train`, `mflow train-report`.
---

# MACE 训练（MFlow）

## 固定顺序 —— 不要跳步

1. **体检**：`mflow check -in train.xyz -model mpa -split 0.1 -outdir check`
   - 读 `check/train_check.json`。`status: fail` → 停下，把 `checks` 里的 fail 项和 `suggestions` 报给用户，**不要训练**。
   - `-split 0.1` 会写出 `check/train_train.xyz` / `check/train_test.xyz`，后面用这两个文件。
   - `zero_shot.raw_offset_eV_per_atom` 绝对值 > 0.4：用户的 DFT 设置和基础模型不一致，向用户确认用的是哪套 DFT（MP PBE+U / OMat24 / MatPES）。
2. **先 dry run**：`mflow train -in check/train_train.xyz -test check/train_test.xyz -name <run> -dry`
   - 把 `<run>/config.yaml` 的关键项（mode、foundation_model、lr、max_num_epochs、E0s）给用户看，**等用户确认**。
3. **训练**：同一命令去掉 `-dry`。命令会自己监控、自己生成报告，等它结束即可。
4. **读报告**：`<run>/<run>_train_report.json`（见 `mflow-report-reading`）。

## 选模式

| 情况 | `-mode` | `-foundation` |
|---|---|---|
| 只关心一个体系，数据几十到几千帧 | `finetune`（默认） | `mpa`（MIT 许可）或 `mh1` |
| 还要保住在其他体系上的泛化能力 | `multihead` | `mh1` / `mpa`；需要 `-pt` 回放数据（默认 `mp` 会联网下载） |
| 数据很少（< 100 帧），想少改动 | `lora` | 同上 |
| 基础模型不包含你的元素 / 数据量大（> 5k 帧） | `scratch` | 不需要 |

## 规则

- 只能通过 `-set key=value` 改超参，键名必须是 `mace_run_train --help` 里有的（MFlow 会校验，不合法直接 fail）。
- 不要手改 `config.yaml` 后自己运行 `mace_run_train`；改参数就重新跑 `mflow train`。
- 中断后续跑：同一命令加 `-resume`。
- 一次只改一个超参，每次都用 `mflow train-report` 比较测试集误差；把结论写进 `LESSONS.md`（用户确认后）。
- 显存不够：先降 `-batch`；GPU 上默认 float32 + cuEquivariance。

## 常用

```bash
mflow train-report -dir <run> -group config_type     # 重新分析、按标签拆误差
mflow calc -in new.xyz -model <run>/<run>.model -ref REF_    # 在新数据上评估
mflow calc -in md.xyz -model a.model,b.model,c.model          # 委员会：<prefix>Fdev_max
```
