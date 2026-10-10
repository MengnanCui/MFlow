---
name: mflow-finetune-strategy
description: Choose how to fine-tune or train a MACE potential with MFlow — which foundation model and head, naive fine-tuning vs multihead replay vs LoRA vs from scratch, E0 handling, learning rate and epochs, committee and bagging, and how to recognise catastrophic forgetting. Use before `mflow train` whenever the user has DFT data and asks which training scheme to use, or when a trained model is accurate on its test set but fails elsewhere.
---

# 微调方案怎么选（经验）

执行在 `mflow-train`；这里决定 `-mode`、`-foundation`、`-set` 的取值。默认值见 `mflow/train/config.py` 的 `MODE_DEFAULTS`。

## 1. 先看两件事（`mflow check` 报告）

1. **DFT 设置是否和基础模型一致**：`zero_shot.raw_offset_eV_per_atom` 绝对值 > 0.4（`thresholds.yaml` 的 `check.e0_offset_warn_eV_atom`）说明不一致。
   - MPA-0 / MP 系列：MP 的 PBE(+U)；OMAT / MH-1 的 `omat_pbe` head：OMat24 的 PBE(+U)；MatPES 系列：PBE 或 r2SCAN。
   - 不一致时：要么换对应的基础模型/head，要么接受系统偏移并用 E0 吸收（`E0s` 不能用 average）。
2. **数据量与覆盖**：帧数、按 `config_type` 的分组、元素是否都在基础模型里。

## 2. 选模式

| 情况 | `-mode` | 说明 |
|---|---|---|
| 单一体系，几十到几千帧，只在这个体系上用 | `finetune` | 收敛最快；会遗忘其他化学空间 |
| 还要在训练集之外的构型 / 成分上稳定（MD 会走远、表面、缺陷、高温） | `multihead` | 唯一能稳定保住分布外鲁棒性和短程排斥的方案（Tompa et al. 2026）；回放数据可用伪标签 |
| 数据很少（< 100 帧） | `lora` | 改动小，对 MACE 比冻结层好；rank 从 4 起，数据更少用 2，欠拟合用 8（MFlow 默认 8，`-set lora_rank=4`） |
| 基础模型不含你的元素，或 > 5000 帧且设置完全不同 | `scratch` | 多数情况下比微调差，先试微调 |

**不要**冻结嵌入层和第一层交互层做微调——在 MACE 上效果最差。

## 3. 数字

| 参数 | finetune | multihead | lora | 依据 |
|---|---|---|---|---|
| lr | 1e-3；**OMAT / MH 系列用 1e-4**（1e-3 以上会遗忘，冻结主干也一样） | 1e-4 | 1e-2 | MACE 文档、MFlow 默认、AtomisticSkills |
| EMA | 0.999 | 0.9999（也有人用 0.99999） | 0.99 | |
| 梯度裁剪 | 1 | 1 | 10 | |
| epoch | 默认上限 15–25，保留验证集最佳；不要靠加 epoch 救 | 10–20 | 30–50 | 看 `train-report` 的学习曲线 |
| 回放样本 `num_samples_pt` | — | 1e4（MFlow 默认）到 5e4–1e5，FPS 选 | — | 正式运行可加大 |
| forces_weight / energy_weight | 10 / 1 | 10 / 1 | 10 / 1 | 力占主导，能量仍有约束 |

- **E0 比策略更重要**：`isolated`（数据里有 `IsolatedAtom`）> `estimated` > `average`。average 会让力误差变成 2–3 倍，多头微调**不能**用 average。
  - 和基础模型同一泛函时，可以直接用基础模型的 E0。
  - 正式比较、主动学习：把 E0 固定成 json，所有轮次用同一套。
  - `estimated` 是最小二乘解；成分配比固定时解不唯一，单个元素的 E0 没有物理意义。
  - 初始能量误差极大，先怀疑 E0。
- 早停：MACE 默认 patience 很大（相当于关闭），要早停就 `-set patience=100`。
- 改超参只用 `-set key=value`，一次改一个，`train-report` 比较验证/测试误差。
- 系统搜索：从保守默认值出发，约 10 次试验 × 约 10 个 epoch，每次只改一项；最好的配置再长训练。数据 < 1000 帧时每次试验用全部数据。

## 4. 委员会

- 要不确定度（主动学习、判断外推）就用 `-committee 4`。
- 微调时保持 bagging（默认 0.8）：同一基础模型出发，只换 seed 的成员太像，σ 偏小。
- 看 `*_committee_report.json`：Spearman ≥ 0.5 才拿 σ 做筛选；`sigma_ratio` > 1 表示过度自信（AL 会自动放大 σ）。

## 5. 遗忘的征兆

- 测试集误差很好，但 `mflow validate dimer` 的短程墙坏了（基础模型是好的）；
- 训练集外的结构（表面、缺陷、高温液体）上误差比基础模型还差：先在留出集上测**基础模型**，再在同一集上测微调模型（`mflow calc`），两者对比；
- MD 在训练覆盖之外很快 `closest pair` fail。

出现任一条：换 `multihead`，或补对应构型。

## 6. 验证清单

- [ ] `check` 无 fail；DFT 设置和基础模型一致（或已知偏移）
- [ ] 测试集固定，且按来源轨迹切分（见 `mflow-dataset-curation`）
- [ ] `-dry` 看过 config.yaml，用户确认
- [ ] `train-report`：grade、overfit、按组误差
- [ ] `validate dimer` 和一次短 MD（`mflow-potential-acceptance`）

出处与改写说明见 `references/SOURCES.md`。
