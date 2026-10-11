---
name: mflow-active-learning-campaign
description: Plan and run an active-learning campaign with `mflow al` and TFlow — choosing explorers (MD grid and regions, perturbations, minima hopping, heat-quench, random structure search, uncertainty-biased MD, PLUMED), committee or single model, uncertainty source and trust window (fixed, adaptive-lower), batch size, DFT hand-off and import rules, reading the ledger and the L1/L2/budget convergence table, and deciding when to stop. Use when a model fails outside its training data, when the user wants to build a potential for a new system with minimal DFT, or when an existing campaign needs its next step.
---

# 主动学习活动（经验）

命令与机制在 README 第四章；这里是选择与判断。阈值在 `thresholds.yaml` 的 `al` 段。

## 1. 开始之前

- 有没有固定的测试集？没有就先做（`mflow-dataset-curation`），之后不变。
- DFT 设置定下来，写进 `round_00/export/run_label.py` 并**整个活动不改**；与基础模型一致（`mflow-finetune-strategy`）。
- 估算预算：每轮 `max_per_round`（默认 150）个单点 × 预计轮数（常见 4–8 轮）。和用户确认。
- 起始结构（`-system`）覆盖要用的相：晶体、液体、表面、缺陷，各给一个代表。

## 2. 探索器

| 体系 | 推荐 | campaign.yaml |
|---|---|---|
| 体相晶体、液体 | `md` 温度网格 + `perturb` | `explore.md.temps` 覆盖到目标温度以上 20–30 %；`press` 需要时加 |
| 团簇、表面、找低能结构 | 加 `minhop` | `explore.minhop.enabled: true` |
| 高压、缺陷 | `perturb` 的 `strain` 加大、`vacancies: 1` | |
| 生产 MD 的条件已知 | `md` 的 `engine: lammps`，直接用生产设置 | 探索和生产同一引擎 |
| 宽温度范围 | DP-GEN 式分区 | `explore.md.regions`：低温区先过 L1 再进高温区；`grow` 让 MD 每轮变长 |
| 非晶、玻璃、急冷结构 | `heatquench` | `T_high` 高于熔点，`steps_quench` 决定冷却速率 |
| 未知晶体结构、相 | `rss`（随机结构搜索，只对 3D 周期体系） | `n`、`vol_jitter`、`min_dist` |
| 反应、相变（有集体变量） | `plumed`（LAMMPS `fix plumed`） | 需要 LAMMPS 编了 PLUMED（`mflow doctor -lmp`）；集体变量自己写 |
| 模型已较好、想找它的盲区 | `ubmd`（不确定度偏置 MD） | 需 ≥ 2 个模型；`kappa` 从 0.5 起，看 failed 比例；会进非物理区 |

第 0 轮用基础模型探索（不需要委员会），只做多样性筛选——这是最便宜的冷启动。

## 3. 筛选

- **委员会不是必需的。** 用户的立场：功能有，但不默认依赖。从同一个基础模型微调的成员容易**塌缩**（分歧远小于真实误差；`train` 报告出现 COLLAPSED，即 RMS(误差)/RMS(σ) ≥ 3）。
  - 不用委员会：`train.committee: 1`，`select.uncertainty: descriptor`（到最近训练帧的距离）或 `quests`（dH > 0，无需校准，`pip install quests`）。
  - 用委员会：≥ 3，建议 4；微调保持 bagging；最有效的多样性是不同基础模型（`train.foundations: [mpa, omat]`，r_max 必须相同）；`member_set` 可逐成员加 `subselect_pt` 等选项。`select` 记录里的 `dev_median_wo_driver` 是去掉驱动 MD 的成员 0 后的分歧——两者差很多说明采样被成员 0 带偏。
  - 改这些之前和用户确认，并写进 NOTEBOOK。
- 信任窗口 `trust: fixed` 起步。MFlow 的偏差就是 DP-GEN 的定义（每个原子 sqrt(mean_m |F_m − ⟨F⟩|²)，取最大），所以 DP 系列的经验值可以直接用：

  | 体系（DP 文献） | lo / hi（eV/Å） |
  |---|---|
  | Al | 0.05 / 0.15 |
  | Cu | 0.05 / 0.20 |
  | W | 0.20 / 0.35 |
  | 水 | 0.15 / 0.25，后期 0.20 / 0.35 |
  | Li₁₀GeP₂S₁₂ | 0.12 / 0.25 |

  经验：hi 比 lo 高 0.15–0.30；lo 要高于模型能达到的最好误差（否则永远"不准"）。其他工具（ai2-kit 的 MACE 脚本、AtomisticSkills）的偏差定义不同，它们的数字不能直接搬。
- 委员会校准好（Spearman ≥ 0.5）后可试 `auto`。力很大的体系用 `relative`。不知道误差下限（微调基础模型时常见）用 `adaptive-lower`（dpgen2）：lo 跟着偏差分布走，连续 2 轮变化 < 0.05 eV/Å 后冻结；冻结前 accurate 比例不算数。
- `recalibrate: true`（默认）：σ 按测试集 RMS(误差)/RMS(σ) 放大。比值 > 10 说明委员会几乎没有信息量——先多一轮标注，或把成员从头训练。
- 读每轮的分类比例：
  - 几乎全 accurate → 探索不够远（升温、加步数、开 minhop、加大 `widen`）；
  - 几乎全 failed → 模型还没准备好（MFlow 会从 failed 里按多样性选），或 MD 炸了（看 explore 目录）；
  - candidate 5–30 % → 正常；
  - accurate 长期 < 90 % → lo 太严（低于模型能力）；
  - 一个候选都没有 → 先查窗口、MD 是否稳定、dump 频率，再动训练。
- 升温分段推进（例如 T_m 以下 → T_m → 1.5 T_m → 2 T_m），每个新温段先用短 MD（500–1000 步）和少量起点。

## 4. 标注与导入

- 只用 `mflow al import -tflow` 或带 `mflow_id` 的 extxyz；不要手工拼数据集。
- 拒收比例 > 10 % 要查：`rejects.jsonl` 的原因分布。SCF 不收敛多 → `run_label.py` 调 ALGO/NELM/混合参数；error fix 多 → 这些结构可能不物理。
- 导入时的"新帧误差"是最诚实的精度估计，每轮都看它。

## 5. 何时停

看 `al status` 的分级表（或 `status.json` 的 `verdict`）：
- **L1（每轮 / 每区）**：accurate ≥ 99 %、failed ≤ 0.1 % 连续 2 轮；FPS 后剩下的帧 < max(10, 0.1 %)；候选比例 < 5 %；新帧误差 < 10 meV/atom、50 meV/Å，且 ≤ 1.5 × 测试集误差；探索 MD 全部跑完。
- **L2（整体）**：`probes` 性质连续 2 轮在容差内、成员之间一致；分布外比例 ≤ 1 %。没配 probes 时这些行是 n/a——**建议配上**（至少 eos + dimer），否则"收敛"只说明误差低，不说明性质稳。
- **预算**：`stop.max_rounds` / `max_dft` 到了，状态是 `budget_exhausted`——**向用户报告"未收敛"**，列出没满足的行，不要说成"完成"。

`verdict: converged` 之后仍要在生产条件下跑 MD 并通过全部闸门（`mflow-potential-acceptance`）。
只满足候选比例：多半是探索不够，不是模型够好。连续两轮候选比例不降：换探索器，而不是继续同样的循环。

## 6. 误区

| 误区 | 后果 |
|---|---|
| 每轮改 DFT 设置 | 数据集自相矛盾 |
| 测试集随轮次变化 | 进步无法判断 |
| 只看委员会 σ 不看新帧误差 | σ 过度自信时会过早停 |
| 窗口上限太低，把高偏差帧全丢掉 | 正是最该学的构型被扔了；看 failed 比例 |
| 一轮标注上千帧 | 大量冗余；FPS 限量，每轮 50–150（DP 文献约 100） |
| 把多样性或能量 std 排序当成"校准过的"采集函数 | 它们只能排序；校准看 `committee_report` |
| 委员会 COLLAPSED 还只靠 σ 选帧 | σ 远小于真实误差，过早"收敛"；换异构成员或描述子 / QUESTS |
| 预算用完就宣布完成 | `budget_exhausted` ≠ converged |
| 改写上一轮的选择 | 台账失真；每轮新目录，数据集重建 |

## 7. 每轮记录（NOTEBOOK）

轮次、区、探索器与参数、explored / candidate / selected、拒收数与原因、新帧误差、Spearman 与 σ 比值（是否 COLLAPSED）、探针变化、`verdict`、下一轮改了什么及原因。`ledger.jsonl` 有数字，NOTEBOOK 写决定。

出处与改写说明见 `references/SOURCES.md`。
