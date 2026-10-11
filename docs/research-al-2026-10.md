# 主动学习训练 MACE：探索方法、委员会/UQ、收敛判据与可复用项目

> 调研日期 2026-10-11 · 面向 MFlow v0.3（主动学习微调 foundation 模型）
> 标注 **[未证实]** 的条目没能在原始文献或代码中核实，仅作线索。数值标注“文献”的出自所引论文，标注“建议”的是本报告综合给出的默认值，需按体系调整。

---

## 0. 结论先行

1. **没有现成框架能直接覆盖 MACE + VASP + LAMMPS 全链路。**
   - DP-GEN / dpgen2 绑定 DeePMD，没有 MACE 训练器。
   - aims-PAX 的 MACE 主动学习做得最好，但标注只支持 FHI-aims。
   - psiflow 无法自动调度 VASP（见其 FAQ）。
   - autoplex 原生支持 VASP，但定位是随机结构搜索（RSS），且绑定 atomate2/jobflow。

   因此保持 MFlow「薄编排 + TFlow 标注」的路线，**复用库和设计，不引入框架**。
2. **探索方法：起步阶段最划算的是用 foundation 模型自己探索。**
   - 流程：用 MPA/OMAT 跑 T–P 梯度 MD，委员会信任窗口加 FPS 选帧，交给 VASP 标注，再微调。
   - 按需要补充：增强采样（反应、稀有事件）、全局搜索（未知结构）、不确定度偏置 MD（委员会校准好以后）。
3. **委员会：从同一个 foundation 微调出的成员会“塌缩”。** 成员间分歧小，与真实误差相关性弱。成员必须刻意做出差异：不同数据子集、不同 replay 子集，最好混入不同的 foundation。同时每轮用新标注的帧检查校准。
4. **收敛不能只看“候选比例”。** 建议用三级判据：
   - L0 校准阈值；
   - L1 每轮 / 每个温压区的选帧比例、first-seen 误差和 MD 稳定性；
   - L2 性质跨轮稳定、成员间性质一致、覆盖度，再加上硬预算。

---

## 1. 探索方法（怎么产生候选结构）

| 方法 | 怎么产生候选 | 适合 | 不适合 | 关键文献 | 可复用实现 | MFlow 现状 |
|---|---|---|---|---|---|---|
| **MD + 委员会信任窗口**（DP-GEN 式） | 在温度/压强梯度上跑 MD，委员会最大力偏差 ε 落在 [σlo, σhi) 的帧作为候选 | 体相、液体、近平衡相 | 稀有事件、势垒；容易在势阱里过采样 | Zhang et al., CPC 253, 107206 (2020), arXiv:1910.12690；Wen et al., arXiv:2203.00393 | DP-GEN/dpgen2（DeePMD 专用）；MACE 自带 `mace/cli/active_learning_md.py`（ASE 委员会 MD，只停不标注，v0.3.16 已核实存在） | ✅ `explore.md`（ASE / LAMMPS），T×P 网格，`widen` 外扩 |
| **foundation 模型驱动探索 + 微调** | 用 MPA/OMAT/MatterSim 跑 MD 或搜索 → FPS/不确定度选帧 → DFT → 微调 | 冷启动；10²–10³ 个 DFT 点 | foundation 模型定性错误的体系（PES 偏软、相序错），采样会被带偏 | aims-PAX arXiv:2508.12888；frozen transfer, Radova et al., npj Comput. Mater. 11, 237 (2025), arXiv:2502.15582；蒸馏 Gardner et al., arXiv:2506.10956 | aims-PAX（MIT，FHI-aims）；MACE `fine_tuning_select.py`（已核实）；`--freeze N`（v0.3.16 已核实） | ✅ round 0 用 foundation 探索 + FPS（`seed.n_select`） |
| **增强采样**（metaD / OPES / 副本交换 / 升温淬火） | 沿集体变量（CV）施加偏置，把体系推过势垒 | 反应、催化、相变（需有好的 CV） | CV 未知；早期势能面不稳定 | Perego & Bonati, npj Comput. Mater. 10, 291 (2024)；ArcaNN, Digital Discovery 2025, DOI 10.1039/D4DD00209A | LAMMPS `fix plumed`、ASE Plumed 计算器；psiflow（MIT）；janus-core（BSD-3）；dpgen2 于 2026-09 加入 PLUMED CV 过滤 | ❌（`anneal` / `quench` 协议只能部分替代升温淬火） |
| **全局搜索**（minima hopping、basin hopping、AIRSS / GAP-RSS、遗传算法、嵌套采样） | 随机生成或变异结构 → 弛豫 → 收集局域极小及弛豫路径 | 未知多晶型、表面、团簇、非晶；从零建库 | 有限温度动力学（需要再补一轮 MD） | GAP-RSS, Bernstein et al., npj Comput. Mater. 5, 99 (2019)；autoplex, Nat. Commun. 16, 7666 (2025)；active Δ-learning + AGOX, arXiv:2507.18485 | ASE `MinimaHopping` / `BasinHopping` / `ga`；autoplex（GPL-3，VASP 原生，MACE 为可选依赖）；ALomancy（MIT，带 GA） | ✅ `explore.minhop`；❌ RSS / GA |
| **不确定度偏置 MD / 对抗攻击** | 在能量上加一个与不确定度相关的偏置，主动把体系推向模型没把握的区域 | 中小分子、反应势垒；能快速找到外推区 | 大周期体系；偏置需要不确定度对坐标的梯度，可能跑进非物理的高能区 | UDD, Kulichenko et al., Nat. Comput. Sci. 3, 230 (2023)；UBMD, Zaverkin et al., npj Comput. Mater. 10, 83 (2024), arXiv:2312.01416；对抗攻击, Schwalbe-Koda et al., Nat. Commun. 12, 5104 (2021) | 对抗攻击代码 learningmatter-mit/Atomistic-Adversarial-Attacks（MIT，非 MACE）；**没有现成的 MACE 实现**，需要自己写 ASE 计算器包装 | ❌ |
| **on-the-fly**（FLARE、VASP MLFF） | 自带廉价贝叶斯不确定度，MD 跑到一半就调用 DFT | 无先验数据的新体系 | 不能直接训 MACE | Vandermause et al. (FLARE)；Jinnouchi et al., PRL 122, 225701 (2019) | FLARE（MIT，活跃）；VASP 的 `ML_MODE=train` | ❌ 只适合当前端采样器：VASP 的 `ML_AB` 帧可以拿来训 MACE |

**排序（针对 MACE + ASE + LAMMPS ML-IAP + VASP/TFlow）**
1. foundation 模型驱动的 MD + 委员会 + FPS：MFlow 已有，下一步是**分区、分阶段**（见 §6）。
2. 增强采样：`fix plumed` + MACE ML-IAP，针对反应和稀有事件。
3. 全局搜索：ASE GA，或借鉴 autoplex / ALomancy 的 RSS 思路，针对未知结构。
4. 不确定度偏置 MD：等委员会校准可靠后再做，需要自己实现。
5. on-the-fly：只在完全没有 foundation 覆盖的元素组合上，作为补充前端。

**分阶段探索的做法（DP-GEN 经验，文献）**
- 温度从 50 K 到 2 Tm 分成约 4 个区，每个区单独判定收敛。
- 新区开头几轮只跑 500–1000 步、少量初始结构，之后逐轮加长、加多。
- 水的例子里，信任窗口随轮次放宽：[0.15, 0.25] → [0.18, 0.32] → [0.20, 0.35] eV/Å，共 36 轮。

---

## 2. 多模型并行训练与不确定度（UQ）

### 2.1 两种并行方式

| 方式 | 做法 | 成本 | 优缺点 |
|---|---|---|---|
| **一个模型驱动 + 委员会重打分**（MFlow 现方案） | 模型 A 在 LAMMPS 里跑 MD，每 10–50 步 dump 一帧；然后在 GPU 上用 `MACECalculator(model_paths=[A,B,C,D])` 重打分 | MD 1×，打分 N× | 便宜。缺点是采样被 A 带偏，建议**同时报告不含 A 的 B–D 分歧**；MD 分成 5–20 ps 的短段，跑坏了能及时截断 |
| **每个成员各跑一条轨迹**（DP-GEN 式） | N 个模型各自驱动 MD，再交叉打分 | MD N× | 采样更分散，能暴露各成员各自的错误 |
| `pair_style hybrid/overlay` 叠多个 `mliap unified` | — | 每个成员都要做完整的前向 + 反向 | **不推荐**：没有逐成员原子力输出；Kokkos 下多个 mliap 实例能否共存 **[未证实]** |
| metatomic `pair_style metatomic_1…9` + `uncertainty_threshold` | 在 LAMMPS 运行时直接报警 | — | 需要换成 metatrain 导出；MACE + LLPR 在其中是否可用 **[未证实]** |

### 2.2 同一个 foundation 微调出的委员会会“塌缩”

- Varga-Umbrich et al., arXiv:2605.03964 (2026)：从同一 checkpoint 微调 3 个 MACE，只改数据顺序或 bootstrap。委员会分歧与真实误差相关性弱；用预训练特征核或 NTK 核反而更好。另见 arXiv:2605.13788（NTK 主动学习）。
- Beck et al., JCP 163, 234103 (2025), arXiv:2508.09907：多头委员会，即冻结骨架、只训多个新读出头。它比 8 个完整微调的 MACE-MP-0 副本更准（力误差约差 4 倍）。对力误差跟踪较好，对能量误差较差。代码是 ACEsuit/mace PR #800，仍是未合并草稿，基于 0.3.7 分支。
- Kurniawan et al., arXiv:2508.06456：独立种子 / bootstrap / dropout / snapshot 集成的不确定度，在分布外误差增大时可能停止增长甚至下降。
- Kahle & Zipoli, PRE 105, 015311 (2022)：集成普遍过度自信，需要逐体系校准。

**让成员保持差异的做法（按效果从高到低）**
1. 混入**不同的 foundation**（MPA + OMAT，或 MACE-MP-0 medium）。前提是 `r_max` 相同，否则用各自的计算器分别打分。
2. 每个成员用**不同的 bagging 子集**（80–90%）、不同的验证集划分、不同的 seed。
3. 用多头 replay 微调时，每个成员抽**不同的 replay 子集**。
   - `--num_samples_pt`、`--subselect_pt random|fps` 和 seed 的组合在 v0.3.16 中存在（已核实）。
   - 但 random 子选是否随 `--seed` 变化 **[未证实]**，实现时要测试。
4. 冻结骨架、随机初始化读出层（`--freeze N`），或采用 PR #800 的多头委员会。

**塌缩检测**：每轮在 20–50 个新标注的帧上计算 Spearman ρ（预测分歧 vs 真实力误差），以及分歧 ÷ RMSE。这个比值远小于 1（例如 < 0.3，建议值）就是塌缩，应加入异构成员。注意 MFlow 的 `sigma_ratio` 定义相反（RMS(误差)/RMS(σ)），所以对应 `sigma_ratio` ≥ 3；v0.3 起由 `uq.collapse_ratio` 判定并标 COLLAPSED。

### 2.3 单模型 UQ（廉价替代或补充）

| 方法 | 文献 | 代码 | MACE 0.3.16 能否直接用 |
|---|---|---|---|
| LLPR（最后一层预测刚度） | Bigi et al., arXiv:2403.02251；MACE 上的测试见 arXiv:2408.14311 | metatrain `llpr`（2025.11 稳定） | 否。metatrain 在 2026.1 加了 MACE 包装，但 LLPR 能否包它 **[未证实]**；ai2-kit 中有人在 MACE-MP-0 / OMAT 上用过（arXiv:2607.14486） |
| 浅集成（DPOSE） | Kellner & Ceriotti, arXiv:2402.16621；Schäfer et al., arXiv:2602.15747 | bananenpampe/DPOSE（演示代码） | 否 |
| 逐原子灵活校准 | Ho, Ortner, Wang, arXiv:2510.00721 | 未找到仓库 | 基于 MACE-MP-0b3 + LLPR |
| 描述符空间距离 / GMM | Zhu et al., JCP 158, 164111 (2023) | — | **可以自己做**：`get_descriptors` 取描述符，再算到训练集的最近邻距离 |
| QUESTS δH | Schwalbe-Koda et al., Nat. Commun. 16, 4014 (2025), arXiv:2404.12367 | github.com/dskoda/quests（BSD-3，`pip install quests`，`quests.entropy.delta_entropy(y, x, h=0.015)`，已核实） | 与模型无关，可以直接用 |
| MC dropout、证据回归 | eIP arXiv:2407.13994；e2IP arXiv:2602.10419 | — | 否 |

**mace-torch 0.3.16 实际提供的 UQ**：`MACECalculator(model_paths=[...])` 返回 `energy_var`、`forces_comm`（每个成员的力）、`forces_var`、`stress_var`。委员会成员必须有相同的 `r_max`。除此之外没有内置的单模型 UQ。

### 2.4 用什么指标、怎么校准

- 主指标用**逐原子力标准差的最大值**（即 DP-GEN 的 `max_devi_f`）。能量方差太全局，抓不到局部的失效（文献普遍做法）。MFlow 已采用这个指标。
- 校准用全局 α（Musil et al., JCTC 15, 906 (2019)；Imbalzano et al., JCP 154, 074102 (2021)），MFlow 的 `recalibration` 已实现。可选：共形分位数、逐原子缩放网络（Ho et al.）。
- 只用能量训练的模型，力的不确定度校准很差（Schäfer et al.）。在异构数据集上，不确定度会系统性低估误差（arXiv:2508.03405）。

---

## 3. 收敛判据

### 3.1 文献中的做法

| 判据 | 出处 | 具体规则 |
|---|---|---|
| 选帧比例 | DP-GEN, Wen et al. arXiv:2203.00393 | accurate 帧比例超过 **99%** 时停止标注，再做一次长训练。accurate 长期低于 90% 说明 σlo 太严；几乎没有 failed 帧说明 σhi 太松 |
| `fixed-levels` | dpgen2 文档（已核实） | accurate ≥ `conv_accuracy`（示例 0.9）；`max_numb_iter` 限制轮数，`fatal_at_max` 决定超出时是否报错 |
| `adaptive-lower` | dpgen2 源码（已核实） | 每轮重设 σlo，使候选帧数为 max(`numb_candi_f`=200, 0.01 × 帧数)；σhi 默认 0.5。σlo 连续 `n_checked_steps`=2 轮变化 < `conv_tolerance`=0.05 eV/Å 时，该阶段收敛。**适合不知道误差下限的场景，比如微调 foundation 模型** |
| 滑动平均阈值 + 预算 | aims-PAX arXiv:2508.12888 | 阈值 δt = (1+c)·最近 400 个不确定度的均值，c ∈ [−0.1, 0.1]；达到预算（100–1000 个结构）或验证 MAE 目标时停止 |
| 误差 vs DFT 次数 | autoplex, Nat. Commun. 16, 7666 (2025) | 每轮 100 个单点计算，目标 10 meV/atom，凭人眼判断平台 |
| 稳定性 | Fu et al., “Forces are not enough”, arXiv:2210.07237 | 力误差低不代表模拟可靠，关键指标是模拟能跑多久不崩。失效的定义：密度离开 0.2–2.0 g/cm³ 或键伸长 > 0.5 Å（arXiv:2409.05590）；100 ps NVT 中出现温度尖峰（MLIPAudit, arXiv:2511.20487） |
| 性质跨轮稳定 | arXiv:2601.16047 | 部分：eos / dimer / phonon 探针；密度、RDF、D 的跨轮比较尚未做 | 扩散系数到第 4 轮才与参考一致，且随轮次**非单调**变化，所以要求连续多轮稳定 |
| 覆盖度 | QUESTS | ✅ | δH > 0 表示新环境在训练集之外（无需校准）；数据增长后熵会饱和，剩余“信息差”与力误差相关（r = 0.89）；论文没给停止阈值 |
| 性质验证 | Morrow, Gardner, Deringer, arXiv:2211.12484；MLIP Arena arXiv:2509.20630；LAMBench arXiv:2504.19578 | 主动学习结束后测 EOS、弹性常数、缺陷、声子、dimer 曲线 |

**first-seen 误差**：用当前模型（重训之前）在本轮新标注帧上算误差。这些帧按构造就是最难的，所以这是最诚实的误差估计。它是常见做法，但没找到把它定为停止规则的正式文献 **[未证实为具名判据]**。MFlow 已在 `fresh_frame_error` 中实现。

### 3.2 推荐的三级判据（本报告综合，非已发表规则）

**L0 校准期（第 0–2 轮）**
- σlo 用 dpgen2 的 adaptive-lower 方式确定，直到连续 2 轮变化 < 0.05 eV/Å（文献值），然后冻结。
- σhi = σlo + 0.15 eV/Å（建议）。
- 用 first-seen 力误差校准委员会：拟合 ε 与真实误差的斜率，即 α。

**L1 每轮 / 每个温压区。** 下面四条全部满足，该区才算通过：
1. accurate ≥ 99%（文献），failed ≤ 0.1%（建议），**连续 2 轮**。
2. FPS 去重后的候选数 < max(10, 0.1% × 帧数)（建议）。
3. first-seen 力 RMSE ≤ 1.5 × 验证集 RMSE，且 ≤ σlo（建议）。
4. 100% 的轨迹跑满设定长度，没有任何失效标志（温度尖峰、键断裂、密度越界、丢原子）。

**L2 整个 campaign**（所有区都通过后，做最终长训练）：
1. 性质跨轮稳定：连续 2–3 轮的变化落在块平均误差棒内。建议容差：

   | 性质 | 容差 |
   |---|---|
   | 密度 | 0.5% |
   | RDF 积分绝对差 | < 0.02 |
   | 扩散系数 D | 10% 且在 2σ 内 |
   | VDOS 重叠 | > 0.98 |
   | 声子 | 0.2 THz |
   | EOS | V0 0.5%，B0 5% |
   | dimer 曲线 | 保持排斥，无伪极小 |

2. 委员会成员之间，同一性质的散布 < 该性质的容差。
3. 覆盖度：生产 MD 中 ≥ 99% 的环境 QUESTS δH ≤ 0；最后一轮带来的熵增 < 数据集熵的 1–2%。
4. 硬停止：达到最大轮数或 DFT 预算时退出，报告**明确写“未收敛”**。

---

## 4. 可复用项目一览

维护状态取自 GitHub 页面和提交记录，截至 2026-09 / 10。

| 项目 | URL | 许可证 | MACE | 探索方式 | 调度 | 维护 | 复用方式 |
|---|---|---|---|---|---|---|---|
| DP-GEN | github.com/deepmodeling/dpgen | LGPL-3.0 [未证实] | 否 | LAMMPS MD，温压梯度，信任窗口 | dpdispatcher | 活跃 | 借鉴设计（分区、比例判据） |
| dpgen2 | github.com/deepmodeling/dpgen2 | LGPL-3.0 | 否 | LAMMPS、CALYPSO、DiffCSP、PLUMED 过滤 | dflow/Argo | 活跃 | **借鉴 adaptive-lower 和 exploration report** |
| aims-PAX | github.com/tohenkes/aims-PAX | MIT | **是（默认）** | foundation 模型 MD、委员会 MD、多轨迹、LoRA | Parsl + Slurm | 活跃 | 借鉴滑动阈值和整体流程；标注只支持 FHI-aims |
| psiflow | github.com/molmod/psiflow | MIT | 是 | MD、PIMD、副本交换、metaD（i-PI + PLUMED） | Parsl | 2026-04 | 增强采样的参考实现；不能自动跑 VASP |
| autoplex | github.com/autoatml/autoplex | GPL-3.0 | 可选依赖（mace-torch 0.3.16） | RSS、声子 / rattle | atomate2 / jobflow-remote | 活跃 | RSS 思路；GPL 不宜直接拷贝代码 |
| wfl + ExPyRe | github.com/libAtoms/workflow | GPL-2.0 | 通过 ASE 计算器 | MD 等构件 | ExPyRe 远程队列 | 有发布 | 库；MFlow 已有 TFlow，暂不需要 |
| ALomancy | github.com/julianholland/ALomancy | MIT | **是（默认）** | Langevin NVT/NPT、GA（EZGA） | ExPyRe | v1.0，小项目 | GA 探索可参考 |
| IPSuite | github.com/zincware/IPSuite | EPL-2.0 | 是 | MD；apax 不确定度偏置 [未证实] | DVC / ZnTrack | 2026-03 | 框架 |
| PAL | github.com/aimat-lab/PAL | [未证实] | 是（论文示例） | 用户自定义 | MPI | 2026-07 | MPI 并行内核 |
| ArcaNN | github.com/arcann-chem/arcann_training | [未证实] | 否 | 增强采样 MD + 委员会 | Slurm | 2026-02 | 借鉴设计 |
| FLARE | github.com/mir-group/flare | MIT | 否 | on-the-fly 稀疏 GP | 用户脚本 | 活跃 | 前端采样器 |
| janus-core | github.com/stfc/janus-core | BSD-3 | 是（另有 MatterSim 等） | MD、NEB、PLUMED、微调；没有主动学习循环 | aiida-mlip | 活跃 | PLUMED 接 MACE 的参考 |
| **QUESTS** | github.com/dskoda/quests | **BSD-3** | 与模型无关 | 熵、多样性、δH、learning curve | — | 活跃 | **直接作为可选依赖** |
| metatrain | metatrain.readthedocs.io | BSD-3 [未证实] | 实验性 MACE 包装 | LLPR、浅集成 | — | 很活跃 | 以后考虑单模型 UQ 时再看 |
| MACE 自带工具 | ACEsuit/mace `mace/cli/` | MIT | — | `active_learning_md.py`、`fine_tuning_select.py` | — | — | **直接调用**（v0.3.16 已核实存在） |
| mace_active_learning | github.com/MadChemLad-12/mace_active_learning | MIT | 是（多头微调） | foundation MD/NEB + 不确定度 + FPS | 手动 / Slurm | 早期，CP2K | 仅作模板参考 |

**结论**
- 可以直接引入的只有两类：**QUESTS**（BSD-3，可选依赖），以及 **MACE 自带的 CLI 和参数**。
- dpgen2 的 adaptive-lower、aims-PAX 的滑动阈值、DP-GEN 的分区分阶段，**照着设计自己实现**。代码量都很小，也避开了 GPL/LGPL 的牵连。

---

## 5. 微调 foundation 模型的要点（为下一阶段准备）

- Tompa et al., arXiv:2606.12704 (2026，Csányi 组)：
  - 先把基础做对（基础模型质量、E0 初始化、超参数），这比选哪种微调策略更重要。
  - 单一体系用朴素全微调最好。
  - **多头 replay 是唯一能稳定保住分布外鲁棒性和短程排斥的策略。**
- frozen transfer（`--freeze N`，v0.3.16 已核实）：MACE-MP-f4 只用 20% 的数据（664 个构型），就达到从零训练用全部数据的水平（Radova et al.）。
- v0.3.16 中已核实的多头 replay 参数：
  - `--multiheads_finetuning`（默认 True）
  - `--pt_train_file`（可写 `mp`）
  - `--num_samples_pt`（默认 10000）
  - `--subselect_pt`（默认 random）
  - `--weight_pt_head`
  - `--foundation_head`
  - `--filter_type_pt`
- 已核实的 LoRA 参数：`--lora`、`--lora_rank`。
- 微调教程：Liu et al., arXiv:2506.21935（J. Appl. Phys. 139, 041101 (2026)）只说“误差饱和或达到数据预算”就停，没给具体数字。
- 前几轮的注意事项：
  - 同源委员会过度自信，用 first-seen 误差校准 σlo，或用 adaptive-lower。
  - MD 先短后长。
  - replay 防止遗忘。

---

## 6. 对照 MFlow：差距与 v0.3 建议路线

MFlow v0.2 已有：
- MD / perturb / minhop 探索；
- 委员会训练（bagging），以及 Spearman 和 sigma_ratio 校准；
- 全局 α 重标定；
- fixed 信任窗口 + FPS；
- TFlow 导出 / 导入；
- `stop_criteria`（候选比例 + first-seen 误差）；
- validate（dimer / EOS / 声子 / 缺陷）；
- md-report（块平均、扩散、VDOS）。

建议按以下优先级补齐（**v0.3.0 已实现，见最右列**；用法见 README 4.4）：

| 优先级 | 内容 | 改动位置 | 依据 | v0.3.0 |
|---|---|---|---|---|
| 1 | **收敛判据升级**：加入连续 N 轮、failed 比例、MD 稳定性门、DFT 预算 / 最大轮数硬停止（标记“未收敛”）；信任窗口增加 `adaptive-lower` 模式 | `mflow/al/commands.py` 的 `stop_criteria`；`mflow/al/select.py` 的 `trust_window`；`thresholds.yaml` 的 `al` 段 | dpgen2、DP-GEN、Fu 2023 | ✅ |
| 2 | **委员会多样性**：每个成员用不同的 replay 子集；允许混合 foundation（检查 r_max）；分歧 < 真实误差的 0.3 倍（即 MFlow 的 sigma_ratio = RMS(误差)/RMS(σ) ≥ 3）时明确判为“塌缩”，并给出处理建议 | `mflow/train/run.py` 的 `train_committee`；`step_train` | arXiv:2605.03964、2508.09907 | ✅ `train.foundations`、逐成员 `member_set`（replay 子集手动设）、COLLAPSED；另加 `committee: 1` 单模型模式 |
| 3 | **分区、分阶段探索**：温度分区，每区单独收敛；新区 MD 先短后长 | `mflow/al/explore.py` 的 `md_grid` / `run_explorers`；`campaign.yaml` | DP-GEN | ✅ 另加多驱动 MD |
| 4 | **覆盖度和分布外检测**：QUESTS δH（可选依赖），再加上描述符最近邻距离 | `mflow/al/select.py` 的 `descriptors` | QUESTS | ✅ |
| 5 | **性质跨轮稳定门**：每轮对固定的性质集（密度、RDF、D、EOS、dimer、声子）做比较，判据用误差棒 | 复用 `mflow/analysis/report.py`、`mflow/validate.py` | arXiv:2601.16047 | 部分：eos / dimer / phonon 探针；密度、RDF、D 的跨轮比较尚未做 |
| 6 | **新探索器**：PLUMED 增强采样（LAMMPS `fix plumed`）、升温淬火、ASE GA；不确定度偏置 MD 放最后 | `mflow/al/explore.py`、`mflow/lmp/inputs.py` | Perego & Bonati、UBMD | ✅ PLUMED（`fix plumed`）、heatquench、ubmd；GA 改为随机结构搜索 rss |

建议先做 1 和 2。理由：它们直接决定“什么时候该停”和“选帧信号是否可信”，改动面也小，可以在下一次微调 foundation 模型时马上用上。

---

## 7. 参考文献（按出现顺序）

- Zhang et al., DP-GEN, Comput. Phys. Commun. 253, 107206 (2020), arXiv:1910.12690
- Wen et al., Deep Potentials for Materials Science, arXiv:2203.00393
- dpgen2 文档：docs.deepmodeling.com/projects/dpgen2（`report_adaptive_lower`、`report_trust_levels_max`）
- Henkes et al., aims-PAX, arXiv:2508.12888
- Radova et al., frozen transfer learning, npj Comput. Mater. 11, 237 (2025), arXiv:2502.15582
- Gardner et al., synthetic distillation, arXiv:2506.10956
- Perego & Bonati, npj Comput. Mater. 10, 291 (2024)
- ArcaNN, Digital Discovery (2025), DOI 10.1039/D4DD00209A
- Bernstein et al., GAP-RSS, npj Comput. Mater. 5, 99 (2019)
- Liu et al., autoplex, Nat. Commun. 16, 7666 (2025), DOI 10.1038/s41467-025-62510-6
- Active Δ-learning with AGOX, arXiv:2507.18485
- Kulichenko et al., UDD, Nat. Comput. Sci. 3, 230 (2023), DOI 10.1038/s43588-023-00406-5
- Zaverkin et al., UBMD, npj Comput. Mater. 10, 83 (2024), arXiv:2312.01416
- Schwalbe-Koda et al., adversarial attacks, Nat. Commun. 12, 5104 (2021), arXiv:2101.11588
- Jinnouchi et al., PRL 122, 225701 (2019)
- Schran, Brezina, Marsalek, JCP 153, 104105 (2020), arXiv:2006.01541
- Musil et al., JCTC 15, 906 (2019), arXiv:1809.07653；Imbalzano et al., JCP 154, 074102 (2021), arXiv:2011.08828
- Varga-Umbrich et al., arXiv:2605.03964；NTK active learning, arXiv:2605.13788
- Beck et al., multi-head committees, JCP 163, 234103 (2025), arXiv:2508.09907；ACEsuit/mace PR #800
- Kurniawan et al., arXiv:2508.06456；Kahle & Zipoli, PRE 105, 015311 (2022)；Tan et al., npj Comput. Mater. 9, 225 (2023), arXiv:2305.01754
- Bigi et al., LLPR, arXiv:2403.02251；arXiv:2408.14311；Bi et al., arXiv:2607.14486
- Kellner & Ceriotti, arXiv:2402.16621；Schäfer et al., arXiv:2602.15747
- Ho, Ortner, Wang, arXiv:2510.00721；Zhu et al., JCP 158, 164111 (2023)；arXiv:2508.03405
- Schwalbe-Koda et al., QUESTS, Nat. Commun. 16, 4014 (2025), arXiv:2404.12367
- Fu et al., Forces are not enough, arXiv:2210.07237；arXiv:2409.05590；MLIPAudit arXiv:2511.20487；arXiv:2601.16047
- Morrow, Gardner, Deringer, arXiv:2211.12484；MLIP Arena arXiv:2509.20630；LAMBench arXiv:2504.19578
- Tompa et al., arXiv:2606.12704；Liu et al., fine-tuning tutorial, arXiv:2506.21935
- MACE v0.3.16 源码：`mace/tools/arg_parser.py`、`mace/cli/active_learning_md.py`、`mace/cli/fine_tuning_select.py`
