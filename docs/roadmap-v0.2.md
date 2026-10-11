# MFlow v0.2 路线图

> 状态：**v0.2.0a1 已实现 a–e**（2026-10），尚未在 GPU + 真实 MACE/LAMMPS 上做端到端验证。实现与规划的差异见第 7 节。v0.1 的内容见 [README](../README.md)。
> 范围约定：**DFT 计算不在 MFlow 里做**，由 TFlow 负责；MFlow 只负责"提议哪些结构值得算、选出来、交出去、收回来、再训练、判断何时停"。

v0.2 有四条主线：

1. **训练方案**：委员会训练 + 主动学习闭环。
2. **精度评估**：补齐验证第 1、2 层，加不确定度校准。
3. **LAMMPS 协议库**：更多恒温器、恒压器和性质协议。
4. **经验类 skill**：9 个。

---

## 1. 训练方案全景（来自 2024–2026 年的综述）

### 1.1 依据的综述

| # | 文献 | 用到的部分 |
|---|---|---|
| R1 | Kulichenko … Tretiak, *Data Generation for Machine Learning Interatomic Potentials and Beyond*, **Chem. Rev.** 124, 13681 (2024), [10.1021/acs.chemrev.4c00572](https://doi.org/10.1021/acs.chemrev.4c00572) | 主动学习、不确定度与真实误差、采样器（MD、增强采样）、修饰势能面（如不确定度驱动的动力学）、公开数据集 |
| R2 | Grasselli, Chong, Kapil, Bonfanti, Rossi, *Uncertainty in the era of machine learning for atomistic modeling*, **Digital Discovery** 4, 2654 (2025), [arXiv:2503.09196](https://arxiv.org/abs/2503.09196) | 不确定度的分类（集成、浅层集成、末层、贝叶斯、保形预测）、校准、分布外检测、主动学习信号（D-optimality、信息熵）；**没有给出正式的停止规则** |
| R3 | Dai, Adhikari, Wen, *Uncertainty Quantification and Propagation in Atomistic ML*, [arXiv:2405.02461](https://arxiv.org/abs/2405.02461) (2024) | 不确定度的评价指标（准确度、精度、校准、效率），以及不确定度如何传播到 MD |
| R4 | Jacobs, Morgan … Csányi, Kozinsky, Ong, *A practical guide to MLIPs – Status and future*, **Curr. Opin. Solid State Mater. Sci.** 35, 101214 (2025), [arXiv:2503.09814](https://arxiv.org/abs/2503.09814) | 训练数据来源、用预训练模型还是从头训练、主动学习概览、超出 E/F 误差的验证 |
| R5 | Tompa … Bernstein, Csányi, *Fine-tuning MLIP foundation models: strategies for accuracy and transferability*, [arXiv:2606.12704](https://arxiv.org/abs/2606.12704) (2026) | 朴素微调、多头回放、LoRA、冻结层四种策略的系统对比 |

### 1.2 一轮主动学习 = 探索 → 筛选 → 标注 → 训练 → 判断

```
            ┌────────────── MFlow ──────────────┐            ┌─ TFlow ─┐
 模型 vN ──► 探索（提议结构） ──► 筛选（不确定度/多样性） ──► export ──► DFT 标注
    ▲                                                                  │
    └── 训练 vN+1 ◄── check（体检）◄── import ◄────────────────────────┘
                     └── status：候选比例、新帧上的 DFT 误差、性质是否收敛 → 停？
```

### 1.3 探索：怎么提议结构

| 方法 | 适用场景 | 成本 | 依据 | v0.2 |
|---|---|---|---|---|
| **MD 温度/压力网格**（DP-GEN 式） | 凝聚相的默认方法 | 低（用 MLIP 跑） | Zhang et al. *PRM* 2019（DP-GEN） | ✅ `explore md`：在 LAMMPS 里跑许多短轨迹，逐轮扩大 T/P 范围 |
| **用基础模型直接探索** | 冷启动：先用 MP/OMAT 跑长 MD，再子采样 | 很低 | Hänseroth et al. [arXiv:2606.23214](https://arxiv.org/abs/2606.23214)（约 2000 个标注通常就够）；NRL 的迭代微调 | ✅ 第 0 轮的默认做法 |
| **微扰、应变、缺陷** | 补齐 EOS、弹性、点缺陷附近的构型 | 低 | 常规做法 | ✅ `explore perturb`（rattle / 应变 / 空位） |
| **极小值跳跃 / basin hopping** | 团簇、表面、找全局极小 | 中 | Python Minima Hopping（*SoftwareX* 2024），接受任何 ASE 计算器 | ✅ `explore minhop`（ASE + MACE，非 LAMMPS） |
| **随机结构搜索**（AIRSS / autoplex） | 多晶型、非晶、未知相空间 | 低（小胞单点计算） | autoplex, *Nat. Commun.* 16, 7666 (2025)（每轮约 100 个单点，目标约 10 meV/atom） | ⏳ `explore rss`（可选） |
| **增强采样**（metadynamics / OPES，PLUMED） | 稀有事件、反应、势垒 | 中 | *Digital Discovery* 2026, 5, 108；ERBS [arXiv:2601.16047](https://arxiv.org/abs/2601.16047) | ⏳ 只提供 `fix plumed` 模板；它和 mliap 能否配合**需要先实测** |
| **不确定度偏置 MD**（UDD / UBMD / HAL） | 普通 MD 迟迟找不到外推区时 | 中 | UDD *Nat. Comput. Sci.* 2023；UBMD *npj* 2024；HAL *npj* 2023 | ❌ LAMMPS 里没有原生的 MACE 不确定度，只能走 ASE；v0.3 再考虑 |
| **对抗攻击不确定度** | 小体系、分子 | 低 | Schwalbe-Koda et al. *Nat. Commun.* 2021 | ❌ v0.3 |
| **合成数据 / 蒸馏** | 需要大量便宜的数据，或要一个更快的学生模型 | 低 | Gardner et al. [arXiv:2307.15714](https://arxiv.org/abs/2307.15714)；[arXiv:2506.10956](https://arxiv.org/abs/2506.10956)（约 25 个 DFT 结构 → 基础模型生成合成数据 → 学生模型，快 10 倍以上） | ⏳ `train -mode distill`（可选） |
| 生成模型（扩散等） | 前沿方向 | — | 尚未找到可靠证据 | ❌ |

### 1.4 筛选：标注哪些

| 方法 | 说明 | 能否与 MACE 配合 | v0.2 |
|---|---|---|---|
| **委员会力偏差 + 信任窗口**（query-by-committee） | `Fdev_max` 落在 [lo, hi] 区间的帧送去标注；低于 lo 说明模型已经会了，高于 hi 多半是不物理的结构 | ✅ 3–5 个不同 seed 的模型对 dump 离线重新打分（ai2-kit 的做法） | ✅ 核心 |
| 信任窗口自适应 | 不用固定的 0.05–0.15 eV/Å，改用 lo = Q3 + 1.5·IQR（基于训练集上的偏差分布）；文献里固定阈值的跨度是 0.2–2 eV/Å | ✅ | ✅ `-trust auto` |
| 相对力偏差 | σ_F / (\|F̄\| + 0.2 eV/Å)，适合力很大的体系；HAL 的目标是 3–10% | ✅ | ✅ `-trust relative` |
| **多头委员会**（Beck et al. *JCP* 2025） | 共享 MACE 描述子，只训练几个额外的输出头；能量偏差几乎不花额外成本 | 代码在 MACE PR #800，尚未合并；能否用于 LAMMPS 未知 | ⏳ 合并后接入 |
| 多样性（FPS / k-means）在 MACE 描述子上 | 去重、限制每轮的数量；有研究显示多样性方法能追平甚至超过不确定度方法 | ✅ `mace_finetuning_select`、skmatter | ✅ 先按不确定度筛，再用 FPS 去重并限量 |
| **QUESTS 信息熵** | 不需要模型：衡量新结构相对训练集的"新颖度"，也衡量数据集是否已经饱和 | ✅（与模型无关，LLNL 维护） | ✅ 去重，并作为停止判据之一 |
| D-optimality / MaxVol（ACE/MTP） | LAMMPS 里在线算外推等级 γ（≈5 时输出该帧，≈25 时停止） | ❌ 只支持 ACE/MTP | 参考它的设计：dump 帧打分 + 停止阈值 |
| 末层不确定度（LLPR）、高斯混合模型（GMM）、保形校准 | 只需单个模型，成本低 | MACE 里没有现成实现 | ❌ v0.3 |

### 1.5 训练策略

| 策略 | 结论（R5 和 MACE 文档） | v0.1 | v0.2 |
|---|---|---|---|
| 朴素微调 | 单一体系收敛最快 | ✅ `finetune` | — |
| 多头回放（原始数据或伪标签） | **唯一能稳定保住分布外鲁棒性和短程排斥墙的策略**；伪标签回放几乎和原始标签一样好 | ✅ `multihead` | 伪标签回放选项 |
| LoRA | 对 MACE 比冻结层好；冻结嵌入层和第一层效果最差 | ✅ `lora` | — |
| 从头训练 | 多数情况下比微调差 | ✅ `scratch` | — |
| **E0** | 比选哪种策略更重要：使用平均 E0 时力误差会变成 2–3 倍 | ✅ isolated → estimated → average | 把 E0 冻结成 json，保证同一个主动学习活动内逐轮一致 |
| **委员会训练** | 不确定度的来源 | ❌ | ✅ `train -committee N`（每个成员换一个 seed；GPU 足够时并行） |
| 每轮再训练 | 从上一轮的模型继续微调，还是从基础模型重新微调 | — | ✅ 两种都支持，默认从基础模型重新微调（避免误差累积） |
| 多保真 / Δ 学习 | 多头交叉学习不同的 DFT 层级 | — | ⏳ 在 multihead 中按泛函分 head |
| 蒸馏到小模型 | 用于大规模生产 | — | ⏳ 可选 |

### 1.6 什么时候停

综述里没有正式的停止规则（R2）。实践中把几条判据**组合**起来用，`al status` 逐轮记录：

1. **候选比例进入平台。** 某轮里落在信任窗口内的帧占比低于约 5%，或者连续两轮不再下降（DP-GEN 的研究在约 10% 处停）。
2. **新帧上的 DFT 抽查误差达标。** 刚标注的帧（模型训练时没见过）上的误差低于目标值，比如 good 档：10 meV/atom、50 meV/Å。
3. **生产长度的 MD 通过全部闸门**，并且目标性质（密度、RDF、扩散系数……）在相邻两轮之间的变化小于设定的容差。
4. （可选）**QUESTS 熵饱和。** 新数据带来的信息增益接近 0。

满足 1、2、3 才判定"可以停"。只满足部分时，报告会说明还差哪一条。

---

## 2. 主动学习闭环：命令设计

```bash
mflow al init   -name cu_al -system start.xyz -model mpa            # 生成 campaign.yaml（探索器、筛选、预算、停止判据）
mflow al seed   -name cu_al                                         # 第 0 轮：基础模型 MD + 微扰 → FPS → 选 N 帧
mflow al export -name cu_al -round 0                                # 写给 TFlow：candidates.xyz + manifest.json
#   …… TFlow 完成 DFT ……
mflow al import -name cu_al -round 0 -in tflow_out.xyz              # 收回 → check → 合并为数据集 v1（重新构建，不追加）
mflow al train  -name cu_al -round 1 -committee 4                   # 训练委员会
mflow al explore -name cu_al -round 1                               # 按 campaign.yaml 的探索器跑（md / perturb / minhop / rss）
mflow al select -name cu_al -round 1                                # 委员会重新打分 → 信任窗口 → FPS/QUESTS 去重 → 限量
mflow al status -name cu_al                                         # 台账 + 停止判据
mflow al run    -name cu_al -until-stop                             # 自动循环到需要 TFlow 时暂停（人工闸门）
```

**`campaign.yaml` 示例：**

```yaml
system: start.xyz
foundation: mpa
mode: finetune            # 每轮的训练策略
committee: 4
explore:
  md:      {temps: [300, 600, 900, 1200], press: [0, 10000], steps: 20000, dump: 50, rounds_widen: true}
  perturb: {rattle: 0.05, strain: 0.03, vacancies: 1, n: 50}
  minhop:  {enabled: false, steps: 200}
select:
  trust: {mode: auto, lo: null, hi: null}      # auto = Q3+1.5·IQR；也可以填固定 eV/Å，或用 relative
  dedupe: fps                                   # fps | quests
  max_per_round: 150
stop:
  candidate_fraction: 0.05
  dft_spotcheck: {energy_meV_atom: 10, force_meV_A: 50}
  observables: {density: 0.01, rdf_mae: 0.05}   # 相邻两轮之间的相对变化
```

**目录与台账：**每一轮一个目录，`ledger.json` 只追加、不改写。每轮记录：模型 hash、数据集版本、各探索器产出的帧数、候选比例、选中的帧数、委员会规模、DFT 抽查误差、各闸门结论。
数据集每轮**重新构建**，测试集**固定不变**（换了测试集，两轮之间的结果就不能比较）。

**人工闸门：**`al run` 每次走到"要交给 TFlow"时暂停，显示本轮要算的帧数和预计的 DFT 成本，等用户确认后才继续。

### 2.1 与 TFlow 的接口约定

| 方向 | 文件 | 内容 |
|---|---|---|
| MFlow → TFlow | `round_N/export/candidates.xyz` | 只有几何结构。info 里包含：`mflow_id`（唯一 ID）、`mflow_campaign`、`mflow_round`、`mflow_source`（产生该帧的探索器）、`mflow_Fdev_max`、`config_type` |
| | `round_N/export/manifest.json` | 帧数、元素、需要的 DFT 设置名称（与基础模型对齐，见 README 1.1）、预算 |
| TFlow → MFlow | 任意 extxyz | 带 `mflow_id`，以及 `REF_energy` / `REF_forces` / `REF_stress`；未收敛的帧标记 `tflow_status=failed`（或不返回） |

`al import` 按 `mflow_id` 对应回去，所有帧先过 `check`；失败的帧记入台账，不进数据集。

---

## 3. 精度评估与 LAMMPS 协议（补齐 v0.1 的空缺）

### 3.1 `mflow validate`：验证第 1 层（静态性质）

| 子命令 | 内容 | 判据 / 输出 |
|---|---|---|
| `validate dimer` | 各元素对的双原子能量曲线 | 平滑；短程单调排斥；力不变号（MLIP Arena 的指标） |
| `validate eos` | E–V 曲线，拟合 a₀、B₀、B′ | 与 DFT 参考（如果提供）的相对误差 |
| `validate phonon` | phonopy + MACE，算色散 | 有没有虚频；ω_max |
| `validate defect` | 空位形成能、表面能 | 与 DFT 参考对比 |

### 3.2 误差与不确定度报告的补充

- **力误差：**
  - 按元素分开统计力误差。
  - 画 \|ΔF\| 对 \|F_DFT\| 的散点图，并给出相对力误差。大力构型会主导 RMSE，这两样能把它分开看。
- **委员会校准**（在有参考的数据上）：
  - σ 与 \|误差\| 的 Spearman 相关；
  - miscalibration area；
  - sharpness（σ 的分布）。
  - 用这三项判断这组委员会的不确定度能不能拿来做筛选。

### 3.3 LAMMPS 协议库

| 新选项 / 协议 | 内容 |
|---|---|
| `-thermostat nh\|csvr\|langevin` | Nosé–Hoover / Bussi（`fix temp/csvr` + `nve`）/ Langevin（`fix langevin` + `nve`） |
| `-baro iso\|aniso\|tri` | 立方晶体、液体用 iso；正交非立方用 aniso；三斜、剪切用 tri |
| `-protocol equilibrate` | NPT 平衡 → NVT |
| `-protocol transport` | NPT/NVT 平衡 → **NVE** 生产段，再算 MSD/扩散：检查 MSD 双对数斜率约为 1，Yeh–Hummer 修正，分块误差棒 |
| `-protocol anneal` / `quench` | 线性变温，速率 1–10 K/ps |
| `-protocol minimize` | `fix box/relax` + `minimize`，循环到晶胞不再变化 |
| `-protocol melt` | 固液两相共存，NPH 下读出 Tm；检查两相比例（固、液各占约一半），用 Q6/CNA 判断相 |
| `md-report` 补充 | 前后两半比较（平衡检验）、分块平均误差棒、VDOS、升温稳定性阶梯 |

---

## 4. 经验类 skills

代码管执行，经验写成 skill。全部按 HeshamFS 的格式：**验证清单 + 常见误区表**，SKILL.md 不超过约 300 行，具体数字放在 `references/`。
所有阈值统一写在 MFlow 自己的 `thresholds.yaml` 里，**skill 文本与代码引用的是同一份**。参考来源只改写不照抄，出处记在每个 skill 的 `references/SOURCES.md`。

| skill | 范围 | 主要参考 |
|---|---|---|
| `mflow-research-protocol` | 先判断请求属于哪类 → 建研究目录 → 写 `plan.md` 并等用户批准 → 每个数字都标明来源（模型、参数、原始输出路径）→ 先估算显存和算力 → 结论必须有经过验证的产物支撑 | AtomisticSkills `general-atomisticskills-rules`；obra/superpowers `verification-before-completion` |
| `mflow-finetune-strategy` | 根据数据量和泛函是否一致来选策略；E0 冻结成 json；学习率区间（OMAT/MH 系列用 1e-4）；15–25 个 epoch 并保留验证集最佳的 checkpoint；"遗忘"的征兆 | AtomisticSkills `ml-mace-finetune` / `ml-mlip-automl`；CatMaster `mace-finetuning-and-benchmark` 及 `vasp_to_mace_finetune.md`；R5 |
| `mflow-dataset-curation` | 按来源轨迹（source run）切分，不按帧切；固定测试集；应力的单位和符号；去重、离群；记录覆盖范围 | CatMaster `mace-dataset-curation`；AtomisticSkills `general-property-units` |
| `mflow-potential-acceptance` | "够不够用"：第 0–3 层验证；按用途设阈值（声子、熔点、扩散要求不同）；力误差低不等于 MD 稳定 | AtomisticSkills `ml-mlip-benchmark`；Morrow 2023；Fu 2023 |
| `mflow-active-learning-campaign` | 探索器的选择（MD 网格、极小值跳跃、高温采样）；委员会至少 3 个、建议 4 个；信任窗口用验证集力 RMSE 来校准；台账；停止判据 | AtomisticSkills `ml-committee-uncertainty`；CatMaster `active-learning-relabel-loop`；ai2-kit `build-tesla`；DP-GEN 文档；R1/R2 |
| `mflow-md-setup` | 系综与恒温器的选择；τ_T ≈ 100·dt；用 NVE 漂移测试收敛 dt；升温梯度；晶胞大小；用 unwrapped 坐标；"爆炸"时的排错阶梯 | AtomisticSkills `mat-md-monitors`；CatMaster `lammps-preparation`；HeshamFS `convergence-study` |
| `mflow-property-protocols` | 熔点（固液共存）、扩散（Arrhenius 拟合、MSD 斜率约为 1、Yeh–Hummer 修正、分块误差）、RDF / 配位数、非晶化 | AtomisticSkills `mat-melting-point` / `mat-diffusion-analysis` / `mat-amorphization`；HeshamFS `md-analysis-planner` |
| `mflow-failure-triage` | MD 爆炸、训练发散、LAMMPS 报 lost atoms；先保留证据；每次只改一个变量；什么时候该停下（结果只靠人为稳定手段才成立时） | HeshamFS `simulation-failure-triage` |
| `mflow-lab-notebook` | 每个活动一份只追加的 `NOTEBOOK.md`（问题、竞争假设、参数、产物路径、结论）。`LESSONS.md` 只收录满足"持久性门槛"的经验：每条一个教训（触发条件 / 做法 / 原因 / 如何检查），旧条目就地更新不重复，只收录已验证的 | EveryInc `ce-compound`；K-Dense `hypothesis-generation`；CatMaster 的台账纪律 |

许可证：AtomisticSkills、ai2-kit、superpowers、compound-engineering、K-Dense 为 MIT；CatMaster、HeshamFS 为 Apache-2.0；DeePMD 相关为 LGPL-3.0（只改写，不复制）。
目前**没有找到**覆盖极小值跳跃、MACE 的 MD 探索信任窗口、主动学习停止策略的现成 skill，这几块要根据文献自己写。

---

## 5. 里程碑与验收

| 版本 | 内容 | 验收 |
|---|---|---|
| **v0.2.0-a** | `train -committee N`；`calc` 的校准报告；力误差按元素拆分、\|ΔF\| 对 \|F\| 作图；`thresholds.yaml` | 小数据集上训练出 4 个成员的委员会；在有参考的数据上给出 Spearman 相关和 miscalibration area |
| **v0.2.0-b** | `al init / seed / explore(md, perturb) / select / export / import / train / status` | 用 EMT 或 LJ 充当"假 TFlow"，端到端跑完 3 轮；台账完整；第 3 轮的候选比例下降 |
| **v0.2.0-c** | `explore minhop`；`al run -until-stop`（含人工闸门） | 团簇体系上用极小值跳跃探索出至少 1 个新极小 |
| **v0.2.0-d** | `validate dimer / eos / phonon / defect`；LAMMPS 协议库（恒温器、恒压器、transport、minimize、anneal、melt）；`md-report` 补充 | Cu（EAM 或 MACE）上的 EOS 与熔点协议都能跑通；transport 协议的生产段是 NVE |
| **v0.2.0-e** | 9 个经验类 skill + `NOTEBOOK.md` / `LESSONS.md` 模板 | 在 pi + DeepSeek/GLM 上完成一个小型主动学习活动，并整理成回归测试集 |

## 6. v0.2 明确不做

- DFT 输入生成、提交和解析（属于 TFlow）。
- 集群作业提交（需要时再接 dpdispatcher）。
- LAMMPS 内在线计算 MACE 不确定度（上游不支持，见调研报告）。
- 不确定度偏置 MD、对抗攻击、生成模型（v0.3 再评估）。

## 7. 实现状态（v0.2.0a1）与规划的差异

| 规划 | 实现 | 说明 |
|---|---|---|
| `al seed` | 并入 `al explore` / `al run` 的第 0 轮 | 第 0 轮没有委员会时自动只做多样性筛选（FPS），不需要单独的命令 |
| `campaign.yaml` 的 `trust: {mode, lo, hi}` | `select: {trust, lo, hi, recalibrate, dedupe, max_per_round}` | 增加 `recalibrate`：按测试集 RMS(误差)/RMS(σ) 放大 σ（微调委员会普遍过度自信）；`auto` 在委员会没收敛时退回 fixed；全部帧都 failed 时从 failed 池里按多样性选（模型还没准备好，而不是结构不物理） |
| 去重 `fps \| quests` | 只有 `fps`（MACE 描述子，相对已有训练集） | QUESTS 熵作为去重和停止判据：⏳ 下一步 |
| 停止判据 1–4 | 1（候选比例）、2（新帧误差，用委员会算）已实现 | 3（生产 MD 闸门 + 目标性质收敛）目前靠用户跑 `md` 后判断；4（QUESTS）⏳ |
| TFlow 返回 extxyz | 直接读 TFlow 的 `job_map.json` + `static/OUTCAR`（也接受 extxyz） | TFlow 没有 MLIP 导出命令；MFlow 侧做拒收规则：非 done、error fix、KPOINTS 被改、SCF 达到 NELM、结构改变 |
| `explore rss`、`fix plumed` 模板、`train -mode distill` | ⏳ | |
| `validate` 弹性常数 | ⏳ | dimer / eos / phonon / defect 已实现；dimer 用 MLIP Arena 的网格和指标 |
| 升温稳定性阶梯 | 用 `-protocol anneal` 代替 | |
| melt 协议 | 实现时改了两处 | 熔化半盒在**固定体积**下进行（只让液相做 NPT 会让盒子失控膨胀）；相判断用 ten Wolde 固相键数（原始 q6 分不清高温晶体和液体），用 Sutton–Chen Cu 的共存测试核对过 |

**建议 TFlow 侧补充（未改动 TFlow）：** `vaspctl export-mlip`（直接写带 `REF_*` 的 extxyz）；static 任务记录 SCF 是否收敛；为 MLIP 标注提供"禁止改结构的 error fix"开关。

---

下一阶段（主动学习微调 foundation 模型）的方法调研见 [research-al-2026-10.md](research-al-2026-10.md)：探索方法、委员会/UQ、收敛判据、可复用项目，以及 v0.3 建议路线。
