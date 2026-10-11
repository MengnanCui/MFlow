# MFlow

**MACE 势函数的一条龙工具：训练 → 评估 → 推理 → LAMMPS 动力学 → 质量判定。**
当前版本 **v0.2.0a1**（Python 包 `mflow/`，命令 `mflow`）：v0.1 的训练/推理/MD 之上，加了委员会训练与不确定度校准、主动学习闭环（接 TFlow）、`validate` 静态性质、LAMMPS 协议库。设计与依据见 [v0.2 路线图](docs/roadmap-v0.2.md)。

两条设计原则：

1. **执行交给代码，结果可以复查。** 每个命令都写一份 JSON 报告：`status` 是 `pass / warn / fail` 之一，附带逐项检查和建议。判定标准写在代码里，不靠人或大模型临场判断。
2. **经验写进 skill。** "该选哪种微调""这个误差够不够""用什么系综"这类判断放在 [`skills/`](skills/)。pi 之类的通用 agent 读 skill、调用 `mflow`、读 JSON，就能走完全流程。

数据生成（DFT 计算）不在本项目范围内，由 TFlow 负责。MFlow 的起点是一个**已经带 DFT 标签的 xyz 文件**。

**第一次来：** 先看 [A. 我能帮你做什么](#a-我能帮你做什么功能总览) 找到你的目标，再照 [B. 场景配方](#b-按目标走10-个常见场景) 走；各章是细节，[第八章](#八命令速查)是命令速查。

---

## 目录

- [安装](#安装)
- [五分钟走完全程](#五分钟走完全程)
- [**A. 我能帮你做什么（功能总览）**](#a-我能帮你做什么功能总览)
- [**B. 按目标走：10 个常见场景**](#b-按目标走10-个常见场景)
- [一、训练](#一训练)：数据与标签 · 体检 · 选方案 · 训练与监控 · **精度怎么看** · 评估方案 · 静态性质 `validate`
- [二、推理](#二推理)：标签 · 精度 · 不确定度（委员会）· 弛豫
- [三、LAMMPS 动力学](#三lammps-动力学)：导出 · 环境检查 · **输入如何生成** · **系综怎么选** · 质量闸门 · 可参考的 skill
- [四、主动学习：`mflow al`](#四主动学习mflow-al)：探索 → 筛选 → TFlow 标注 → 收回 → 委员会再训练 → 何时停
- [五、给 agent 用：skills](#五给-agent-用skills)：12 个 skill · 跟 agent 怎么说
- [六、输出文件与约定](#六输出文件与约定)
- [七、版本](#七版本)
- [八、命令速查](#八命令速查)
- [九、常见问题](#九常见问题)
- [十、能力边界与已知限制](#十能力边界与已知限制)

---

## 安装

装好之后，所有功能都是一个命令 **`mflow`**，在任何目录下都能用，不需要 `python xxx.py`。

**第 0 步（只做一次）：装与 CUDA 匹配的 torch。** 这一步没法替你自动选，见 [pytorch.org](https://pytorch.org)，例如：

```bash
conda create -n mflow python=3.11 -y && conda activate mflow       # 建议单独一个环境
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**第 1 步：三选一。**

```bash
# A. 一键（推荐）：克隆后在仓库里执行。装 mflow + mace-torch 0.3.16 + phonopy，链接 skill，最后自检
git clone https://github.com/MengnanCui/MFlow.git && cd MFlow
bash install.sh                  # --all 再加 cuequivariance-torch、pytest；--completion 加 Tab 补全；--no-skills 不装 skill

# B. 不克隆，直接装成命令（要更新时重跑同一行；装某个分支在 .git 后加 @分支名）
pip install "mflow[mace,validate] @ git+https://github.com/MengnanCui/MFlow.git"

# C. 开发模式：改了代码立刻生效
pip install -e ".[all]"
```

**第 2 步：自检 + 记住你的默认设置。**

```bash
mflow doctor                                   # 缺什么、哪些命令受影响、怎么装，一次列清
mflow config set model ~/models/my.model       # 以后 calc / relax / check / validate 不用每次写 -model
mflow config set lmp ~/lammps/build/lmp        # 你自己编译的 LAMMPS（KOKKOS + ML-IAP）
mflow skills install                           # 把 12 个 skill 链接到 ~/.agents/skills，pi 启动时自动加载
eval "$(mflow completion bash)"                # Tab 补全（zsh 用 mflow completion zsh；写进 ~/.bashrc 就永久生效）
mflow                                          # 不带参数：按用途分组列出所有命令
```

- 配置文件在 `~/.config/mflow/config.yaml`（`mflow config path`）；环境变量 `MFLOW_MODEL` / `MFLOW_LMP` 优先于它，命令行的 `-model` / `-lmp` 再优先。
- 没设默认模型时用 MACE-MPA-0（`mpa`，第一次用时由 mace 自动下载）。
- 缺可选依赖时，命令会直接告诉你装哪个包，不会只抛 traceback。
- **LAMMPS** 需要另外从源码编译，打开 KOKKOS(CUDA)、ML-IAP、ML-SNAP、PYTHON、`MLIAP_ENABLE_PYTHON`，见 [3.1](#31-导出与环境检查)。
- **`mace-torch` 固定为 0.3.16**：上游正在重写 v1.0，命令行参数会变。
- 测试：`pytest`（64 个；装了 mace-torch 会真训练，找得到 `lmp` 会真跑 LAMMPS 的 LJ / EAM 算例）。

## 五分钟走完全程

```bash
mflow doctor                                                                 # 0 环境就绪？
mflow check -in dft.xyz -model mpa -split 0.1 -outdir check                  # 1 体检 + 零样本 + 切测试集
mflow train -in check/dft_train.xyz -test check/dft_test.xyz -name cu -dry   # 2 先看 config.yaml
mflow train -in check/dft_train.xyz -test check/dft_test.xyz -name cu        # 3 训练 + 实时监控 + 报告
mflow calc  -in new_dft.xyz -model cu/cu.model -e0 none -group config_type   # 4 在新数据上评估
mflow validate dimer -model cu/cu.model -elements Cu                         # 5 短程排斥墙是否完整
mflow deploy -model cu/cu.model -outdir deploy                               # 6 导出给 LAMMPS（在 GPU 上）
mflow lmp-check -deploy deploy                                               # 7 LAMMPS 能不能跑它
mflow md -in start.xyz -deploy deploy -protocol equilibrate -temp 600 -steps 200000 -outdir md600   # 8 MD + 闸门
```

每一步结束都会打印 `verdict PASS / WARN / FAIL`。**出现 FAIL 就先看建议，不要往下走。**

---

## A. 我能帮你做什么（功能总览）

按"你想做的事"列出。最后一列是**目前验证到什么程度**，请据此决定用到多正式的场合：

- ✅ **已用真实 mace-torch 测过**（CPU 上的小模型，见 `tests/`）
- 🧪 **流程已测，物理部分用替身**：MD 用 LAMMPS 的 LJ / Sutton–Chen EAM 代替 MACE；`validate` 用 ASE 的 EMT；主动学习用"假 TFlow"目录
- ⚠️ **还没在 GPU + 真实 MACE/LAMMPS 上跑过**：第一次用请先跑 `mflow lmp-check`，把结果和小体系的手算对一下

### 安装与设置

| 你想… | 命令 | 你会得到 | 验证 |
|---|---|---|---|
| 看看有哪些命令 | `mflow` | 按用途分组的命令总览 | ✅ |
| 检查机器缺什么 | `mflow doctor` | Python 包、GPU、默认模型、LAMMPS、skill、阈值逐项 pass/warn，附安装命令 | ✅ |
| 不想每次都写 `-model` / `-lmp` | `mflow config set model …`、`mflow config set lmp …` | 写进 `~/.config/mflow/config.yaml` | ✅ |
| 让 pi 用上 MFlow 的经验 | `mflow skills install` | 12 个 skill 链接到 `~/.agents/skills`（`-dest` 换位置、`-copy` 复制） | ✅ |
| Tab 补全命令和参数 | `eval "$(mflow completion bash)"` | 子命令、选项补全，参数值处补全文件名 | ✅ |
| 体系专属的小工具 | `mflow tool`、`mflow tool izo-formation …` | 例如 In–Zn–O 形成能随 In 比例 | ✅ |

### 数据与训练

| 你想… | 命令 | 你会得到 | 验证 |
|---|---|---|---|
| 检查一批 DFT 数据能不能拿来训练 | `mflow check -in dft.xyz` | 每项 pass/warn/fail：标签缺失、NaN、力过大、原子过近、重复帧、能量离群、E0、分组规模；坏帧的索引 | ✅ |
| 看基础模型（MPA-0 / OMAT / MH-1）在你的数据上准不准、DFT 设置对不对得上 | `mflow check -in dft.xyz -model mpa` | 零样本误差；原始能量偏移（> 0.4 eV/atom 说明 DFT 设置与基础模型不一致） | ✅ |
| 切出测试集 | `mflow check -in dft.xyz -split 0.1` | 按 `config_type` 分层的 `*_train.xyz` / `*_test.xyz` | ✅ |
| 微调 / 多头回放 / LoRA / 从头训练 | `mflow train -in train.xyz -test test.xyz -name run -mode finetune` | MACE 官方推荐超参的 `config.yaml`；实时进度条（误差、GPU/显存）；NaN 立即停；训练报告 | ✅ |
| 先看配置再决定 | `mflow train … -dry` | 只写 `config.yaml` | ✅ |
| 改某个超参 | `mflow train … -set lr=5e-4 patience=100` | 键名按 `mace_run_train --help` 校验，写错直接 fail | ✅ |
| 训练中断了 | `mflow train … -resume` | 从最新 checkpoint 续 | ✅ |
| 一次训练一组委员会（要不确定度） | `mflow train … -committee 4` | 4 个成员（不同 seed + 每人 80 % 数据）、测试集上的误差和**校准报告** | ✅ |
| 判断训练好不好 | `mflow train-report -dir run` | 学习曲线、过拟合判断、测试集误差（按组、按元素、最差 5 个结构）、分级 | ✅ |

### 评估、推理与不确定度

| 你想… | 命令 | 你会得到 | 验证 |
|---|---|---|---|
| 用模型预测一批结构 | `mflow calc -in x.xyz -model run/run.model` | 能量、力写回 xyz（不覆盖已有标签） | ✅ |
| 和 DFT 比误差 | `mflow calc … -ref dft_ -e0 none -group config_type` | MAE/RMSE/max/R²，按组、**按元素**、按力大小分档；parity 图 | ✅ |
| 知道模型对哪些结构"拿不准" | `mflow calc -in md.xyz -model a.model,b.model,c.model` | 每个原子的力偏差、每个结构的 `Fdev_max` | ✅ |
| 判断委员会的不确定度可不可信 | 同上，数据带 DFT 标签 | Spearman ρ、miscalibration area、RMS(误差)/RMS(σ)、校准图 | ✅ |
| 批量弛豫 | `mflow relax -in x.xyz -nproc 3` | 多进程、可续跑，未收敛的单独标出 | ✅ |
| 只重画图 | `mflow plot -in x_mace.xyz` | 不重算 | ✅ |

### 静态性质（MD 之前先过这一关）

| 你想… | 命令 | 你会得到 | 验证 |
|---|---|---|---|
| 查短程排斥墙有没有"洞"（MD 塌缩的头号原因） | `mflow validate dimer -model m -elements Cu,O` | 每个元素对的 E(r)/F(r)，MLIP Arena 的指标；墙坏了判 fail | 🧪 |
| 晶格常数、体模量 | `mflow validate eos -model m -in relaxed.xyz -ref "V0=..,B0=.."` | V₀、B₀、B′，与 DFT 的相对误差 | 🧪 |
| 声子有没有虚频 | `mflow validate phonon -model m -in relaxed.xyz` | 色散 + DOS 图、最低频率、虚频 q 点数（需要 phonopy） | 🧪 |
| 空位形成能、表面能 | `mflow validate defect -model m -in conv.xyz -surface 111,100` | E_f（单质）、γ（J/m²） | 🧪 |

### LAMMPS 动力学（MACE 经 ML-IAP + KOKKOS 在 GPU 上）

| 你想… | 命令 | 你会得到 | 验证 |
|---|---|---|---|
| 把模型交给 LAMMPS | `mflow deploy -model m` | `deploy.json`（元素顺序、head、精度、hash） | ⚠️ |
| 确认 LAMMPS 编译对了、依赖齐了 | `mflow lmp-check -deploy deploy` | 包、内嵌 Python 依赖、2 原子实跑，逐项 pass/fail | ⚠️ |
| 跑 NVT / NPT / NVE | `mflow md -in s.xyz -deploy deploy -ensemble npt -temp 600` | 输入自动生成 → 语法检查 → **LAMMPS 与模型逐原子力对比** → 运行 → 质量闸门 | 🧪 |
| 先平衡再取样 | `-protocol equilibrate` | NPT → 平均体积 → NVT | 🧪 |
| 算扩散系数 / VDOS | `-protocol transport` | 生产段 NVE；D（含 x/y/z）、扩散区判断、VDOS、可选 Yeh–Hummer 修正 | 🧪 |
| 退火、淬火做非晶 | `-protocol anneal` / `quench` | 变温速率 K/ps | 🧪 |
| LAMMPS 里弛豫晶胞 | `-protocol minimize` | `relaxed.xyz`、最大力、压强 | 🧪 |
| 算熔点 | `-protocol melt` | 固液共存的 T_m ± 误差、液相比例 | 🧪（Sutton–Chen Cu 上核对过物理） |
| 换恒温器 / 恒压方式 | `-thermostat csvr`、`-baro aniso` | | 🧪 |
| 中断后续跑 | `mflow md … -resume` | 按阶段续 | 🧪 |
| 重新分析一条轨迹 | `mflow md-report -dir run -ref aimd.xyz` | 闸门、分块误差、平衡检验、RDF（含与 AIMD 对比）、MSD、VDOS | 🧪 |

### 主动学习（DFT 由 TFlow 算）

| 你想… | 命令 | 你会得到 | 验证 |
|---|---|---|---|
| 开一个活动 | `mflow al init -name camp -system start.xyz -data dft.xyz -test test.xyz` | `campaign.yaml`、台账 | 🧪 |
| 自动走到"需要 DFT"为止 | `mflow al run -name camp` | 探索（MD / 微扰 / 极小值跳跃）→ 委员会筛选 → 导出给 TFlow 的文件和命令 | 🧪 |
| 把 TFlow 的结果收回来 | `mflow al import -name camp -tflow <calc_dir>` | 合格的标注进数据集；不合格的（SCF 不收敛、错误修复改过结构……）列出原因 | 🧪 |
| 看进度、要不要停 | `mflow al status -name camp` | 每轮的探索/候选/标注数、新帧误差、停止判据 | 🧪 |

### 让 agent 帮你做

| 你想… | 怎么做 | 验证 |
|---|---|---|
| 用 pi 之类的 agent 走全流程 | 安装 `skills/`（见第五章），直接说目标 | 格式测试 ✅；与 pi 的实际对话 ⚠️ |
| 让 agent 帮你选方案、判断够不够用、排错 | 9 个经验类 skill | ⚠️ |
| 留下可复查的记录 | `mflow-lab-notebook`：`NOTEBOOK.md` + `LESSONS.md` | — |
| 按自己的体系改判定标准 | 写 `./mflow_thresholds.yaml`（只写要改的键），或设 `MFLOW_THRESHOLDS` | ✅ |

---

## B. 按目标走：10 个常见场景

每个场景给出最短路径。出现 FAIL 先看报告里的建议；对应的经验类 skill 写在标题后面。

### B1. 我有一批 DFT 数据，想得到一个能跑 MD 的势（`mflow-finetune-strategy`）

```bash
mflow check -in dft.xyz -model mpa -split 0.1 -outdir check            # 数据体检 + 零样本 + 切测试集
mflow train -in check/dft_train.xyz -test check/dft_test.xyz -name cu -committee 4 -dry
mflow train -in check/dft_train.xyz -test check/dft_test.xyz -name cu -committee 4
mflow validate dimer -model cu/m0/cu_m0.model -elements Cu               # 短程墙
mflow deploy -model cu/m0/cu_m0.model -outdir deploy && mflow lmp-check -deploy deploy
mflow md -in start.xyz -deploy deploy -ensemble nve -temp 300 -steps 20000 -outdir nve_test   # 守恒性
```

判断：训练报告的分级 → dimer 无 fail → NVE 漂移通过。三关都过再进入生产。只要一个模型时去掉 `-committee 4`。

### B2. 只想知道基础模型对我的体系准不准

```bash
mflow check -in dft.xyz -model mpa            # 或 omat / mh1（mh1 加 -head omat_pbe）
mflow calc  -in dft.xyz -model mpa -group config_type
```

看 `energy (E0 aligned)` 和力的误差；原始偏移很大但对齐后很小，说明只是 DFT 参考零点不同，微调能轻松修正。

### B3. MD 崩了（原子飞走、温度失控、原子重叠）（`mflow-failure-triage`）

```bash
mflow md -in start.xyz -deploy deploy -dt 0.5 -outdir md_dt05 ...         # 1 先减半步长（新目录）
mflow validate dimer -model run.model -elements A,B                       # 2 查短程墙
mflow calc -in md_dt05/traj.lammpstrj -index ::10 -model m0.model,m1.model,m2.model   # 3 委员会给崩之前的帧打分
```

偏差高 → 模型在外推 → 走 B4 补数据。偏差低还崩 → 查起始结构和导出（`lmp-check`）。

### B4. 让模型覆盖它还不会的构型（高温、表面、缺陷）（`mflow-active-learning-campaign`）

```bash
mflow al init -name camp -system starts.xyz -data dft.xyz -test test.xyz -committee 4
mflow al run -name camp                       # → round_00/export/：交给 TFlow
mflow al import -name camp -tflow <calc_dir>
mflow al run -name camp                       # 训练 → 探索 → 筛选 → 导出下一轮
mflow al status -name camp                    # 每轮看一次
```

编辑 `camp/campaign.yaml` 调探索温度、是否开极小值跳跃、信任窗口、每轮数量。

### B5. 算熔点（`mflow-property-protocols`）

```bash
# 正交晶胞，沿 z 拉长（Lz ≈ 2–4 Lx），几千原子
mflow md -in slab_z.xyz -deploy deploy -protocol melt -temp 1300 -melt_temp 2000 -steps 200000 -outdir tm1300
mflow md -in slab_z.xyz -deploy deploy -protocol melt -temp 1400 -melt_temp 2000 -steps 200000 -outdir tm1400
```

`melt: phases` pass 时 `T_melt_estimate` 就是结果；全熔了 → 降 `-temp`，全结晶 → 升 `-temp`。两个起点收敛到同一温度才算数。

### B6. 算扩散系数和活化能

```bash
for T in 800 900 1000 1100; do
  mflow md -in liq.xyz -deploy deploy -protocol transport -temp $T -steps 500000 -outdir D_$T
done
```

每个温度看 `diffusive regime` 是否 pass、`diffusion_cm2_s` 的值；多个 seed（`-seed`）给误差。Arrhenius 拟合目前需要你（或 agent）把各目录的 `md_report.json` 汇总后自己拟合。

### B7. 做非晶结构

```bash
mflow md -in liquid.xyz -deploy deploy -protocol equilibrate -temp 3000 -steps 20000 -outdir melt
mflow md -in melt/traj.lammpstrj -deploy deploy -protocol quench -ensemble npt -temp 3000 -temp_end 300 -steps 300000 -outdir quench
mflow md -in quench/traj.lammpstrj -deploy deploy -protocol minimize -outdir relaxed
```

报告给出淬火速率（K/ps）；至少比较两个速率。`-in` 可以直接给上一步的 `traj.lammpstrj`，默认取最后一帧（`-index`）。

### B8. 晶格常数、体模量、声子、缺陷能

```bash
mflow md -in cu.xyz -deploy deploy -protocol minimize -baro aniso -outdir relax   # 或 mflow relax
mflow validate eos    -model run.model -in relax/relaxed.xyz -ref "V0=11.8,B0=140"
mflow validate phonon -model run.model -in relax/relaxed.xyz -supercell 4
mflow validate defect -model run.model -in cu_conventional.xyz -surface 111,100
```

### B9. 判断这个势能不能用于发表 / 生产（`mflow-potential-acceptance`）

四层逐一过：测试集误差（每个元素都看）→ `validate` → 目标条件下的 MD 闸门 → **从生产轨迹抽 20–50 帧让 TFlow 复算**（`al export/import` 可以直接复用）。没做的层在结论里写"未验证"。

### B10. 批量弛豫几千个结构

```bash
mflow relax -in candidates.xyz -model run.model -nproc 3 -fmax 0.02
mflow relax -in candidates.xyz -model run.model -nproc 3 -fmax 0.02 -resume   # 中断后
```

按输出里的 `<p>converged` 筛选，不要按 `<p>fmax`。

---

## 一、训练

### 1.1 数据与标签：输入要长什么样

输入是 **extxyz**，一个文件放多个结构。每个结构需要以下内容：

| 内容 | 放在哪 | 单位 | 说明 |
|---|---|---|---|
| 总能量 | `info["<前缀>energy"]` | eV | **总能量**，不是每原子能量 |
| 力 | `arrays["<前缀>forces"]` | eV/Å | 每个原子 3 个分量 |
| 应力（可选） | `info["<前缀>stress"]` | eV/Å³，ASE 符号约定 | 3×3 或 Voigt 6 分量；要么所有结构都有，要么都没有 |
| 构型类别（推荐） | `info["config_type"]` | — | `bulk` / `surface` / `liquid` 等；用于按组统计误差和分层切分 |
| 孤立原子（推荐） | `config_type="IsolatedAtom"` 的单原子结构 | eV | 用同一套 DFT 设置算出的每个元素的孤立原子能量，也就是 E0 |

**前缀会自动识别**，依次尝试：无前缀（`energy`）、`REF_`、`ref_`、`dft_`、`DFT_`、`pbe_`、`scan_`。也可以用 `-ref dft_` 指定。
训练前，MFlow 把识别到的标签**复制**成 MACE 的默认键 `REF_energy / REF_forces / REF_stress`，写到 `<run>/data/`，原文件不动。

**DFT 标签最容易出错的三件事：**

1. **应力的单位和符号。** VASP 的 OUTCAR 和 vasprun.xml 输出 kBar，而且符号与 ASE 相反。
   - 用 `ase.io.read` 读出来的值已经转好了（eV/Å³，ASE 符号）。
   - 自己解析 OUTCAR 时要乘 −1/1602.18；否则应力大约 160 倍，训练会发散。
   - 快速自检：压缩过的晶胞，应力对角元应当为负。
2. **DFT 设置与基础模型一致。** 微调基础模型时，你的 DFT 设置最好和它的训练数据一致；否则会出现每原子几百 meV 的系统偏移：

   | 基础模型 / head | 它的训练数据 | 你的 VASP 最好用 |
   |---|---|---|
   | MACE-MP-0 / MPA-0 | MPtrj / sAlex，PBE(+U) | pymatgen `MPStaticSet`（ISMEAR 改为 0） |
   | MACE-OMAT-0、MH-1 的 `omat_pbe` head | OMat24（PBE，PBE_54 赝势） | `OMat24StaticSet`（fairchem-data-omat） |
   | MATPES 系列、MH-1 的 `matpes_r2scan` head | MatPES（PBE 或 r2SCAN，不加 U） | `MatPESStaticSet` |

   - 用 `mflow check -model ...` 看**零样本原始偏移**。超过 0.4 eV/atom，基本就是 DFT 设置不一致。
   - 处理办法：给每个元素补 IsolatedAtom；或者训练时用 `E0s=estimated`，同时考虑多头微调。
3. **不同泛函不能混。** PBE 和 PBE+U、PBE 和 r2SCAN 的标签不能放进同一个 head。要么分开训练，要么用多头。

### 1.2 体检：`mflow check`

```bash
mflow check -in dft.xyz                               # 只做不需要模型的检查
mflow check -in dft.xyz -model mpa -split 0.1         # + 基础模型零样本 + 按 config_type 分层切出 10% 测试集
mflow check -in train.xyz -test test.xyz              # + 测试集标签检查、训练集与测试集是否重叠
```

| 检查 | 判据 | 不通过时的建议 |
|---|---|---|
| 标签完整 | 每帧都有能量和力 | 补齐或删掉 |
| 数值健康 | NaN/inf → fail；单个力分量 > 50 eV/Å → warn，> 200 → fail | 多半是 SCF 不收敛或原子重叠 |
| 最近原子对 | < 0.6 Å → fail | 结构坏了 |
| 重复帧 | 几何指纹相同 | 去重，否则会在训练/验证/测试集之间泄漏 |
| 能量离群 | 对"逐元素线性拟合"的残差超过 6 MAD（先剔除粗大离群，再重拟合） | 多半是 DFT 设置混用；但表面、团簇本来就离得远，先看 config_type |
| E0 | 每个元素是否都有 IsolatedAtom | 没有就用 `--E0s estimated` |
| 零样本（`-model`） | 是否覆盖所有元素、误差多少、原始偏移是否超过 0.4 eV/atom | 换基础模型，或检查 DFT 设置 |
| 规模与分组 | 总帧数 < 50 → 提示；某组 < 10 → 提示 | 补该类构型 |
| 测试集 | 有没有独立测试集；和训练集是否重叠 | 用 `-split` 或 `-test` |

- 报告：`<stem>_check.json`，其中 `flags` 列出每类坏帧的索引。
- 图：`<stem>_check.png`，三格分别是每原子能量分布、力分布、零样本 parity。

### 1.3 选训练方案

| 情况 | `-mode` | `-foundation` | 说明 |
|---|---|---|---|
| 只关心一个体系，几十到几千帧 | **`finetune`**（默认） | `mpa`（MIT 许可）/ `mh1` / `omat` | 收敛最快 |
| 还要保住在其他体系上的泛化能力 | `multihead` | `mh1` / `mpa` | 回放预训练数据（`-pt`，默认 `mp`，需要联网下载），能保住短程排斥、减少"遗忘" |
| 数据很少，想少改动 | `lora` | 同上 | 对 MACE，LoRA 比冻结层效果好 |
| 基础模型没有你的元素，或数据超过约 5k 帧 | `scratch` | — | 最后 25% 的 epoch 进入 stage two |

超参采用 MACE 官方微调指南的推荐值：

| mode | lr | EMA | clip | 能量 : 力 权重 |
|---|---|---|---|---|
| finetune | 1e-3 | 0.999 | 1.0 | 1 : 10，全程不变 |
| multihead | 1e-4 | 0.9999 | 1.0 | 1 : 10 |
| lora | 1e-2 | 0.99 | 10 | 1 : 10 |
| scratch | 1e-2 | 0.99 | 10 | 1 : 100，之后进入 stage two |

- 比"选哪种方案"影响更大的是：**基础模型的质量、E0 是否正确、超参是否合理**（Tompa et al. 2026, arXiv:2606.12704）。
- 指南没给推荐值的参数（epoch、batch、stress 权重、r_max）用保守默认值。可以用 `-set key=value` 覆盖；键名按 `mace_run_train --help` 校验，不合法直接 fail。

### 1.4 训练与实时监控：`mflow train`

```bash
mflow train -in train.xyz -test test.xyz -name cu -dry                  # 只写 <run>/config.yaml，先看
mflow train -in train.xyz -test test.xyz -name cu                       # 训练
mflow train -in train.xyz -name cu -mode multihead -foundation mh1      # 多头回放
mflow train -in train.xyz -name cu -set lr=5e-4 max_num_epochs=30       # 覆盖超参
mflow train -in train.xyz -name cu -resume                              # 中断后续跑
```

`train` 按以下顺序执行：

1. **门槛检查。** 规则与 `check` 相同；有 fail 就不训练（`-force` 可以跳过）。
2. **准备数据。** 复制进 run 目录，统一标签键名。
3. **选 E0。** 每个元素都有孤立原子时用孤立原子；否则微调用 `estimated`，从头训练用 `average`。
4. **生成配置。** 写 `config.yaml`，并校验每个键名。
5. **启动 `mace_run_train` 并实时跟踪** `results/*_train.txt`：
   - 进度条显示 epoch、验证集 E/F RMSE，以及 GPU/CPU/内存占用。
   - loss 出现 NaN 立即停止，判 fail。
   - 验证 loss 连续 5 次上升，告警。
   - 显存超过 90%，告警并建议降低 `-batch`。
6. **出报告。** 结束后自动执行 `train-report`。

run 目录可以完整复现：`config.yaml`、`data/`、`mace_train.out`，以及 MACE 自己写出的 `results/ logs/ checkpoints/ <name>.model`。

### 1.5 精度怎么看：`mflow train-report`

精度按四层给出，从"训练有没有出问题"一直到"在你关心的结构上准不准"：

| 层 | 看什么 | 在哪里 |
|---|---|---|
| ① 训练过程 | 学习曲线：train/valid 的 loss，valid 的 E/F RMSE 随 epoch 的变化；标出最佳 epoch | `<run>_learning_curve.png`；报告里的 `curves` |
| ② 训练是否健康 | 无 NaN；过拟合（最终验证 loss 超过最佳值 1.5 倍时告警）；训练前后验证误差改善的倍数 | `checks` |
| ③ 测试集误差 | 能量（meV/atom）、力（meV/Å）的 MAE / RMSE / 最大误差 / R²；**按 config_type 拆分**，从差到好排序；**最差的 5 个结构** | `<run>_test_summary.png`；`test.errors`、`test.by_group` |
| ④ 分级 | excellent：能量 < 5 meV/atom 且力 < 20 meV/Å；good：< 10 且 < 50；其余为 needs work | `test.grade` |

- 测试集误差与 `mflow calc` 用**完全相同**的定义（见 [2.2](#22-推理精度)），所以训练报告和日后的评估可以直接比。
- 测试集与训练集来自同一套 DFT，E0 已经在模型里，所以测试集误差**不做 E0 拟合**（`E0 none`）。

> ⚠️ **力误差低不等于 MD 稳定**（Fu et al., arXiv:2210.07237）：在他们的基准里，力误差最低的模型最先在 MD 中崩溃。
> 训练报告只是第一关，后面还要过第三部分的 MD 闸门。

### 1.6 精度评估的系列方案

验证类文献（Morrow, Gardner & Deringer, *J. Chem. Phys.* 158, 121501 (2023)；LAMBench；MLIP Arena）里的检查可以归成四层。MFlow 目前的覆盖情况：

| 层 | 内容 | MFlow |
|---|---|---|
| **0 误差** | 测试集 E/F 的 RMSE、MAE；按组、**按元素**拆分；\|ΔF\| 对 \|F\|（大力构型单独看）；相对力误差 | `train-report` / `calc -group` |
| **1 静态性质** | 双原子曲线（平滑、短程排斥）、E–V/EOS（V₀、B₀、B′）、声子（有无虚频）、空位/表面能 | `mflow validate dimer / eos / phonon / defect`（见 1.7） |
| **2 动力学** | NVE 能量漂移、平衡检验、与 AIMD 的 RDF 对比、扩散（含扩散区判断）、VDOS | `md` 协议 + `md-report` 闸门（见 3.3–3.4） |
| **3 应用相关** | 从**你自己的生产轨迹**抽帧送 DFT 复算（Morrow 等最推荐的一条）；直接算目标性质 | `mflow al`（见第四章），TFlow 做 DFT |

### 1.7 静态性质：`mflow validate`

```bash
mflow validate dimer  -model cu/cu.model -elements Cu,O                        # 每个元素对的双原子曲线
mflow validate eos    -model cu/cu.model -in cu_relaxed.xyz -ref "V0=11.8,B0=140"
mflow validate phonon -model cu/cu.model -in cu_relaxed.xyz -supercell 4       # 需要 pip install phonopy seekpath
mflow validate defect -model cu/cu.model -in cu_conv.xyz -surface 111,100      # 空位形成能、表面能
```

| 子命令 | 怎么做 | 判据（`thresholds.yaml` 的 `validate` 段） |
|---|---|---|
| `dimer` | MLIP Arena 的网格（0.9·r_cov → 3.1·r_vdW，0.01 Å），扣孤立原子能量 | 短程墙有"洞"或 E(r_min) ≤ E(∞) → **fail**（MD 会塌缩）；力变号次数 ≠ 1、多余极值、能量跳变、\|F + dE/dr\| > 0.05 eV/Å、尾部不归零 → warn |
| `eos` | ±4 % 线应变 11 个体积，每个体积内弛豫原子位置，Birch–Murnaghan 拟合 | 极小值在采样范围内；E(V) 凸；与 `-ref` 相对误差 < 10 % pass、< 25 % warn |
| `phonon` | 先在固定晶胞下对称弛豫（fmax 1e-3），phonopy 有限位移 0.01 Å，Γ 中心 q 网格 | 低于 −0.1 THz 的频率算虚频；出现即 warn，低于 −0.5 THz fail |
| `defect` | 晶胞扩到每个方向 ≥ 10 Å；去一个原子后弛豫；`ase.build.surface` 建板 | 单质给 E_f = E(N−1) − (N−1)/N·E(N)；化合物只给原始差值（形成能需要化学势） |

所有命令都会先检查输入是否已经弛豫（|F| < 0.05 eV/Å、|σ| < 0.5 GPa），没弛豫会 warn：先 `mflow relax` 或 `mflow md -protocol minimize`。
`-model emt` 用 ASE 的 EMT 代替 MACE，方便没有模型时先试命令。

---

## 二、推理

### 2.1 推理与输出标签：`mflow calc`

```bash
mflow calc -in data.xyz -model cu/cu.model                  # 预测
mflow calc -in data.xyz -model cu/cu.model -ref dft_ -e0 none   # 与同一套 DFT 对比
mflow calc -in data.xyz -model mpa -group config_type       # 基础模型（在线下载），误差按组拆分
mflow calc -in data.xyz -model a.model,b.model,c.model      # 委员会：均值 + 不确定度
mflow plot -in data_mace.xyz -ref dft_ -group formula       # 只重画，不重算
```

**写进输出 xyz 的标签**（`<p>` 为前缀，默认 `mpa0_`）。输入里已有的标签**全部保留**；前缀冲突时自动顺延为 `mpa0_c1_`，不会覆盖：

| key | 含义 |
|---|---|
| `<p>energy`、`<p>forces` | 预测的总能量（eV）和力（eV/Å） |
| `<p>dE` | 与参考值之差，meV/atom，带符号，已做 E0 对齐（有参考时才有） |
| `<p>dF` | 该结构的力 RMSE，meV/Å（有参考时才有） |
| `<p>Fdev`（逐原子）、`<p>Fdev_max`、`<p>Edev` | 委员会力偏差（DP-GEN 定义：√(各模型 \|F_m − F̄\|² 的平均)）、它在该结构内的最大值、能量标准差（meV/atom） |

| 参数 | 默认 | 说明 |
|------|------|------|
| `-model` | 本地 MPA-0 路径 | `.model` 文件路径；或基础模型名（`small` / `medium` / `large` / `medium-mpa-0` / `medium-omat-0`，别名 `mpa` / `omat`），会在线下载；用逗号分隔多个模型即为委员会 |
| `-batch` | `32` | batch 大小；批量前向失败时自动退回逐结构计算 |
| `-prefix` | `mpa0_` | 输出标签前缀 |
| `-ref` | 自动探测 | 参考标签前缀（`dft_` / `REF_` / 无前缀） |
| `-e0` | `fit` | 能量对齐方式：`fit` / `mean` / `none` / json 文件 |
| `-group` | 无 | 按已有标签分组：任意 info 键，或虚拟键 `formula` / `natoms`；数值键自动切成四分位 |
| `-index` | `:` | ASE 切片，例如 `:100` |
| `-device` `-dtype` | `auto` `float64` | |
| `-head` | 无 | 多 head 模型指定使用哪个 head |
| `-out` `-outdir` `-log` `-overwrite` `-noplot` | | 输出控制 |

### 2.2 推理精度

```
error  mpa0_ vs dft_
                          MAE       RMSE     max|Δ|        R²      N        unit
energy (E0 aligned)      7.30       8.70      18.69    0.9982     40    meV/atom
energy (raw)           116.03     147.65     354.00    0.8951     40    meV/atom
forces                  15.39      19.32      73.22    0.9975  1,302       meV/Å
E0 alignment (fit), eV per atom of element: O -0.2076 · Si +0.3542
```

- **E0 对齐。** 不同 DFT 设置之间（或 DFT 与基础模型之间），总能量的零点**每种元素差一个常数**：E_pred − E_ref = Σ_j n_ij·δ_j。
  - 默认 `-e0 fit` 用最小二乘从数据里解出 δ，扣掉后再算误差。
  - 看 **`energy (E0 aligned)`** 这一行；`raw` 那行只用来看偏移有多大。力不受 E0 影响。
  - ⚠️ δ 是从被评估的数据里拟合出来的，会把模型真实的系统偏差也吸收进去。用**自己训练的模型**评估**同一套 DFT** 的数据时，请用 `-e0 none`。
- **分组。** `-group config_type` 按组列出误差，从差到好排序，图上多一列排名条形图。输出 xyz 里的 `<p>dE` / `<p>dF` 可以作为下一轮的分组键（例如 `-group mpa0_dE`，即按误差大小分组）。
- **最差的结构。** 列出索引、所在组、化学式、ΔE 和力 RMSE，方便按图索骥。
- **图。** `<stem>_summary.png` 包含能量 parity、力 parity 和两个误差分布；点数超过 2 万时自动改用 hexbin。

### 2.3 没有 DFT 参考时：用委员会估计不确定度

新结构（例如从 MD 里抽出来的帧）往往没有 DFT 标签，这时只能看模型对它有多"拿不准"：

```bash
mflow calc -in md_frames.xyz -model s1.model,s2.model,s3.model -prefix com_
```

- **委员会怎么组。** `mflow train -committee 4` 一键训练 4 个成员：每个成员换 seed，微调时默认再做 **bagging**（每个成员随机用 80 % 的训练集）。原因：从同一个基础模型微调出来的成员起点相同，只换 seed 时彼此高度相关，σ 会严重偏小。输出 `committee.json` 和 `<name>_committee_report.json`（测试集上的误差 + 校准）。
- **主动学习看什么。** 看 `com_Fdev_max`，即每个结构里最大的原子力偏差。`metrics.json` 的 `committee` 段给出它的分布。DP-GEN 惯用的信任窗口是 0.05–0.15 eV/Å：
  - 低于下限：模型已经会了；
  - 高于上限：多半是不物理的结构；
  - 落在窗口中间：送去 DFT。
- **只用力偏差排序。** 能量标准差的指示性比力差。
- ⚠️ **不确定度不等于误差。** 委员会的不确定度通常校准得不好。在有参考标签的数据上 `calc` 会给出校准表（`<stem>_calibration.png`）：
  - **Spearman ρ**（σ 与 \|误差\| 的秩相关）：≥ 0.5 pass，≥ 0.2 warn，否则 σ 连排序都不可靠；
  - **miscalibration area**、RMS 校准误差、sharpness；
  - **RMS(误差)/RMS(σ)**：> 1 表示过度自信。主动学习筛选时会用这个比值把 σ 放大（variance scaling）。

### 2.4 结构弛豫：`mflow relax`

```bash
mflow relax -in data.xyz                           # 位置 + 晶胞，fmax 0.05
mflow relax -in data.xyz -cell none -fmax 0.02     # 固定晶胞，更严
mflow relax -in data.xyz -nproc 3 -dtype float32   # 3 个 worker 共享一张卡
mflow relax -in data.xyz -resume                   # 接着上次中断的地方继续
```

- **优化器。** 用 ASE 自带的（`fire` / `fire2` / `lbfgs` / `bfgs`）；`-cell full` 时套用 `FrechetCellFilter`。
- **输出标签。** `<p>steps`、`<p>converged`、`<p>energy0`、`<p>Erelax`、`<p>fmax`、`<p>dmax`、`<p>dvol`、`<p>walltime`。
  - 未收敛的结构也会写出，**请按 `<p>converged` 筛选**，不要按 `<p>fmax`。
- **并行。** 用 spawn 方式开多进程，每个 worker 只加载一次模型。实测 `-nproc 3` 与 `-nproc 1` 的能量差在 2×10⁻¹⁴ eV。
- **中断恢复。** 结果流式写入 `.part` 文件，进程被 kill 也不丢，`-resume` 接着跑。
- **资源监控。** 进度条右侧实时显示整卡/整机的 GPU、CPU、内存占用。
  - GPU 利用率低于 30% 显示为红色，说明该加 worker；显存超过 90% 说明该减 worker。
  - 设 `MFLOW_NO_MONITOR=1` 可以关闭监控。
- ⚠️ **单卡上跑不满。** ASE 这条路线无法占满一张 GPU；大批量的小结构以后可能改用 torch-sim（尚未实现）。

---

## 三、LAMMPS 动力学

MACE 通过 **ML-IAP unified + KOKKOS** 在 GPU 上运行。**积分、恒温、恒压、dump、restart 全部由 LAMMPS 自己的模块完成**。
MFlow 只负责生成输入、检查、启动和分析，进程内不加载 LAMMPS。

### 3.1 导出与环境检查

```bash
mflow deploy -model cu/cu.model -outdir deploy                  # 单 head 模型
mflow deploy -model cu/cu.model -head Default -outdir deploy    # 多 head 必须指定（多头微调后用自己的 head，不要用 pt_head）
mflow deploy -model cu/cu.model -dtype float64 -outdir deploy64 # 另导一份双精度，用于严格对比
mflow lmp-check -deploy deploy                                  # 换机器或换 LAMMPS 后做一次
```

**`deploy`**
- 调用 `mace_create_lammps_model --format mliap`。**要在跑 LAMMPS 的那种 GPU 上执行。**
- 写出 `deploy.json`，内容包括：源模型和导出文件的 sha256、head、**元素顺序**（即 LAMMPS 原子类型的顺序）、r_max、精度、版本号。
- 后续所有命令只认 `deploy.json`。

**`lmp-check`** 依次检查三件事：
1. `lmp -h` 里是否有 KOKKOS(CUDA)、ML-IAP、PYTHON。
2. **LAMMPS 内嵌的 Python** 能否 import mace、torch、cuequivariance、cupy，torch 能否看到 GPU。
3. 用导出的模型实际跑一次 2 原子的 `run 0`。

`lmp` 可执行文件的查找顺序：`-lmp` > 环境变量 `MFLOW_LMP` > `mflow config set lmp` > PATH。

LAMMPS 编译选项（摘自 MACE 文档）：

```bash
cmake ../cmake -D BUILD_MPI=ON -D BUILD_SHARED_LIBS=ON -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_<你的GPU>=ON -D PKG_ML-IAP=ON -D PKG_ML-SNAP=ON -D PKG_PYTHON=ON -D MLIAP_ENABLE_PYTHON=ON
```

### 3.2 输入是怎么生成的

```bash
mflow md -in start.xyz -deploy deploy -ensemble nvt -temp 300 -steps 100000 -outdir md300 -dry   # 只写输入，先看
mflow md -in start.xyz -deploy deploy -ensemble nvt -temp 300 -steps 100000 -outdir md300        # 运行
mflow md -in start.xyz -deploy deploy -steps 200000 -outdir md300 -resume                         # 续跑；-steps 是总步数
```

**只有一个固定模板，所有选择都写在文件里：**

```
units metal / atom_style atomic / atom_modify map yes / newton on / boundary p p p
read_data       data.lmp                       # 由 ASE 写出；原子类型 = 结构里出现的元素，按原子序数排序
pair_style      mliap unified <模型>.pt 0
pair_coeff      * * Al Cu                      # 与 data 文件、dump_modify element 用同一份元素列表
neighbor 2.0 bin / neigh_modify every 1 delay 0 check yes
timestep        <dt/1000>                      # -dt 的单位是 fs
thermo_style    custom step time temp pe ke etotal press vol density pxx pyy pzz
thermo_modify   lost error
velocity        all create T seed dist gaussian mom yes rot yes
fix             md all nvt|npt|nve ...         # 见 3.3
fix             runaway all halt N v_tnow > 5×T error soft     # 温度失控就停
dump            traj all custom N traj.lammpstrj id type element xu yu zu vx vy vz fx fy fz
restart         N restart.a restart.b          # 两个文件轮流写
run             STEPS upto                     # 续跑时跑到总步数为止
```

**运行前自动做三件事：**

1. **参数检查。**
   - 步长：含 H 时 ≤ 1 fs，否则 ≤ 2 fs。
   - 阻尼：tdamp ≈ 100·dt，pdamp ≈ 1000·dt。
   - NPT 必须是全周期体系。
   - 每张卡 1000–4000 原子最合适。
   - 晶胞宽度小于 2·r_max 时提示扩胞。
2. **`lmp -sr`。** 把真实输入完整解析、初始化一遍，但不积分。语法错误、缺少的 style、原子类型数不对，都会在这一步暴露。
3. **一致性检查。** 对起始结构 `run 0`，要求 LAMMPS 算出的**能量、逐原子力、压强**与同一模型在 LAMMPS 外的计算一致。
   - 容差：fp64 导出时为 1e-5 eV/atom、1e-4 eV/Å；fp32 导出时为 1e-3、2e-2。
   - **元素顺序错位、单位错误、维里符号、导错 head、版本不一致**，都会在这里被拦下。
   - **一致性检查失败时，不要用 `-nocheck` 或 `-force` 绕过。**

运行中实时显示步数、温度、ns/day 和 GPU 占用；结束后自动执行 `md-report`。

### 3.3 系综怎么选

`-ensemble nvt|npt|nve`，`-thermostat nh|csvr|langevin`（Nosé–Hoover 链 / Bussi 随机速度重标 / Langevin），`-baro iso|aniso|tri`；
多阶段的常用流程用 `-protocol`（见下一小节的表）。

| 目的 | 推荐做法 | 原因 |
|---|---|---|
| 给定 T、P 下的平衡结构 / 密度 | `-protocol equilibrate -temp T -press P`；液体、非晶、立方晶体 `-baro iso`，正交非立方 `aniso`，三斜 `tri` | NPT 让体积平衡，再固定在后半段平均体积上做 NVT |
| 给定体积下的结构、RDF | 先 npt 平衡，再 `-ensemble nvt` | 体积固定，涨落更小 |
| **扩散系数、热导等输运性质** | `-protocol transport`：NPT → 平均体积 → NVT → **NVE 生产段** | 强耦合的局部恒温器（如 Langevin）会**压低扩散**；NH、CSVR 与 NVE 在统计上无差别（Basconi & Shirts, *JCTC* 2013） |
| 检查势函数是否守恒、步长是否合适 | `-ensemble nve`，看 NVE 漂移闸门 | 漂移 < 5×10⁻⁴ eV/atom/ps 为通过（LAMBench 标准） |
| 升温 / 退火 / 淬火 | `-protocol anneal` 或 `quench`，`-temp T1 -temp_end T2` | 报告给出 K/ps，> 10 K/ps warn；越慢越接近真实 |
| 熔点 | `-protocol melt`：固液两相共存，**NPH** 下温度自己落到 Tm；按层的固相键数判断两相是否都在 | 直接加热到熔化会有很大的过热滞后（完美晶体可过热到约 1.2 Tm） |
| 0 K 结构 / 晶格常数 | `mflow relax`，或 `-protocol minimize`（LAMMPS 的 `fix box/relax` + 反复 `minimize`，输出 `relaxed.xyz`） | box/relax 不是严格的极小化，要重复到晶胞不变（LAMMPS 文档） |
| 初始结构很差（原子重叠、能量很高） | 先 `-protocol minimize`，再 `-thermostat csvr` 或 `langevin` 平衡 | 稳健。不要用 Berendsen 做生产：系综不对，还会出现 flying ice cube |

**参数经验：**

| 参数 | 经验值 | 说明 |
|---|---|---|
| `-dt` | 1 fs；含 H 或温度高于约 2000 K 时 0.5 fs；不含 H 的金属、氧化物 ≤ 2 fs | 用 NVE 漂移闸门验证 |
| `-tdamp` | 默认 100·dt | 太小温度剧烈振荡，太大平衡很慢（LAMMPS `fix nh` 文档） |
| `-pdamp` | 默认 1000·dt | 同上 |
| 原子数 | 每张卡 1000–4000 | 太少 GPU 闲置；晶胞宽度不要小于 2·r_max |
| `-skip` | 默认 0.2 | 平衡段所占比例，这部分不参与闸门 |
| 是否已平衡 | 生产段的温度、体积、势能都不再有趋势 | `md-report` 自动比较前后两半（差 > 3 个分块误差 warn） |

**协议一览（`-protocol`）：**

| 协议 | 阶段 | 报告里多出的内容 |
|---|---|---|
| `equilibrate` | NPT 50 %（结束时设为后半段平均体积）→ NVT 50 % | 闸门只看最后一段 |
| `transport` | NPT 20 % → 平均体积 → NVT 10 % → NVE 70 % | NVE 漂移；D（总的和 x/y/z 分量）、MSD 双对数斜率（0.85–1.15 才算扩散区）；`-viscosity` 加 Yeh–Hummer 修正 |
| `anneal` / `quench` | 升温–保温–降温 / 保温–降温 | 变温速率 K/ps |
| `minimize` | box/relax + minimize 循环（最多 20 次） | 最大力、压强、体积变化、`relaxed.xyz` |
| `melt` | NPT 晶体 → 上半盒在 `-melt_temp` 熔化（下半冻结）→ 两半都设到 `-temp` → NPH | T_m 估计 ± 分块误差、液相比例、体积是否失控（沸腾） |

续跑（`-resume`）按阶段进行：已完成的阶段跳过，`-steps` 仍是总步数。

### 3.4 质量闸门：`mflow md-report`

```bash
mflow md-report -dir md300                    # md 结束时会自动执行
mflow md-report -dir md300 -skip 0.3 -stride 5
mflow md-report -dir md300 -ref aimd.xyz      # 与 AIMD 比较 RDF
```

| 闸门 | 判据 |
|---|---|
| LAMMPS ERROR、丢原子、fix halt 触发 | 出现即 fail |
| Dangerous builds | > 0 告警（步长或 skin 不合适） |
| NVE 能量漂移 | 生产段 E_total/atom 对时间的斜率：< 5×10⁻⁴ eV/atom/ps 为 pass，< 5×10⁻³ 为 warn |
| 温度（NVT/NPT） | 生产段均值与目标值相差 < 5% 为 pass，< 15% 为 warn；生产段线性趋势需 < 5% |
| 体积（NPT） | 生产段体积变化 < 2% 为 pass，< 5% 为 warn |
| 最近原子对 | 任何一帧 < 0.6 Å 即 fail（势函数在这里有"洞"，需要补数据） |
| RDF 对比参考（`-ref`） | 每个元素对 g(r) 的平均绝对差 < 0.1 为 pass |
| 平衡（前后两半） | T、PE（NPT 时加 V）前后两半之差 < 3 个分块误差 |
| 扩散区 | 可动元素的 MSD 双对数斜率在 0.85–1.15；在 NVT/NPT 下测扩散会提示改用 transport |

统计：均值的误差用分块平均（Flyvbjerg–Petersen），报告给出 `± SE` 和有效样本数 `n_eff`。
分析与作图（`md_summary.png`，2×3）：T、P、E、V 随时间的变化（灰色为平衡段，虚线为协议阶段分界）、按元素对分开的 RDF、VDOS（帧间隔 ≤ 25 fs 时，否则画 MSD）。VDOS 由质量加权的速度自相关函数算出，写进 `vdos.dat`。

**MD 崩了怎么办：** 先把 `-dt` 减半重跑。如果还崩，说明势函数在这个区域是外推，需要把这些构型补进训练集（见第四章的主动学习），并跑 `mflow validate dimer` 看短程墙。

### 3.5 有没有 skill 能辅助设定

有。[`skills/mflow-md`](skills/mflow-md/SKILL.md) 规定执行顺序、参数范围和常见问题；
[`mflow-md-setup`](skills/mflow-md-setup/SKILL.md)（系综、步长、盒子）、[`mflow-property-protocols`](skills/mflow-property-protocols/SKILL.md)（熔点、扩散、RDF、非晶）、[`mflow-failure-triage`](skills/mflow-failure-triage/SKILL.md)（崩溃排错）是经验类 skill。
它们参考了下面这些公开 skill，只借鉴写法和经验、不直接复制，出处记在各 skill 的 `references/SOURCES.md`：

| 来源 | 可借鉴的经验 |
|---|---|
| [AtomisticSkills](https://github.com/learningmatter-mit/AtomisticSkills) `mat-md-monitors` / `mat-melting-point` / `mat-diffusion-analysis` / `mat-amorphization`（MIT 许可） | 爆炸、过冲、体积失控时的停止规则；固液共存法测熔点的建模与判据；扩散的温度范围、盒子大小与 Arrhenius 拟合；熔融–淬火速率 |
| [HeshamFS/materials-simulation-skills](https://github.com/HeshamFS/materials-simulation-skills) `md-analysis-planner` / `simulation-failure-triage`（Apache-2.0） | 检验 MSD 是否处于线性区、Yeh–Hummer 有限尺寸修正、分块误差棒；"每次只改一个变量"的逐级排错 |
| [CatMaster](https://github.com/q734738781/CatMaster) `lammps-preparation` / `trajectory-analysis`（Apache-2.0） | 不根据成分猜势函数或类型映射；先确认轨迹健康，再解读结果 |

---

## 四、主动学习：`mflow al`

**MFlow 负责"提议、挑选、交出去、收回来、再训练、判断何时停"，DFT 由 [TFlow](https://github.com/MengnanCui/TFlow) 算。**

```bash
mflow al init   -name cu_al -system start.xyz -data dft.xyz -test test.xyz -foundation mpa -committee 4
mflow al run    -name cu_al                 # 第 0 轮：基础模型探索 → FPS 选帧 → 导出给 TFlow，然后停下
#   …… 在 TFlow 里算 round_00/export/candidates.extxyz（命令写在 round_00/export/README.txt）……
mflow al import -name cu_al -tflow /path/to/calc_dir
mflow al run    -name cu_al                 # 训练委员会 → 探索 → 委员会筛选 → 导出（下一轮），再停下
mflow al status -name cu_al                 # 台账 + 停止判据
```

每一步也能单独跑：`al explore / select / export / import / train`。`al run` 走到"需要 DFT"时一定停下（人工闸门）。

### 4.1 一轮做什么

| 步骤 | 做法 | 在哪改 |
|---|---|---|
| 探索 | `md`：T（和 P）网格上的短 MD，每轮温度乘 `widen`（默认 1.25）；引擎 `ase`（Langevin）或 `lammps`（走 `mflow md`，自动导出成员 0）。`perturb`：rattle + 随机应变（+ 空位）。`minhop`：ASE 极小值跳跃，**所有到过的局部极小**都收（不只被接受的） | `campaign.yaml` 的 `explore` |
| 筛选 | 第 0 轮没有委员会：只做多样性（MACE 描述子上的 FPS）。之后：委员会力偏差 `Fdev_max`，DP-GEN 式分类 accurate / candidate / failed；信任窗口 `fixed`（0.05–0.15 eV/Å）、`auto`（训练集偏差的 Q3 + 1.5·IQR，委员会没收敛时自动退回 fixed）、`relative`；σ 先按测试集上的 RMS(误差)/RMS(σ) 放大；候选再用 FPS（相对已有训练集）限量 `max_per_round` | `select` |
| 导出 | `round_NN/export/candidates.extxyz`（只有几何和 `mflow_id`）、`run_label.py`（给 `vaspctl submit --src-run` 的单点脚本）、`manifest.json`、`README.txt`（四条命令） | `run_label.py` 里的 INCAR：**整个活动保持一致**，并和基础模型的 DFT 设置对齐 |
| 收回 | 读 TFlow 的 `job_map.json`，按 `info.mflow_id` 对回去，解析 `static/OUTCAR`；**拒收**：状态不是 done、跑过 error fix（`error_job_*`、KPOINTS 被改）、SCF 步数达到 NELM（TFlow 对 static 不查这个）、结构被改过、OUTCAR 读不了；再过一遍 `check` 的规则（NaN、力过大、原子过近）。拒收原因写 `rejects.jsonl` | `-tflow <calc_dir>`，或 `-in labelled.xyz`（任何带 `mflow_id` 和参考标签的 extxyz） |
| 训练 | 数据集**每轮重建**（initial + 所有已标注轮次），测试集固定；委员会 `train -committee N`（bagging），记下 Spearman 和 σ 比值 | `train` |
| 判断 | `al status`：候选比例 < 5 %；**新帧误差**（这一轮 DFT 刚算、模型没见过的帧）能量 < 10 meV/atom、力 < 50 meV/Å。都满足才建议停，然后还要跑生产 MD 看闸门 | `stop`，或 `thresholds.yaml` 的 `al` 段 |

`ledger.jsonl` 只追加：每一步的设置、数量、模型、误差都记在里面；`state.json` 是当前轮次和阶段。

### 4.2 与 TFlow 的约定

```bash
vaspctl prepare <calc_dir> round_00/export/candidates.extxyz --id-key mflow_id --tag mflow_r00
vaspctl submit  <calc_dir> --src-run round_00/export/run_label.py --src-submit <submit.sh> --max-concurrent 20 --auto-resume --watch
vaspctl report  <calc_dir> --task-dirs static --no-queue
mflow al import -name cu_al -tflow <calc_dir>
```

- `REF_energy` 取 VASP 的 free energy（TOTEN，力是它的导数），σ→0 的能量另存为 `REF_energy_sigma0`；`REF_stress` 是 ASE 约定（eV/Å³，符号已翻转）。
- TFlow 的错误修复可能缩放体积或改 KPOINTS，修复后算出来的不是你要的那个结构或设置，MFlow 一律拒收而不是"将就"。

### 4.3 方案怎么选

依据见[路线图第 1 节](docs/roadmap-v0.2.md#1-训练方案全景来自-20242026-年的综述)（Kulichenko 2024 *Chem. Rev.*、Grasselli 2025、Tompa 2026 等）。经验写在 [`mflow-active-learning-campaign`](skills/mflow-active-learning-campaign/SKILL.md)：

- 体相、液体：`md` 网格 + `perturb`；团簇、表面：加 `minhop`。
- 委员会 ≥ 3，建议 4；微调时保持 bagging。
- `trust: fixed` 起步；委员会校准好（Spearman ≥ 0.5）后再试 `auto`。
- 一轮 50–150 个结构；连续两轮候选比例不降，说明探索不够（加温度、加步数、开 minhop），而不是模型够好了。

---

## 五、给 agent 用：skills

[`skills/`](skills/) 里的 skill 遵循 [agentskills.io](https://agentskills.io/specification) 标准，pi 会自动加载。
安装：`mflow skills install`（默认链接到 `~/.agents/skills`；pi 也读 `~/.pi/agent/skills`，用 `-dest` 指定；`-copy` 复制而不是链接）。`mflow skills` 列出哪些已经装上。

| skill | 用途 |
|---|---|
| [`mflow-train`](skills/mflow-train/SKILL.md) | 体检 → dry run → 训练 → 报告；怎么选模式；超参只能用 `-set` 改 |
| [`mflow-md`](skills/mflow-md/SKILL.md) | 导出 → 环境检查 → dry run → MD → 报告；参数范围；一致性检查失败绝不绕过 |
| [`mflow-report-reading`](skills/mflow-report-reading/SKILL.md) | 怎么读 JSON 报告；遇到 fail 必须停；不编造数字 |

经验类（v0.2）：

| skill | 用途 |
|---|---|
| [`mflow-research-protocol`](skills/mflow-research-protocol/SKILL.md) | 请求分类、计划与批准、成本估算、证据先于结论 |
| [`mflow-finetune-strategy`](skills/mflow-finetune-strategy/SKILL.md) | 选基础模型、模式、E0、学习率；遗忘的征兆 |
| [`mflow-dataset-curation`](skills/mflow-dataset-curation/SKILL.md) | 标签与单位、按来源切分、固定测试集 |
| [`mflow-potential-acceptance`](skills/mflow-potential-acceptance/SKILL.md) | 四层验证，按用途设门槛 |
| [`mflow-active-learning-campaign`](skills/mflow-active-learning-campaign/SKILL.md) | 探索器、信任窗口、拒收、何时停 |
| [`mflow-md-setup`](skills/mflow-md-setup/SKILL.md) | 系综、步长、盒子、中止条件 |
| [`mflow-property-protocols`](skills/mflow-property-protocols/SKILL.md) | 熔点、扩散、RDF、VDOS、非晶 |
| [`mflow-failure-triage`](skills/mflow-failure-triage/SKILL.md) | 出错时的排查顺序 |
| [`mflow-lab-notebook`](skills/mflow-lab-notebook/SKILL.md) | `NOTEBOOK.md` / `LESSONS.md` 与模板 |

**分工：** 代码控制执行（顺序、校验、闸门）；skill 提供经验（选什么、够不够、下一步做什么）。
经验类 skill 改写自公开 skill 集合与文献，出处固定到 commit，记在各自的 `references/SOURCES.md`。

### 5.1 跟 agent 怎么说

直接说**目标**，不用说命令。按 skill 的约定，agent 会先写计划给你看，在耗时的 GPU 步骤和任何 DFT 之前停下来等你确认。

| 你说 | agent 会做什么 | 在哪里停下来问你 |
|---|---|---|
| "看看 `dft.xyz` 能不能拿来微调 MPA" | `check -model mpa -split 0.1`，读报告，告诉你坏帧和 DFT 设置是否匹配 | 有 fail 时 |
| "用这些数据微调，要能跑 1000 K 的液体" | 选模式（多半是 multihead）、`train -dry` 给你看配置 | 开始训练前 |
| "这个模型够不够算扩散系数？" | 按四层验证走：训练报告 → `validate dimer` → NVE → 目标温度 MD | 需要 DFT 复算时 |
| "MD 在 1500 K 崩了" | 按排错表一次改一个变量：减半步长 → dimer → 委员会打分 | 结论指向"需要补数据"时 |
| "帮我做一轮主动学习" | `al run`，汇报候选比例、选了多少帧、预计 DFT 量 | 交给 TFlow 之前（一定会停） |
| "TFlow 算完了，在 /path/calc" | `al import`，汇报收下多少、拒收多少及原因、新帧误差 | 拒收比例 > 10 % 时 |
| "算一下 Cu 的熔点" | 建拉长的盒子 → 估成本 → 两个起点跑 `melt` | 开跑前（几个 GPU 小时） |

agent 汇报时每个数字都应带出处（哪个 JSON 的哪个字段）；没有出处的数字请追问。

---

## 六、输出文件与约定

| 命令 | 主要输出 |
|---|---|
| `check` | `<stem>_check.json` · `<stem>_check.png` · （加 `-split` 时）`<stem>_train.xyz` / `<stem>_test.xyz` |
| `train` / `train-report` | `<run>/config.yaml` · `<run>/<run>.model` · `<run>_train_report.json` · `_learning_curve.png` · `_test_summary.png` · `_test_mace.xyz` |
| `calc` / `plot` | `<stem>_mace.xyz` · `<stem>_metrics.json` · `<stem>_summary.png` |
| `relax` | `<stem>_relax.xyz` · `<stem>_relax_metrics.json` · `<stem>_relax.png` |
| `deploy` | `deploy/deploy.json` · `*.model-mliap_lammps.pt` · `deploy_report.json` |
| `lmp-check` | `lmp_check/lmp_env.json` |
| `md` / `md-report` | `<dir>/in.lammps` · `data.lmp` · `log.lammps` · `traj.lammpstrj` · `md.json` · `md_report.json` · `md_summary.png`；协议另有 `vdos.dat` · `phase_profile.dat`（melt）· `relaxed.xyz`（minimize） |
| `train -committee N` | `<dir>/m0..m{N-1}/` · `committee.json` · `<name>_committee_report.json` · `committee_calibration.png` |
| `validate` | `validate/<stem>_<kind>.json` · `.png` |
| `al` | `<campaign>/campaign.yaml` · `state.json` · `ledger.jsonl` · `status.json` · `data/train_vNN.xyz` · `round_NN/{explore,select,export,train}/` |
| 计算类命令 | `py.log`：终端输出的无色副本（`doctor` / `config` / `skills` / `tool` / `completion` 不写） |

**约定：**
- 所有功能都从一个命令 `mflow` 进入；`mflow` 不带参数列出全部命令。
- 参数一律用多字母（`-in`、`-model`……），避免冲突。
- 终端彩色、日志无色，内容相同。
- 不覆盖已有标签。
- 长任务都能 `-resume`。

**其他文档：**
- MFlow 替你做的所有决定和对应的风险点，记录在 [`docs/conventions.html`](docs/conventions.html)：E0 对齐、单位、训练默认值、LAMMPS 的单位 / 元素顺序 / 维里约定、一致性检查容差。
- 针对特定体系的小工具：`mflow tool` 列出，例如 `mflow tool izo-formation`（In–Zn–O 形成能，说明见 [`tools/README.md`](tools/README.md)）。

---

## 七、版本

| 版本 | 内容 |
|---|---|
| v0.1 | 包结构；`check` / `train` / `train-report`；`calc`（含委员会）/ `relax` / `plot`；`deploy` / `lmp-check` / `md` / `md-report`；3 个流程类 skill |
| **v0.2.0a1（当前）** | a：`train -committee`（bagging）、不确定度校准、力误差按元素、`thresholds.yaml`；b：`mflow al` 主动学习闭环 + TFlow 对接；c：极小值跳跃探索；d：`validate`（dimer/eos/phonon/defect）、LAMMPS 协议库（恒温器/恒压器、equilibrate/transport/anneal/quench/minimize/melt）、`md-report` 统计（分块误差、平衡检验、扩散区、VDOS、Yeh–Hummer）；e：9 个经验类 skill。打包：`mflow doctor` / `config` / `skills` / `tool` / `completion`、`install.sh`。64 个测试。**尚未在 GPU + 真实 MACE/LAMMPS 上做端到端验证**（测试用 EMT / LJ / EAM 代替） |
| 以后 | 随机结构搜索探索、多头委员会（MACE PR #800 合并后）、不确定度偏置 MD、PLUMED 增强采样模板、集群提交，见路线图 |

v0.1 之前是单文件脚本 `mflow.py`（内部版本 0.2–0.6，只有 calc/relax/plot），现已并入当前的包，计算结果逐位一致。

---

## 八、命令速查

所有命令都有 `-log`（默认 `py.log`）和 `-h`。完整说明见各章。

| 命令 | 必填 | 常用选项 |
|---|---|---|
| `check` | `-in` | `-model` `-head` `-test` `-split` `-seed` `-ref` `-group` `-outdir` |
| `train` | `-in` `-name` | `-test` `-valid` `-mode finetune\|multihead\|lora\|scratch` `-foundation` `-head` `-pt` `-epochs` `-batch` `-committee` `-bagging` `-set k=v …` `-dry` `-resume` `-force` |
| `train-report` | `-dir` | `-test` `-group` |
| `calc` | `-in` | `-model a[,b,c]` `-head` `-ref` `-e0 fit\|mean\|none\|file.json` `-group` `-prefix` `-batch` `-dtype` |
| `relax` | `-in` | `-model` `-fmax` `-steps` `-opt` `-cell full\|none` `-nproc` `-traj` `-resume` |
| `plot` | `-in` | `-pred` `-ref` `-group` |
| `validate` | `dimer\|eos\|phonon\|defect` | `-model` `-in` `-elements` `-ref "k=v,…"` `-strain` `-supercell` `-mesh` `-surface` `-min_width` `-outdir` |
| `deploy` | `-model` | `-head` `-dtype float32\|float64` `-outdir` |
| `lmp-check` | — | `-lmp` `-deploy` |
| `md` | `-in` `-deploy` | `-protocol` `-ensemble` `-thermostat` `-baro` `-temp` `-temp_end` `-press` `-dt` `-steps` `-dump` `-melt_temp` `-fmax` `-seed` `-ngpu` `-outdir` `-dry` `-resume` |
| `md-report` | `-dir` | `-skip` `-stride` `-ref aimd.xyz` `-viscosity` |
| `al init` | `-name` `-system` | `-data` `-test` `-foundation` `-mode` `-committee` `-engine ase\|lammps` |
| `al run` | `-name` | `-lmp` `-force` |
| `al explore / select / export / train` | `-name` | `-round` |
| `al import` | `-name` + `-tflow DIR` 或 `-in FILE` | `-round` |
| `al status` | `-name` | |
| `doctor` | — | `-lmp PATH`（检查这个 LAMMPS 的编译）`-full` `-json FILE` |
| `config` | — | `show` · `set <key> <value>` · `unset <key>` · `path`；key = `model` / `lmp` / `skills_dir` |
| `skills` | — | `list` · `install [-dest DIR] [-copy] [-force]` · `path` |
| `tool` | — | `mflow tool` 列出；`mflow tool izo-formation -in … -eZnO … -eIn2O3 …` |
| `completion` | — | `bash` / `zsh` |

单位：能量 eV（报告里 meV/atom）、力 eV/Å（报告里 meV/Å）、`-dt` / `-tdamp` / `-pdamp` 用 fs、`-press` 用 bar、温度 K、扩散 cm²/s、频率 THz。

---

## 九、常见问题

**Q：一定要 `python xxx.py` 吗？**
不用。安装后（见[安装](#安装)）所有功能都是 `mflow <命令>`，在任何目录都能用；`python -m mflow` 等价。旧的 `python tools/izo_formation.py` 仍然能用，但推荐 `mflow tool izo-formation`。

**Q：每次都要写 `-model` 和 LAMMPS 路径吗？**
不用。`mflow config set model …`、`mflow config set lmp …` 设一次即可；单次想换，命令行照样写 `-model`。

**Q：MFlow 会帮我算 DFT 吗？**
不会。DFT 由 TFlow 负责；MFlow 写出给 TFlow 的结构和单点脚本，再把结果读回来。也可以用任何带 `mflow_id` 和参考标签的 extxyz 导入。

**Q：没有 GPU 能用吗？**
`check` / `calc` / `relax` / `train` / `validate` 在 CPU 上都能跑（慢）。`deploy` 和 LAMMPS 的 MACE 路线需要 GPU。

**Q：没有 MACE 模型，想先试试命令？**
`mflow validate … -model emt` 用 ASE 的 EMT；PATH 里有 `lmp` 时，`pytest` 会用 LJ / EAM 把 LAMMPS 流程完整跑一遍。

**Q：我的 LAMMPS 是 conda 装的 CPU 版，能用吗？**
MACE 的 ML-IAP 路线需要自己编译的 KOKKOS(CUDA) + ML-IAP + PYTHON 版本（见 3.1），`mflow lmp-check` 会告诉你缺什么。

**Q：为什么一致性检查不让我跳过？**
它失败只有几种原因：元素顺序错位、导出了错误的 head、导出环境与 LAMMPS 内嵌 Python 的版本不一致、单位或维里约定错误。每一种都会让整段 MD 静默出错。

**Q：阈值太严 / 太松怎么办？**
在工作目录写 `mflow_thresholds.yaml`，只写要改的键（键名见 `mflow/thresholds.yaml`），或用 `MFLOW_THRESHOLDS=/path/file.yaml`。改了什么、为什么，记进 `LESSONS.md`。

**Q：多头模型（MH-1、多头微调后的模型）怎么用？**
`calc` / `validate` / `deploy` 都加 `-head`。多头微调之后用你自己的 head（通常 `Default`），不要用 `pt_head`。

**Q：结果放在哪？怎么给别人复查？**
每个命令一个 JSON 报告（`status` + 逐项检查 + 建议 + 数据）和一张图；MD 目录里有完整的 LAMMPS 输入和 `md.json`，可以原样重跑。见第六章。

**Q：mace-torch 能升级吗？**
目前固定 0.3.16。上游 v1.0 会改命令行参数，升级后先跑 `pytest`。

---

## 十、能力边界与已知限制

**不做的事：**
- DFT 计算、输入生成和作业提交（TFlow 负责）；集群作业提交（MFlow 在本机直接跑）。
- 在 LAMMPS 里实时计算 MACE 的不确定度（上游不支持；委员会打分在 MD 之后离线做）。

**还没实现：**
- 主动学习：随机结构搜索探索、QUESTS 信息熵去重与停止判据、不确定度偏置 MD、多头委员会。
- 性质：弹性常数、Arrhenius 自动拟合、离子电导（Nernst–Einstein）、热导、增强采样（PLUMED）。
- 大批量 GPU 弛豫（torch-sim）。

**已知限制：**
- ⚠️ **尚未在 GPU + 真实 MACE + 自编译 LAMMPS 上端到端跑过。** 第一次使用请从 `lmp-check` 开始，并先跑一条短 NVE 看漂移。
- ML-IAP 多卡在部分 LAMMPS 版本上会崩，默认单卡；多分区、副本交换不支持。
- `fix temp/csvr` 的 KOKKOS 版本要 LAMMPS ≥ 2Sep2026，旧版本在 CPU 上算这个 fix（正确但慢）。
- `melt` 协议需要正交晶胞、界面垂直于 z。
- `validate defect` 对化合物只给原始能量差（形成能需要化学势）。
- `validate phonon` 需要 `pip install phonopy seekpath`。
- 委员会成员都从同一个基础模型微调时，不确定度偏小；bagging 和测试集校准能缓解，但不能完全消除。

