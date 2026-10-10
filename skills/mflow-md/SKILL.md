---
name: mflow-md
description: Run molecular dynamics in LAMMPS with a MACE model through ML-IAP + KOKKOS using MFlow, then analyse it. Use when the user wants NVT / NPT / NVE MD, cell relaxation, annealing or quenching, melting points, diffusion, RDF, VDOS, or to check whether a MACE potential is stable in MD. Covers `mflow deploy`, `mflow lmp-check`, `mflow md` (protocols), `mflow md-report`. For choosing ensembles and timesteps see mflow-md-setup; for property recipes see mflow-property-protocols.
---

# LAMMPS 动力学（MFlow，ML-IAP + KOKKOS）

动力学（积分、恒温、恒压）**全部由 LAMMPS 完成**；MFlow 只生成输入、做检查、启动、分析。

## 固定顺序

1. **导出模型**（每个模型做一次，必须在 GPU 上）：
   `mflow deploy -model <run>/<run>.model -head Default -outdir deploy`
   - 多 head 模型必须给 `-head`；多头微调后用你自己的 head（通常 `Default`），**不要用 `pt_head`**。
   - 生产用 `-dtype float32`（默认）；需要严格对比时另导一份 `-dtype float64`。
2. **检查环境**（换机器 / 换 LAMMPS 后做一次）：`mflow lmp-check -deploy deploy`
   - 任何 fail → 停下报告给用户（多半是 LAMMPS 编译选项或 LAMMPS 内嵌 Python 里缺 mace/cuequivariance/cupy）。
3. **先 dry run**：`mflow md -in start.xyz -deploy deploy -ensemble nvt -temp 300 -steps 100000 -outdir md_300K -dry`
   - 看 `md_300K/in.lammps` 和 `md_report.json` 里的参数检查（timestep、damping、atoms per GPU、cell size）。
4. **运行**：去掉 `-dry`。命令会依次做 `lmp -sr`（语法与初始化）→ run 0 一致性检查 → 正式运行 → 报告。
   - 一致性检查 fail = LAMMPS 和模型本身算出的能量/力/压强对不上 → **绝不能用 `-nocheck` 或 `-force` 绕过**，报告给用户。
5. **读报告**：`md_300K/md_report.json`（见 `mflow-report-reading`）。

## 参数范围

| 参数 | 建议 |
|---|---|
| `-dt` | 1 fs（含 H）；无 H 的金属/氧化物最多 2 fs |
| `-tdamp` / `-pdamp` | 默认 100·dt / 1000·dt，一般不用改 |
| 原子数 | 每张 GPU 1000–4000 原子效率最高；小于 2·r_max 的晶胞先扩胞 |
| `-ensemble npt` | `-press` 单位 bar；只用于全周期体系 |
| `-ngpu N` | 多卡是实验性的（ML-IAP 多卡在部分 LAMMPS 版本会崩），先单卡 |
| `-dump` / `-thermo` | 总帧数控制在 1e4 量级以内 |

## 协议（`-protocol`，多阶段，一个 in.lammps 里完成）

| 协议 | 阶段 | 用途 |
|---|---|---|
| `single`（默认） | `-ensemble` 一段 | 普通 NVT / NPT / NVE |
| `equilibrate` | NPT 50 %（结束时体积设为后半段平均值）→ NVT 50 % | 得到目标 T/P 下的平衡结构 |
| `transport` | NPT 20 % → 平均体积 → NVT 10 % → **NVE 70 %** | 扩散系数、VDOS：生产段没有恒温器干扰动力学 |
| `anneal` | 升温 40 % → 保温 20 % → 降温 40 %（`-ensemble npt` 时全程 NPT） | 退火、消除缺陷 |
| `quench` | 保温 20 % → 降温 80 % 到 `-temp_end` | 非晶/玻璃；报告给出 K/ps |
| `minimize` | `fix box/relax` + 反复 `minimize` 直到晶胞不变 | LAMMPS 里的晶胞弛豫，输出 `relaxed.xyz` |
| `melt` | NPT 晶体 20 % → 上半盒（z）在 `-melt_temp` 熔化、下半冻结 15 % → 两半都设到 `-temp` → **NPH** 65 % | 固液共存法熔点 |

- `-thermostat nh|csvr|langevin`：默认 Nosé–Hoover；`csvr`（Bussi）+ nve 是更稳的选择，它的 KOKKOS 版本要 LAMMPS ≥ 2Sep2026，旧版在 CPU 上跑（结果对，只是慢）。`langevin` 会改变动力学，不能用于扩散。
- `-baro iso|aniso|tri`：立方/液体用 iso，正交非立方用 aniso，三斜/剪切用 tri（MFlow 自动写倾斜因子）。
- `melt`：盒子沿 z 拉长（Lz ≈ 2–4 Lx），`-temp` 取熔点的猜测值，`-melt_temp` ≈ 1.5 × 猜测值（不要超过沸点）。报告按层统计 ten Wolde 固相键数判断哪层是晶体：
  - `melt: phases` pass → 两相共存，T 就是 T_m 估计；
  - 全液 → `-temp` 降低再跑；全晶 → 升高；"one phase only" → 看建议（熔化不够或晶体也熔了）。
- `minimize` 不能 `-resume`（重跑即可）；其他协议按阶段续跑。

## 常见问题

| 现象 | 下一步 |
|---|---|
| `runaway halt` / `lost atoms` fail | 先把 `-dt` 减半重跑；还崩 → 势函数在该区域外推，需要补数据（主动学习） |
| `NVE energy drift` warn/fail | 减小 dt；fp32 导出漂移偏大时换 fp64 导出对比 |
| `temperature drift` warn | 跑更长的平衡段，或调大 `-skip` 重新出报告 |
| 想继续跑 | 同一命令加 `-resume`，`-steps` 是**总步数**（run ... upto） |

| `equilibrated (half vs half)` warn | 生产段前后两半差 > 3 个分块误差：延长或加大 `-skip` |
| `diffusive regime` warn | MSD 双对数斜率不在 0.85–1.15：跑更久，或多个 seed 平均 |
| `transport ensemble` warn | 扩散在 NVT/NPT 下测的：改用 `-protocol transport` |
| 没有 VDOS | 帧间隔太大（Nyquist < 20 THz）：`-dump 5` 或 `10`（1 fs 步长时） |

```bash
mflow md-report -dir md_300K -skip 0.3 -stride 5     # 重新分析
mflow md-report -dir md_300K -ref aimd.xyz            # 和 AIMD 比 RDF
mflow md-report -dir d900 -viscosity 4.0              # 加 Yeh–Hummer 有限尺寸修正（η 单位 mPa·s，同一模型算的）
```
