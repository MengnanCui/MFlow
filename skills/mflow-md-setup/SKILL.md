---
name: mflow-md-setup
description: Decide how to set up an MD run with a MACE potential before calling `mflow md` — which ensemble, thermostat and barostat, which protocol, timestep, damping, box size, run length and dump interval. Use when the user asks "how should I run MD for X", when choosing between NVT / NPT / NVE / Langevin / CSVR, or when a run needs to be designed for a property (density, diffusion, RDF, VDOS, melting, quench).
---

# MD 怎么设（经验）

执行顺序和命令在 `mflow-md`；这里只回答"选什么、设多少"。所有阈值引用 `mflow/thresholds.yaml` 的 `md` 段。

## 1. 先问清楚三件事

1. **要什么量？** 结构/密度、输运（D、电导）、振动（VDOS）、相变（熔点、玻璃化）、只是"稳不稳"。
2. **什么条件？** T、P、是否周期、体系大小、是否含 H。
3. **势函数在这里可信吗？** 没跑过 `mflow validate dimer` 和一次短的 NVE 测试，就先跑。

## 2. 系综与协议

| 目的 | `mflow md` 选项 | 为什么 |
|---|---|---|
| T、P 下的平衡结构 / 密度 | `-protocol equilibrate -baro iso`（液体、非晶、立方） / `aniso`（正交非立方） / `tri`（三斜、剪切） | NPT 让体积平衡，再在平均体积上 NVT |
| 扩散、电导、黏度 | `-protocol transport` | 生产段 NVE；恒温器会扰动动力学，Langevin 尤其压低扩散 |
| VDOS | `-protocol transport -dump 5`（1 fs 步长） | 帧间隔要 ≤ 25 fs 才能分辨到 20 THz |
| 势函数是否守恒、dt 是否合适 | `-ensemble nve` 短跑 | NVE 漂移闸门 < 5e-4 eV/atom/ps |
| 退火 / 淬火 / 非晶 | `-protocol anneal` / `quench` | 报告给出 K/ps；比较结果时报告速率 |
| 熔点 | `-protocol melt` | 见 `mflow-property-protocols` |
| 0 K 结构 | `-protocol minimize`（或 `mflow relax`） | |
| 起始结构很差（重叠、高能） | 先 `minimize`，再 `-thermostat csvr` 短平衡 | 稳健；**不要用 Berendsen 做生产** |

恒温器：
- `nh`（默认）：确定性，正确的正则系综；小体系或刚性振动模式可能非遍历。
- `csvr`（Bussi）：正确的正则系综，平衡快，对动力学扰动小——是多数情况下的安全选择。
- `langevin`：局部随机力，平衡最稳，但**改变动力学**，不能用于扩散或 VDOS。

## 3. 数字

| 参数 | 经验值 | 怎么验证 |
|---|---|---|
| `-dt` | 1 fs；含 H 或 T > ~2000 K 用 0.5–1 fs；不含 H 的金属/氧化物 ≤ 2 fs | 同一结构 NVE 跑 dt 和 dt/2，漂移都要过闸门 |
| 升温 | 从低温（~50 K）开始，或用 anneal 的升温段，不要直接在高温起跑 | |
| 弛豫 `-fmax` | 筛选 0.02–0.05 eV/Å；精细结构 0.005–0.02；收紧 10 倍复查一次 | 到达迭代上限 ≠ 收敛（报告 `minimize: forces`） |
| `-tdamp` | 100·dt（默认） | 太小温度振荡，太大平衡慢 |
| `-pdamp` | 1000·dt（默认） | 同上 |
| 盒子 | 每个方向 ≥ 2·r_max（MFlow 会 warn）且 ≥ 10 Å；扩散至少 20 个可动离子；熔点 ~1e4 原子 | 两种尺寸结果一致 |
| GPU 负载 | 每卡 1000–4000 原子 | `ns_per_day` |
| 平衡 | `-skip 0.2` 起步；报告 `equilibrated (half vs half)` 必须 pass | 前后两半差 < 3 个分块误差 |
| 生产长度 | 让关心的量的分块误差（`block_averages.*.se`）小于你需要的精度；`n_eff` ≥ ~20 | |
| `-dump` | 结构/RDF：每 100–1000 步；VDOS：5–10 步；总帧数 ~1e4 以内 | |

## 4. 验证清单（跑之前）

- [ ] 元素都在模型里（`mflow md` 会检查），head 正确（多头模型）
- [ ] `mflow validate dimer` 没有 fail（短程墙完整）
- [ ] `-dry` 看过 `in.lammps`：系综、阶段、温度、压强单位（bar）对
- [ ] 一致性检查（run 0）会跑——**不加 `-nocheck`**
- [ ] 对输运：生产段是 NVE 或弱耦合恒温器

## 5. 常见误区

| 误区 | 后果 | 正确做法 |
|---|---|---|
| 用 Langevin 跑扩散 | D 偏小 | `-protocol transport` |
| NVT 用初始结构的体积 | 压强 GPa 量级，结构失真 | 先 NPT（equilibrate） |
| 一个 seed 一条轨迹报 D | 误差被低估 | 3 个以上 seed，分块误差 |
| 熔点直接加热到熔化 | 过热，偏高 10–20 % | 两相共存（melt） |
| dt 只看"没崩" | 漂移悄悄破坏统计 | NVE 漂移闸门 |
| 盒子比 2·r_max 小 | 原子和自己的镜像作用 | 扩胞 |
| 从爆炸过的轨迹续跑 | 继承了坏构型 | 回到最后一个健康的 restart 或重新开始 |
| 跑完就当平衡了 | 统计含漂移 | 看 `equilibrated (half vs half)` |

**中止条件**（MFlow 的 `fix halt` 在 T > 5 × 目标时停；更早发现可看报告）：T 偏离目标 > 200 K 且不回落、NPT 体积超过初始的 2 倍或小于 0.2 倍、出现 NaN。恢复顺序：dt 降到 0.5 fs → 放慢升温 → 怀疑构型超出势函数的训练范围（`mflow-failure-triage`）。

出处与改写说明见 `references/SOURCES.md`。
