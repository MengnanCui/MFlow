---
name: mflow-property-protocols
description: Recipes for computing material properties with a MACE potential through MFlow — melting point by solid-liquid coexistence, diffusion coefficients and Arrhenius fits, RDF and coordination numbers, vibrational density of states, melt-quench amorphous structures, equation of state, phonons and defect energies. Use when the user wants a number for a property, not just a stable MD run, and needs to know how to set it up, how long to run, and how to judge and report the result.
---

# 性质计算协议（经验）

先确认势函数通过了 `mflow-potential-acceptance` 的相应层级。每个性质都要报告：**方法、体系大小、长度、误差棒、和参考值的差**。

## 熔点：固液共存

```bash
mflow md -in slab.xyz -deploy deploy -protocol melt -temp <T猜> -melt_temp <1.5×T猜> -steps 200000 -outdir tm_<T猜>
```

1. 盒子：正交，沿 z 拉长（Lz ≈ 2–4·Lx），几千到 1e4 原子；z 方向是界面法向。
2. `-temp` 取猜测值；`-melt_temp` 要高于熔点、低于沸点（`melt: volume` fail = 沸腾了）。
3. 读 `melt: phases`：
   - pass：两相共存，`T_melt_estimate` ± 分块误差就是结果；
   - 全液：T 太高，降低 `-temp`；全晶：升高；
   - "one phase only"：液半没熔（提高 `-melt_temp`/延长）或晶体也熔了（`-temp` 超过过热极限，约 1.2 T_m）。
4. 换 2–3 个起始温度，结果都收敛到同一 T 才算数；换更大的盒子检查有限尺寸效应（每边至少 20 Å）。
5. 另一种常见做法（AtomisticSkills）是 NVT 热化后 NVE 生产、取最后 5 ps 的平均 T，并要求界面区域 40–60 % 为固相；MFlow 用 NPH（压力也平衡），判断依据相同：两相都在，T 才是 T_m。

## 扩散系数

```bash
mflow md -in liquid.xyz -deploy deploy -protocol transport -temp T -steps 500000 -dump 100 -outdir D_T
```

1. 生产段 NVE（transport 协议）；`diffusive regime` 必须 pass（MSD 双对数斜率 0.85–1.15）。
2. 看 `diffusion_axes_cm2_s`：各向同性体系的三个分量应一致。
3. 误差：3 个以上 seed（`-seed`），报告均值 ± 标准误。
4. 有限尺寸：立方盒可用 `md-report -viscosity η`（η 用同一模型算，mPa·s）做 Yeh–Hummer 修正；否则至少两个盒子大小。
5. Arrhenius：4–6 个温度，ln D 对 1/T 线性拟合给 E_a，误差要传播到 E_a；温度区间内不要跨过相变。
6. 长度：去掉平衡段（几 ps）后，跑到 D 的相对误差约 5–10 % 即可。

## RDF / 配位数

- 生产段 ≥ 几百个不相关帧；`-ref aimd.xyz` 时报告 `rdf_mae_vs_ref`（< 0.1 好）。
- 配位数 = g(r) 积分到第一极小；第一极小随温度移动，写明截断。
- 小盒子 RDF 只到半个最薄宽度（MFlow 自动截断）。

## VDOS

- `-protocol transport -dump 5`（1 fs 步长），生产段 ≥ 10 ps。
- 晶体和 `mflow validate phonon` 的 DOS 对照：峰位应一致；小盒子只采样少数 q 点，峰高会不同。

## 非晶（熔融–淬火）

1. 至少 100 原子（更好几百）；先在远高于 T_m（常见 T_m + ~1000 K）下平衡液体约 10 ps（`-protocol equilibrate`），确认 MSD 在增长、RDF 无长程峰。
2. `-protocol quench -temp <液体T> -temp_end 300 -ensemble npt`，速率 1–10 K/ps，报告 K/ps；
3. 至少两个淬火速率，性质（密度、配位、RDF）随速率的变化要报告；
4. 最后 300 K 下 `equilibrate`（csvr），再 `-protocol minimize`，然后统计；RDF 在长程趋于 1，配位数合理。

## 静态性质

| 性质 | 命令 | 注意 |
|---|---|---|
| V₀、B₀ | `mflow validate eos -in relaxed.xyz -ref "V0=..,B0=.."` | 先弛豫；±4 % 应变足够 |
| 声子 | `mflow validate phonon -supercell N` | 超胞每边 ≥ 10 Å；虚频先查是否没弛豫干净 |
| 空位 / 表面 | `mflow validate defect -surface 111` | 化合物需要化学势；表面用常规胞的 Miller 指数 |

## 长轨迹的可信度

几 ns 的轨迹，隔一段抽帧用委员会打分（`mflow calc -model a,b,c,d`）：后期 `Fdev_max` 上升说明轨迹走出了训练分布，后半段的性质不可信。不同泛函的能量不能直接混用比较。

## 报告模板

```
性质：…  方法：…（协议、系综、步长、长度）
体系：N 原子，盒子 …，seed 数 …
结果：值 ± 误差（误差来源：分块 / seed 间）
对照：参考值 … 来源 …，偏差 …%
可信度：哪些闸门通过；已知局限（有限尺寸、淬火速率…）
原始数据：<run_dir>/md_report.json
```

出处与改写说明见 `references/SOURCES.md`。
