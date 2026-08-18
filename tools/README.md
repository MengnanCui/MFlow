# tools/

体系专属的小工具，放不进通用的 `mflow.py`，但自己有用。

每个脚本直接从上一级的 `mflow.py` 里 import 读文件、日志、表格、配色和绘图风格
（`sys.path` 会自己指到仓库根目录），所以**要和 `mflow.py` 放在同一个仓库里跑**，
不能单独拷一个文件走。这样换来的是每个工具只剩它自己那点逻辑。

---

## `izo_formation.py` — In–Zn–O 形成能 vs In 比例

读一个已经用 MACE 算完能量的 xyz，按 In 比例分组画形成能。

```
E_f = [ E − (n_In/2)·E(In2O3) − n_Zn·E(ZnO) − Δn_O·μ_O ] / N_atoms      eV/atom
x   = n_In / (n_In + n_Zn)
Δn_O = n_O − (1.5·n_In + n_Zn)
```

x 一律**从成分数出来**，不依赖任何命名或标签。

```bash
# 参考能量直接给 xyz —— 工具自己数化学式单位，省掉人为除法
python tools/izo_formation.py -in izo_mace.xyz -eZnO zno_mace.xyz -eIn2O3 in2o3_mace.xyz

# 也可以直接给数字（eV / 化学式单位）
python tools/izo_formation.py -in izo_mace.xyz -eZnO -8.12 -eIn2O3 -30.4

# 有氧空位 / 富氧时必须给 μ_O，否则报错停下
python tools/izo_formation.py -in izo_mace.xyz -eZnO zno.xyz -eIn2O3 in2o3.xyz -muO -4.9 -xyz
```

| 参数 | 说明 |
|---|---|
| `-in` | 输入 xyz（`mflow.py calc` 的输出） |
| `-eZnO` / `-eIn2O3` | 数字按 **eV/化学式单位** 解释；给路径则读该 xyz 并自己除以单元数 |
| `-muO` | 氧的参考能量 eV/atom。任一结构偏离化学计量而没给它 → 报错退出 |
| `-prefix` | 指定能量标签，如 `mpa0_`。默认自动挑，`dft_`/`REF_` 这类参考标签排在最后 |
| `-out` | 输出前缀，默认跟输入同目录 |
| `-xyz` | 另写一份 xyz，`info` 里加 `izo_x_In` / `izo_Ef_atom`，原有标签一律不动 |

输出：终端一张按 x 分组的表（每个比例的最低 / 平均 / 极差 + 最低结构的帧号）、
`*_izo_ef.csv`（逐结构）、`*_izo_ef.png`（全部结构散点 + 每个比例最低点连线）。

**几个约定**

- 归一化是 **eV/atom**，表里保留 3 位有效数字。
- 完美化学计量的纯 ZnO、纯 In2O3 应当给出 `E_f = 0`，可以拿来自检参考能量对不对。
- 出现 In/Zn/O 以外的元素直接报错 —— 两个二元参考态配不平它们，硬算只会得到没有意义的数。
- 连的是**每个 x 上的最低点**，不是凸包下沿。两者在数据密的时候接近，但这条线不保证热力学稳定性。
