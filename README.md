# XAS_analysis

用于处理 2026-03-27 MEX XAS 数据的一组分析脚本与 notebook，当前主要目标是按照实验分析流程完成：

- 用一条测试 spectrum 确定 flatten 参数
- 在单一电位下检查 raw data、转换 EXAFS 并做 convergence test
- 对已经确认收敛的不同电位数据做平均与对比

## 实际工作流

这个仓库当前最核心的不是“通用脚本库”，而是一套偏实验驱动的分析流程。`process_data.ipynb` 基本就是这套流程的主入口。

### 1. 先用测试 spectrum 确定 flatten 参数

先挑一条代表性的测试谱，反复调整 flatten / normalization 参数，直到：

- pre-edge / post-edge 拟合看起来合理
- normalized `mu(E)` 与 flattened `mu(E)` 形状稳定
- `e0`、`edge_step` 等关键量在预期范围内

在当前 notebook 里，这一步主要通过 `flatten_transmission_frame()` 和 `extract_exafs_transmission_frame()` 试参数，当前常用参数例如：

```python
e0=13470
pre1=-90
pre2=-50
norm1=210
norm2=730
nnorm=2
rbkg=1.0
ft_kmin=2
ft_kmax=8
ft_kweight=2
```

一旦这组参数在测试 spectrum 上确定，后面的同批数据默认沿用，不再逐条重新调。

### 2. 在单一电位下做 convergence test

确定参数后，先选一个电位，把这个电位下的一组重复 scan 全部拿出来检查。

这一步在 notebook 中分成三个子步骤：

- 先把所有 raw / flattened 谱线叠加画出来，检查有没有异常 scan、跳点或明显偏离的数据
- 用同一组固定参数把所有 scan 都转换成 EXAFS，查看 `k` 空间和 `R` 空间结果
- 对累计平均后的 `chi(k)` 与 `|chi(R)|` 做 convergence test，判断需要多少条 scan 才能稳定

当前 notebook 的做法是：

- 先构造 `available_scan_nums`
- 用 `flatten_transmission_frame()` 画某个电位下全部 flattened 曲线
- 再用 `extract_exafs_transmission_frame()` 得到每条 scan 的 `chi(k)` / `chi(R)`
- 然后把前 `N` 条 scan 逐步平均，和最终全平均结果比较
- 用相对 RMS 作为收敛指标，当前 notebook 里示例阈值是 `< 5%`

也就是说，这一步的核心问题是：当前电位下到底要平均多少条 scan，结果才算稳定。

### 3. 对已收敛的不同电位正式比较

如果上一步表明某个电位的数据已经收敛，就继续把不同电位的数据分别平均，然后画在一起比较。

当前 notebook 的思路是：

- 为每个电位指定一个 scan range
- 用同一组 EXAFS 参数把该 range 内的 scan 全部转换
- 在统一的 `k` 网格上做平均
- 再对平均后的 `chi(k)` 做 FT，得到平均后的 `chi(R)`
- 最后把不同电位的 `|chi(R)|` 叠加绘图，作为正式结果比较

`average_exafs_scans_in_range()` 这个函数就是为这一步服务的。

## 快速开始

### 1. 在 notebook 里按工作流分析

推荐优先使用 [`process_data.ipynb`](/media/Computation_bkp/Haoyu_MEX_XAS_20260327/XAS_analysis/process_data.ipynb)。

notebook 当前大致分成三段：

- flatten 参数测试
- 单一电位的 convergence 检查
- 多个电位的平均后 EXAFS / FT 对比

如果你是在继续同一批实验数据，通常只需要：

1. 先确认测试 spectrum 得到的参数还适用
2. 为目标电位设置 `requested_scan_nums`
3. 检查 raw/flattened 数据是否存在异常 scan
4. 运行 EXAFS 转换和 convergence test
5. 如果收敛，再把不同电位的 scan range 填进 `range_group` 做正式比较

### 2. 用 Python API 处理透射谱

输入数据需要至少包含这三列：

- `energy_eV`
- `i0_nanoamps`
- `i1_nanoamps`

示例：

```python
import pandas as pd
from larch_xas import flatten_transmission_frame, extract_exafs_transmission_frame

df = pd.read_csv("your_scan.csv")

flat = flatten_transmission_frame(
    df,
    e0=13470,
    pre1=-90,
    pre2=-50,
    norm1=210,
    norm2=730,
    nnorm=2,
)

exafs = extract_exafs_transmission_frame(
    df,
    e0=13470,
    pre1=-90,
    pre2=-50,
    norm1=210,
    norm2=730,
    nnorm=2,
    rbkg=1.0,
    ft_kmin=2,
    ft_kmax=8,
    ft_kweight=2,
)
```

主要返回字段：

- `flat["energy"]`, `flat["mu"]`, `flat["norm"]`, `flat["flat"]`
- `flat["pre_edge"]`, `flat["post_edge"]`, `flat["e0"]`, `flat["edge_step"]`
- `exafs["k"]`, `exafs["chi"]`, `exafs["chi_kw"]`
- `exafs["r"]`, `exafs["chir_mag"]`, `exafs["chir_re"]`, `exafs["chir_im"]`

### 3. 从 MDA 文件导出 `chi(k)` / `chi(R)`

`extract_chi_r.py` 适合快速把原始 `.mda` 文件变成可直接查看和进一步拟合的文本结果：

```bash
python3 extract_chi_r.py raw_data/MEX1_84446.mda \
  --output-dir exafs_output \
  --monitor i1 \
  --fluor-detectors 16,17,18,19 \
  --e0 13470 \
  --kmin 2 \
  --kmax 12 \
  --kweight 2
```

输出目录默认包含：

- `mu_normalized.csv`
- `chi_k.csv`
- `chi_r.csv`
- `mu_e.svg`
- `chi_k.svg`
- `chi_r.svg`
- `summary.txt`

## 当前数据约定

从代码和 notebook 看，当前流程默认采用以下 MEX 通道约定：

- 能量探测器索引：`0`
- `I0` 探测器索引：`2`
- `I1` 探测器索引：`3`
- 荧光通道默认求和：`16,17,18,19`

如果你的 beamline 导出格式或通道编号不同，需要在脚本中调整这些索引，或者扩展成命令行参数。

## Notebook 中当前对应关系

`process_data.ipynb` 里目前大致可以这样理解：

- 前半段：用单条测试 spectrum 调 flatten / EXAFS 参数
- `# Check convergence` 部分：针对单一电位检查多条重复 scan
- `## Plot converted EXAFS`：把该电位下每条 scan 转换到 `k` / `R` 空间
- `## Convergence test for an electric potential data`：用累计平均结果判断收敛
- `# Plot EXAFS spectrum for various electric potential`：把不同电位平均后结果叠加比较

## 已确认的限制与注意事项

### 环境限制

- 在当前仓库默认 `python3` 环境下，`extract_chi_r.py` 因缺少 `mda_reader` 无法直接运行。
- 在当前仓库默认 `python3` 环境下，`larch_xas.py` 因缺少 `larch` 无法直接 import。

### 代码层面的使用约束

- `larch_xas.py` 目前只处理透射模式输入，即通过 `mu = ln(I0 / I1)` 计算吸收。
- `extract_chi_r.py` 目前面向 MEX `.mda` 文件和固定 detector index，通用性有限。
- `requirements.txt` 是一个较大的整环境导出文件，不是最小依赖清单，移植到新机器时可能比实际需要更重。
- notebook 里包含绝对路径，例如原始数据目录 `/media/Computation_bkp/Haoyu_MEX_XAS_20260327/xas_raw_data_20260327`，换机器后需要手动修改。
- notebook 里的 scan range、样品标签和电位分组目前是手工维护的，适合当前实验批次，但还没有抽象成统一配置文件。

## 建议的后续整理方向

- 把 `mda_reader` 的来源补进仓库或 README
- 增加一个最小可复现示例数据文件
- 将 detector index 做成可配置参数
- 区分 `requirements.txt` 和 `requirements-min.txt`
- 为 `larch_xas.py` 增加简单单元测试


## 仓库结构

- `larch_xas.py`
  - 轻量 Python API。
  - 面向已经整理成表格/数组的透射数据。
  - 提供 `flatten_transmission[_frame]()` 和 `extract_exafs_transmission[_frame]()`。
- `extract_chi_r.py`
  - 命令行脚本。
  - 直接读取 MEX `.mda` 文件，输出 `mu(E)`、`chi(k)`、`chi(R)` 的 CSV 和 SVG。
- `process_data.ipynb`
  - 当前主要分析 notebook。
  - 按实际分析顺序组织：参数测试、单电位 convergence、不同电位对比。
- `old_test.ipynb`
  - 早期探索 notebook，保留了手工处理 MDA/XAS 的中间思路。
- `requirements.txt`
  - 当前环境导出的完整依赖列表。

## 依赖

这个仓库目前没有打包成可直接安装的 Python package，建议在单独虚拟环境中使用。

最少需要的核心依赖大致包括：

```bash
pip install numpy scipy pandas matplotlib seaborn xraydb xraylarch
```

如果你希望复现作者当时的完整 notebook 环境，也可以尝试：

```bash
pip install -r requirements.txt
```

## 外部依赖说明

当前仓库依赖以下仓库外模块，README 先明确列出来：

- `mda_reader`
  - `extract_chi_r.py` 直接 `from mda_reader import read_mda`
  - 该模块不在当前仓库中，需要你本地额外提供
- `larch`
  - `larch_xas.py` 依赖 `xraylarch`
- notebook 中还使用了 `mda`、`xraydb`、`seaborn` 等交互分析依赖

如果这些模块没有安装好，脚本会在 import 阶段直接失败。


## License

本仓库使用 [MIT License](./LICENSE)。

