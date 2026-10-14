# 观察性数据的 A/B 测试方法学演练

TalkingData AdTracking · PySpark + pandas + statsmodels/scipy/sklearn

> ⚠️ **项目定位（务必读）**：本项目的「实验组/对照组」（app12/app18）由来自数据集中的不同 app **天然形成，属观察性对照，不是随机 A/B**。
> 所以它是用 A/B 测试的方法学工具的演练**不是**"做了 A/B 测试并上线"。

---

## 1. 项目与数据

- **数据集**：[TalkingData AdTracking](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection)（Kaggle），
  约 **1.85 亿行**广告点击日志，字段 `ip / app / device / os / channel / click_time / is_attributed`。
- **对照构造**：`app=12` 对照组、`app=18` 实验组，`is_attributed=1` 为转化。
- **分析口径**：用户级 `(ip, app)` 去重 → **296,709** 个分析单元。
  聚合规则 `converted = MAX(is_attributed)`、`n_clicks = SUM`、`os/device/channel` 取众数。

## 2. 技术栈

| 层 | 工具 | 用途 |
|---|---|---|
| 计算引擎 | PySpark 4.2.0（JDK 25 + Hadoop 3.3.x + winutils） | 全量读取与分布式聚合 |
| 接入 | pandas + numpy | 分块聚合、parquet 读写 |
| 统计 | statsmodels / scipy / sklearn / numpy | Z 检验、功效、SRM、OBF、IPW/DID/E-value |
| 可视化 | matplotlib + Tableau Public（可选） | 仪表盘 |

## 3. 八步流程

每个步骤是独立可读的 `.py`，步骤之间用 `data/prepared/*.parquet` 与各自 JSON 交接，
**可单独重跑、可独立调试**。`scripts/运行全流程.py` 只是按序调用的极简入口，无框架封装。

| 步骤 | 模块 | 内容 |
|---|---|---|
| ① | `01_问题定义与指标假设.py` | 业务目标 / 主·护栏·次要指标 / H0-H1 / 决策规则 |
| ② | `02_实验设计.py` | 样本量 / 周期 / 哈希分桶方案 / 预注册 |
| ③ | `03_数据接入与准备.py` | CSV→Parquet 缓存 + Spark 直读 → 聚合 → 落 `data/prepared/` |
| ④ | `04_分流有效性与质量校验.py` | SRM / 分桶质量 / 协变量均衡 |
| ⑤ | `05_显著性检验与效应量.py` | 两比例 Z 检验 / Cohen's h / 95%CI / 观测功效 |
| ⑥ | `06_序贯监控与安全护栏.py` | 序贯偷看 OBF 修正 / 护栏单侧劣效检验 |
| ⑦ | `07_因果推断与稳健性.py` | SUTVA / 朴素 / PSM(诊断) / IPW / DID / E-value |
| ⑧ | `08_报告可视化与决策.py` | 汇总主报告 + 兜底仪表盘 + 决策建议 |

## 4. 关键结论（全量 app12/18，用户级口径）

**整体指标**：对照 1.4939%（158,175 单元）vs 实验 4.6833%（138,534 单元），
绝对提升 **+3.1894pp**、相对提升 **+213.5%**；Z = 50.95，p ≈ 0，Cohen's h = 0.1912（小效应）。

**有效性闸门（④）**：
- SRM 卡方 = 1300.16，p ≈ 1.04e-284 → **失败**（观测 53.31% : 46.69%，非 50:50）
- 协变量最大占比差：os 0.0205 / device 0.0207 / **channel 0.7030** → 均失衡
- 哈希分桶机制本身可用（确定性一致；正交性 p=0.9054、方案级 50:50 p=0.2738）；
  均匀性 p=0.0234 略低于 0.05 属**大样本卡方检验的过度敏感**，不代表方案坏掉

**序贯与护栏（⑥）**：7 次偷看名义假阳性 15.2% → OBF 修正后 4.4%；
护栏（人均点击，单侧劣效，容差 −5%）相对差 **−25.59%**，90% CI 上界 −22.08% → **击穿**。

**因果推断（⑦）**：PS 的 CV-AUC 0.9894、共同支撑内仅 16.15% 样本 → 重叠不足、不可识别；
SUTVA 违反（唯一 ip 口径 72.39%、分析单元口径 83.98%）；
IPW 调整后关联不降反升（+213.5% → **+231.4%**），说明可观测混杂**放大**而非制造了表观差异；
去污染敏感性分析中效应缩水 67.36%（+3.1894pp → +1.0411pp）。

**决策（⑧）**：**不采纳为因果结论**（主指标显著但 ④ 闸门失效 + ⑥ 护栏击穿），结论限定为关联。

## 5. 运行

```powershell
# 一条命令跑通八步（默认 full_csv：Spark 直读原始 CSV）
python -u scripts/运行全流程.py
```

- **数据入口**：默认 `ABTEST_DATA_SOURCE=full_csv`，需指定 `train.csv`。首次跑会顺手生成同目录 `train.parquet` 缓存，
  之后重跑直接读缓存。
- **冒烟调试**：`$env:SAMPLE_ROWS=200000` 只读前 20 万行（不走缓存）。
- **产物**：`data/0N_*.json`（各步小结）、`data/details/*.csv`、`figures/*.png`。

### 5.1 所需配置

**环境依赖**

| 项 | 要求 | 说明 |
|---|---|---|
| Python | 3.12.x | 建议 conda 独立环境 |
| JDK | 17+ | Spark 运行所需，设 `JAVA_HOME` |
| Hadoop | 3.3.x + `winutils.exe` + `hadoop.dll` | Spark 在 Windows 运行所需，设 `HADOOP_HOME` |
| Spark | 4.x（`pip install pyspark`） | 分布式读取与聚合 |

**Python 包**：见 `requirements.txt`，`pip install -r requirements.txt` 一次装齐。

**`scripts/配置.py`**（需自行创建，已被 `.gitignore` 忽略、不会上传）：只负责 Spark 运行环境，无数据库配置。最小可用模板：

```python
import os

_SPARK_ENV_DEFAULTS = {
    "JAVA_HOME": "D:/Program Files/.../jdk-xx",   # 改成你的 JDK 安装路径
    "HADOOP_HOME": "D:/hadoop",                   # winutils.exe 与 hadoop.dll 所在目录
    "TEMP": os.environ.get("TEMP", "C:/Windows/Temp"),
    "TMP": os.environ.get("TMP", "C:/Windows/Temp"),
}


def configure_spark_env():
    """在 import pyspark 之前调用（03 顶部已调用）。"""
    import sys
    for k, v in _SPARK_ENV_DEFAULTS.items():
        os.environ.setdefault(k, v)
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
```

**环境变量**

| 变量 | 默认 | 说明 |
|---|---|---|
| `ABTEST_DATA_SOURCE` | `full_csv` | `full_csv`=Spark 直读原始 CSV；`csv`=pandas 直读 CSV 子集 |
| `ABTEST_RAW_DIR` | `D:/AB测试数据` | 原始数据目录 |
| `ABTEST_RAW_CSV` | `train.csv` | 原始文件名 |
| `ABTEST_PARQUET_DIR` | 与 CSV 同目录 | 原始层 Parquet 缓存位置 |
| `SAMPLE_ROWS` | 空（全量） | 只读前 N 行冒烟调试 |
| `SPARK_DRIVER_MEMORY` | `6g` | driver 堆内存 |
| `SPARK_LOCAL_DIR` | `D:/spark_tmp` | shuffle 落盘目录 |

## 6. 文件结构

```
AB测试项目/
├── scripts/                    # 平铺大步骤代码
│   ├── 运行全流程.py            # 极简顺序入口
│   ├── 公共工具.py              # 常量（规划假设+分流参数）+ 统计函数 + 中间文件读写
│   ├── 配置.py                 # 本地配置（.gitignore 忽略，勿提交；所需配置见 §5.1）
│   ├── 汇总报告.py              # 生成 docs/ 下的 HTML 汇总报告
│   └── 01~08_*.py              # 八个步骤模块
├── data/
│   ├── prepared/               # ③ 产出的中间面板（④~⑧ 读它）
│   ├── 0N_*.json               # 各步小结产物
│   └── details/                # 各步明细 CSV
├── figures/                    # 各步图表
├── docs/                       # 最新汇总报告（其余文档本地保留、不随仓库分发）
├── LICENSE                     # MIT
├── requirements.txt            # Python 依赖
└── README.md
```

## 7. 局限

**非随机分流**：观察性对比，因果结论不可信。