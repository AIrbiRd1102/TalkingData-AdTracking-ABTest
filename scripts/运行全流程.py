"""
运行全流程.py —— 平铺大步骤管道的极简顺序入口（设计优先的完整 A/B 测试生命周期）

按"正确顺序"依次调用 8 个大步骤模块，不做任何框架封装。
每个步骤模块独立、可读、可单独重跑；步骤之间靠 data/prepared/*.parquet 与各个 JSON 交接。

流程顺序（设计在最前，且不偷看结果）：
  01_问题定义与指标假设  →  data/01_问题定义与指标假设.json（规划假设，不看数据）
  02_实验设计            →  data/02_实验设计.json + 2 csv + figures/02_实验设计图.png
  03_数据接入与准备      →  data/prepared/*.parquet
  04_分流有效性与质量校验→  data/04_分流有效性与质量校验.json + csv + figures/04_分流有效性检验图.png
  05_显著性检验与效应量  →  data/05_显著性检验与效应量.json + csv
  06_序贯监控与安全护栏  →  data/06_序贯监控与安全护栏.json + 2 csv + figures/06_序贯监控护栏图.png
  07_因果推断与稳健性    →  data/07_因果推断与稳健性.json + 4 csv + figures/07_因果推断图.png
  08_报告可视化与决策    →  data/08_报告可视化与决策.json + figures/08_汇总仪表盘图.png

注：原第 ⑧ 步「异动归因（Oaxaca-Blinder 分解）」已移除。原因：其分析粒度是
"一天内的小时桶"而非时间序列，在本数据 6 小时窗口下无法支撑"异动"结论，
且在 4 天全量上会产出单格 -90pp 这类失真值。详见 README「方法取舍」。

运行（从项目根目录执行）：
  python scripts/运行全流程.py                          # 默认 full_csv 源（Spark 直读全量 CSV）
  ABTEST_DATA_SOURCE=csv python scripts/运行全流程.py   # pandas 直读 CSV 子集（冒烟调试）
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import 公共工具 as C
import importlib

STEPS = [
    "01_问题定义与指标假设",
    "02_实验设计",
    "03_数据接入与准备",
    "04_分流有效性与质量校验",
    "05_显著性检验与效应量",
    "06_序贯监控与安全护栏",
    "07_因果推断与稳健性",
    "08_报告可视化与决策",
]


def main():
    """按顺序 import 并执行 8 个步骤模块。

    【入口约定（本项目不成文的接口契约）】
      每个步骤模块必须暴露 `main()` 或 `run()` 之一作为入口，且【不接受必需参数】
      （配置统一从 公共工具 / 环境变量读取）。这里按 main 优先、run 兜底自适应调用。
      新增步骤时：把模块名加进 STEPS，并实现 main() 或 run() 即可。
    【为什么用 importlib 而不是直接 import】
      ① 模块名含中文与数字前缀（如 "04_分流有效性与质量校验"），无法写成
         普通的 `import 04_...` 语句（不是合法标识符）；
      ② 顺序写在列表里，流程顺序一眼可见，且便于单步注释掉做调试。
    【架构】步骤之间【不通过内存传参】，各自读/写 data/ 下的 parquet 与 json，
      因此任一步骤都能独立重跑：注释掉前几步、只跑最后一步也能出结果。
    """
    C.hr("AB 测试平铺大步骤管道启动（按设计优先顺序执行 8 个模块）")
    for name in STEPS:
        mod = importlib.import_module(name)
        C.hr(f"▶ 执行 {name}")
        # 每个步骤模块暴露 main() 或 run() 作为入口，自适应调用
        entry = getattr(mod, "main", None) or getattr(mod, "run", None)
        entry()
    C.hr("✅ 全部 8 个大步骤完成")


if __name__ == "__main__":
    main()
