"""
05_显著性检验与效应量.py —— 大流程步骤 ⑤：对主指标做显著性检验与效应量估计

本步在"有效性闸门（④）"之后：只有分流可信，检验结论才站得住。
注意顺序：② 设计（定样本量/目标 MDE）→ ③ 接入 → ④ 校验 → 本步才检验，
所以这里是"用设计阶段锁定的方法，对真实结果做一次检验"，不是反过来。

覆盖：
  · 5.1 整体指标（用户级去重口径）+ 两比例 Z 检验 + Cohen's h + 95% CI
  · 5.2 观测功效（post-hoc，仅作复盘，绝不用于反推设计）

注：CUPED 已移除。CUPED 的协变量必须来自 pre-period 且不受处理影响，
本数据集无窗口前行为记录（唯一候选 n_clicks 为同期变量，会调掉处理效应本身），
因此 CUPED 在本数据上不可执行——项目约定不构造数据，故不强行实现。

【本文件怎么读】run() 只做编排：5.1 显著性检验、5.2 观测功效各是一个函数，
各自算自己的部分并返回结果；run() 负责组装 summary 与落盘。

数据来源：03 产出的 data/prepared/用户级面板.parquet。
产物：data/05_显著性检验与效应量.json + data/details/05_显著性检验明细.csv

运行：python 05_显著性检验与效应量.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

import 公共工具 as C


def _ci_for_proportion_diff(x1, n1, x0, n0, alpha=C.ALPHA):
    """两比例差 (p1 - p0) 的 Wald 置信区间（近似）。

    【输入】x1/n1 —— 实验组转化数/样本量；x0/n0 —— 对照组；alpha —— 显著性水平（0.05 → 95% CI）。
    【输出】(diff, ci_low, ci_high)，均为比例（非百分比）。
    【统计原理 · 关键细节】这里用的是【非合并】标准误
        SE_unpooled = √[ p1(1-p1)/n1 + p0(1-p0)/n0 ]
      而 ⑤ 的 Z 检验用的是【合并】标准误 √[ p̄(1-p̄)(1/n1 + 1/n0) ]。
      为什么不同？
        · 检验时我们站在 H₀ 的世界里（假设 p1 = p0 = p），合并估计最有效；
        · 构区间时不再假设 H₀ 成立，两组各有各的真实比例，应各自估计。
      一句话记忆：**检验用合并 SE，区间用非合并 SE**。
      后果：两者的半宽不同，"检验显著"与"CI 不含 0"在数学上并不严格等价
      （本项目数值接近，结论一致）。
    【为什么叫 Wald】用点估计 ± z·SE 的直接构造法，依赖正态近似；
      在 p 接近 0 或 1、或 n 很小时近似较差（可用 Wilson / Agresti-Coull 改进）。
      本项目 n 达 13 万量级、p 约 1.5%~4.7%，近似误差可忽略。
    """
    p1, p0 = x1 / n1, x0 / n0
    diff = p1 - p0
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)   # 非合并标准误
    z = norm.ppf(1 - alpha / 2)                                # 双侧 95% → 1.95996
    return diff, diff - z * se, diff + z * se


def _significance(ctrl, trt):
    """5.1 整体指标 + 两比例 Z 检验 + Cohen's h + 95% CI。

    【返回】dict，含两组样本量/转化数/转化率、绝对与相对提升、
    Z 统计量、p 值、Cohen's h、95% CI 与显著性判定。
    """
    C.hr("5.1 整体指标 + 两比例 Z 检验 + Cohen's h + 95% CI")
    n_c, x_c = len(ctrl), int(ctrl.converted.sum())
    n_t, x_t = len(trt), int(trt.converted.sum())
    p_c, p_t = x_c / n_c, x_t / n_t
    lift = (p_t - p_c) / p_c

    print(f"对照组 app{C.CONTROL_APP}: n={n_c:,}, 转化={x_c}, 转化率 p_c = {p_c:.6%}")
    print(f"实验组 app{C.TREATMENT_APP}: n={n_t:,}, 转化={x_t}, 转化率 p_t = {p_t:.6%}")
    print(f"绝对提升 = {p_t - p_c:.6%}；相对提升 = {lift:+.1%}")

    z_stat, p_value = proportions_ztest([x_t, x_c], [n_t, n_c], alternative="two-sided")
    h = proportion_effectsize(p_t, p_c)
    diff, ci_lo, ci_hi = _ci_for_proportion_diff(x_t, n_t, x_c, n_c)
    sig = p_value < C.ALPHA
    print(f"Z 检验: Z = {z_stat:.4f}, p = {p_value:.3e}；Cohen's h = {h:.4f}")
    print(f"95% CI of (p_t - p_c): [{ci_lo:.6%}, {ci_hi:.6%}]")
    print(f"结论: p {'< ' if sig else '>= '}{C.ALPHA} -> {'统计显著' if sig else '不显著'}")
    print(f"效应量解读: Cohen's h={h:.4f} 属"
          f"{'小' if abs(h) < 0.2 else ('中' if abs(h) < 0.5 else '大')}效应"
          f"（低频转化场景绝对差小但相对提升大，属正常）")
    return {"control_n": n_c, "control_conv": x_c, "control_rate": p_c,
            "treatment_n": n_t, "treatment_conv": x_t, "treatment_rate": p_t,
            "abs_lift": p_t - p_c, "relative_lift": lift,
            "z_stat": float(z_stat), "p_value": float(p_value), "cohens_h": float(h),
            "ci95_low": float(ci_lo), "ci95_high": float(ci_hi), "significant": bool(sig)}


def _observed_power(sig_pack):
    """5.2 观测功效（post-hoc）—— 仅作复盘，绝不用于反推设计。

    ⚠️【方法学纪律】观测功效只作【复盘】，绝不能反推设计。
    用实测效应反算"我当初只需要多少样本"是【循环论证】：效应大 ⇒ p 小 ⇒ 功效高，
    它没有提供任何超出 p 值的新信息。设计只能用 ② 的规划假设。

    ⚠️【已知近似】这里传 nobs1=min(n_c,n_t) 且 ratio=1.0，即按【等臂】计算；
    而真实两组不等臂（158,175 vs 138,534）。严格写法应为
        solve_power(effect_size=h, nobs1=n_c, alpha=ALPHA, ratio=n_t/n_c, ...)
    两者在本项目下都得到 power ≈ 1.0，结论相同，故保留原写法以免改变输出；
    但在正式报告中应注明"按等臂近似"（已写入产物的 power_note 字段）。
    """
    C.hr("5.2 观测功效（post-hoc，仅复盘，不用于定设计）")
    h = sig_pack["cohens_h"]
    n_c, n_t = sig_pack["control_n"], sig_pack["treatment_n"]
    analysis = NormalIndPower()
    achieved_power = analysis.solve_power(effect_size=h, nobs1=min(n_c, n_t),
                                          alpha=C.ALPHA, ratio=1.0, alternative="two-sided")
    print(f"基于观测效应量 Cohen's h = {h:.4f}，实际每组样本量 {min(n_c,n_t):,} -> "
          f"观测功效 ≈ {achieved_power:.3f}")
    n_required = int(math.ceil(analysis.solve_power(
        effect_size=proportion_effectsize(C.ASSUMED_BASELINE_RATE*(1+C.TARGET_MDE_REL),
                                          C.ASSUMED_BASELINE_RATE),
        alpha=C.ALPHA, power=C.POWER, ratio=1.0, alternative='two-sided')))
    print(f"对照 ② 设计：目标 MDE = +{C.TARGET_MDE_REL:.0%}，"
          f"假设基线 {C.ASSUMED_BASELINE_RATE:.4%} 下需 ≈ {n_required:,} / 组")
    # 注：CUPED 已从本步骤移除。
    # 原因：CUPED 要求协变量 X 来自 pre-period 且不受处理影响；本数据集唯一的连续候选
    # 协变量 n_clicks 是"实验期同期"变量（受 app 影响），用它调整会把处理效应本身调掉
    # （over-adjustment）。项目要求不构造数据，故本数据集不存在合法的 pre-period 协变量，
    # CUPED 在此不可执行——与其给出无效数字，不如不做。
    return {"achieved_power": float(achieved_power),
            "power_note": "achieved_power 按【等臂近似】计算（nobs1=min(n_c,n_t), ratio=1.0）；"
                          "真实两组不等臂，严格写法应传 ratio=n_t/n_c（本项目两者均≈1.0）"}


def run():
    """⑤ 主流程：5.1 显著性检验 → 5.2 观测功效 → 组装 summary 落盘。"""
    C.hr("⑤ 显著性检验与效应量（Z 检验 / Cohen's h / CI / 功效）")
    prepared, ctrl, trt = C.load_arms()

    sig_pack = _significance(ctrl, trt)
    power_pack = _observed_power(sig_pack)

    summary = dict(sig_pack)
    summary["achieved_power"] = power_pack["achieved_power"]
    # 【显式化】把上面注释里的"等臂近似"假设也写进产物，避免下游只看数字
    # 却不知道它是在 ratio=1.0 下算出来的（真实两组 158,175 vs 138,534 不等臂）。
    # 注意：这个字段只做标注，不改变 achieved_power 的数值。
    summary["power_note"] = power_pack["power_note"]

    pd.DataFrame([
        {"metric": "control_rate", "value": sig_pack["control_rate"]},
        {"metric": "treatment_rate", "value": sig_pack["treatment_rate"]},
        {"metric": "abs_lift", "value": sig_pack["abs_lift"]},
        {"metric": "relative_lift", "value": sig_pack["relative_lift"]},
        {"metric": "z_stat", "value": sig_pack["z_stat"]},
        {"metric": "p_value", "value": sig_pack["p_value"]},
        {"metric": "cohens_h", "value": sig_pack["cohens_h"]},
        {"metric": "ci95_low", "value": sig_pack["ci95_low"]},
        {"metric": "ci95_high", "value": sig_pack["ci95_high"]},
        {"metric": "achieved_power", "value": power_pack["achieved_power"]},
    ]).to_csv(C.DETAILS_DIR / "05_显著性检验明细.csv", index=False)
    C.dump_json(summary, C.DATA_DIR / "05_显著性检验与效应量.json")

    C.hr("✅ ⑤ 显著性检验与效应量完成")
    print(f"  产物: {C.DATA_DIR / '05_显著性检验与效应量.json'}")
    print(f"        {C.DETAILS_DIR / '05_显著性检验明细.csv'}")
    return summary


if __name__ == "__main__":
    run()
