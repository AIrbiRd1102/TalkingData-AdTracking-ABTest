"""
02_实验设计.py —— 大流程步骤 ②：实验设计（必须在看实验结果之前完成）

本步是 A/B 测试流程里"设计优先"的核心体现：
  · 只消费 ① 写下的规划假设（α / 功效 / 假设基线 / 目标 MDE / 日均流量）；
  · 【不读取任何实验观测数据】（不偷看真实转化率与效应），否则就变成"先看结果再定设计"。
  · 输出：所需样本量、实验周期、随机分流方案、预注册分析计划。

覆盖：
  · 6.1 样本量规划：MDE ↔ 样本量 ↔ 功效 三者互推（基于假设基线 p0）
  · 6.2 流量分配与实验周期估算（基于假设日均流量）
  · 6.3 随机分流方案（确定性哈希分桶：只"定方案"，真实数据上的校验放到 ④）
  · 6.4 预注册分析计划（锁定检验方法、偷看策略、异常处置）

【本文件怎么读】run() 只做编排：6.1~6.4 各是一个函数，画图单独一个函数；
每个函数算自己的部分并返回结果，run() 负责组装 summary 与落盘。

数据来源：① 的规划假设（common 中的常量 / data/01_问题定义与指标假设.json）。
产物：data/02_实验设计.json + data/details/02_最小可检测效应曲线.csv / 02_流量分配与周期估算.csv
      + figures/02_实验设计图.png

运行：python 02_实验设计.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

import 公共工具 as C


def _sample_size_plan():
    """6.1 样本量规划：由规划假设（基线 p0 + 目标 MDE + 功效）反推所需样本量。

    【返回】dict，含 MDE 曲线表、目标 MDE 所需样本量、功效曲线等。
    ⚠️ 全程只用 ① 的【规划假设】，绝不使用真实转化率。
    """
    C.hr("6.1 样本量规划（基于假设基线 p0，非观测值）")
    alpha = C.ALPHA
    power = C.POWER
    p0 = C.ASSUMED_BASELINE_RATE          # 假设基线
    target_mde = C.TARGET_MDE_REL         # 目标相对提升
    p1_target = p0 * (1 + target_mde)
    solver = NormalIndPower()

    print(f"假设基线 p0 = {p0:.4%}；目标相对提升 = +{target_mde:.0%} → p1 = {p1_target:.4%}")
    lift_grid = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.00, 1.26]
    rows = []
    for rel in lift_grid:
        pp = p0 * (1 + rel)
        h = proportion_effectsize(pp, p0)
        n_req = solver.solve_power(effect_size=h, alpha=alpha, power=power,
                                   ratio=1.0, alternative="two-sided")
        rows.append({
            "relative_lift": rel, "p1": pp, "abs_diff": pp - p0,
            "cohens_h": float(h), "n_per_group": int(math.ceil(n_req)),
            "n_total": int(math.ceil(n_req)) * 2,
        })
    mde_table = pd.DataFrame(rows)

    h_target = proportion_effectsize(p1_target, p0)
    n_req_target = solver.solve_power(effect_size=h_target, alpha=alpha, power=power,
                                      ratio=1.0, alternative="two-sided")
    n_req_target = int(math.ceil(n_req_target))
    print(f"→ 检测 +{target_mde:.0%} 相对提升所需样本量 ≈ {n_req_target:,}/组 "
          f"（合计 ≈ {n_req_target*2:,}）@ 功效 {power:.0%}")

    n_curve = np.array([1e4, 2.5e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6])
    power_curve = np.array([
        solver.solve_power(effect_size=h_target, nobs1=n, alpha=alpha,
                           ratio=1.0, alternative="two-sided") for n in n_curve])
    return {"alpha": alpha, "power": power, "p0": p0, "target_mde": target_mde,
            "p1_target": p1_target, "h_target": float(h_target),
            "n_req_target": n_req_target, "mde_table": mde_table,
            "n_curve": n_curve, "power_curve": power_curve}


def _allocation_and_duration(n_req_target):
    """6.2 流量分配与实验周期估算：不同分配比例下，达到目标 MDE 需要多少天。

    周期 = 所需样本量 / 每天进入实验组的单元数，全部基于【规划假设】的日均流量。
    """
    C.hr("6.2 流量分配与实验周期估算（基于假设日均流量）")
    daily_traffic = C.ASSUMED_DAILY_TRAFFIC
    print(f"假设日均 (ip,app) 流量 = {daily_traffic:,}")
    allocations = [
        ("50/50 (2 臂)", 0.50),
        ("33/33/33 (3 臂)", 1 / 3),
        ("25/25/25/25 (4 臂)", 0.25),
        ("90/10 (小流量灰度)", 0.10),
    ]
    alloc_rows = []
    for name, trt_share in allocations:
        trt_per_day = daily_traffic * trt_share
        days = n_req_target / trt_per_day
        alloc_rows.append({
            "allocation": name, "treatment_share": trt_share,
            "treatment_per_day": int(trt_per_day),
            "days_for_target_mde": round(days, 1),
            "weeks_rounded": math.ceil(days / 7.0),
        })
        print(f"  {name}: 实验组 {trt_per_day:,.0f}/天 → 达目标 MDE 约需 {days:.1f} 天 "
              f"（≈ {math.ceil(days/7.0)} 周）")
    return alloc_rows


def _randomization_scheme():
    """6.3 随机分流方案：确定性哈希分桶（只"定方案"；真实数据上的校验见 ④）。

    用 2000 个示例 ip 演示"确定性 + 近似 50/50"，【不触及真实数据】。
    【返回】randomization_scheme dict（含 demo_A，供画图用）。
    """
    C.hr("6.3 随机分流方案（只定方案；真实数据上的质量校验见 ④）")
    print(f"方案：hash('{C.BUCKET_SALT}' + ip) % {C.N_BUCKETS} 个桶；"
          f"桶编号 < {C.N_BUCKETS*C.SPLIT_A_FRACTION:.0f} 归 A(对照)，其余归 B(实验)。")
    print("特性：确定性（同 ip 永远同桶，可复现）、可并行多实验（不同 salt 正交）。")
    demo_ips = [f"user_{i}" for i in range(2000)]
    demo_b = np.array([C.bucket_of(ip) for ip in demo_ips])
    demo_A = int((demo_b < C.N_BUCKETS * C.SPLIT_A_FRACTION).sum())
    print(f"演示（2000 个示例 ip，非真实数据）：A={demo_A}, B={2000-demo_A} → "
          f"≈ {demo_A/2000:.1%} / {1-demo_A/2000:.1%}，验证分桶均衡。")
    return {"method": "deterministic_hash", "salt": C.BUCKET_SALT,
            "n_buckets": C.N_BUCKETS, "split_a_fraction": C.SPLIT_A_FRACTION,
            "demo_A_share": demo_A / 2000}, demo_A


def _preregistration(alpha):
    """6.4 预注册分析计划：锁定后不再因数据改动（检验方法 / 偷看策略 / 异常处置）。"""
    C.hr("6.4 预注册分析计划（锁定后不再因数据改动）")
    pre_reg = {
        "primary_test": "两比例 Z 检验（two-sided），α=" + f"{alpha}",
        "effect_size": "Cohen's h + 相对/绝对提升 + 95% CI",
        "peeking_policy": "固定节奏偷看 + O'Brien-Fleming α 消耗边界（见 ⑥），禁止 naive z>1.96 反复偷看",
        "outlier_policy": "用户级 (ip,app) 去重；converted=max(is_attributed)；不删异常转化",
        "covariate_adjustment": ("不使用 CUPED：其协变量须为 pre-period 且不受处理影响，"
                                 "本数据集无窗口前行为记录（n_clicks 为同期变量，会调掉处理效应）"),
        "validity_gate": "④ SRM/协变量均衡 不通过则结论作废（A/A 为流程自检，不构成数据闸门）",
    }
    for k, v in pre_reg.items():
        print(f"  - {k}: {v}")
    return pre_reg


def _plot(mde_table, n_req_target, target_mde, n_curve, power_curve, power, alloc_rows, demo_A):
    """画 ② 的 2×2 图：样本量-MDE 曲线 / 功效曲线 / 流量分配周期 / 分桶演示。"""
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))
    fig.suptitle(
        "Experiment Design (pre-data, planning assumptions only)\n"
        f"assumed baseline p0 = {C.ASSUMED_BASELINE_RATE:.4%} | target MDE = +{target_mde:.0%} | power = {power:.0%}",
        fontsize=13.5)
    ax = axes[0, 0]
    ax.plot(mde_table.relative_lift * 100, mde_table.n_per_group / 1e3, "o-",
            color="#4C72B0", lw=2, ms=6)
    ax.axhline(n_req_target / 1e3, color="#C44E52", ls="--", lw=1.6,
               label=f"required n = {n_req_target/1e3:.0f}k")
    ax.axvline(target_mde * 100, color="#55A868", ls=":", lw=1.8,
               label=f"target MDE = +{target_mde:.0%}")
    ax.set_yscale("log")
    ax.set_xlabel("Target relative lift (%)")
    ax.set_ylabel("Required n per group (thousands, log)")
    ax.set_title("(1) Sample size vs. MDE (power=0.80)")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    ax = axes[0, 1]
    ax.plot(n_curve / 1e3, power_curve, "o-", color="#DD8452", lw=2, ms=6)
    ax.axhline(power, color="#8172B2", ls="--", lw=1.4, label=f"power = {power:.0%}")
    ax.axvline(n_req_target / 1e3, color="#C44E52", ls="--", lw=1.4,
               label=f"required n = {n_req_target/1e3:.0f}k")
    ax.set_xscale("log")
    ax.set_xlabel("n per group (thousands, log)")
    ax.set_ylabel("Statistical power")
    ax.set_title("(2) Power curve at target effect")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    ax = axes[1, 0]
    labels = [a["allocation"].split(" (")[0] for a in alloc_rows]
    bars = ax.bar(labels, [a["days_for_target_mde"] for a in alloc_rows], color="#4C72B0")
    ax.set_ylabel("Days to reach target MDE")
    ax.set_title("(3) Duration by traffic allocation")
    for b, v in zip(bars, [a["days_for_target_mde"] for a in alloc_rows]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.0f}d", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", labelrotation=15); ax.grid(alpha=0.3, axis="y")
    ax = axes[1, 1]
    ax.bar(["A (control)", "B (treatment)"], [demo_A, 2000 - demo_A], color=["#4C72B0", "#55A868"])
    ax.set_ylabel("count (demo 2000 ips)")
    ax.set_title("(4) Hash bucketing demo (deterministic ~50/50)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = C.FIG_DIR / "02_实验设计图.png"
    fig.savefig(fig_path, dpi=130)
    print(f"\nSaved figure: {fig_path}")
    return fig_path


def run():
    """② 主流程：6.1 样本量 → 6.2 分配周期 → 6.3 分流方案 → 6.4 预注册 → 落盘。

    本函数只用 ① 的规划假设，【不读取任何实验观测数据】。
    """
    C.hr("② 实验设计（MDE / 样本量 / 周期 / 分流方案 / 预注册）—— 仅用规划假设，不读观测数据")

    plan = _sample_size_plan()
    mde_table, n_req_target = plan["mde_table"], plan["n_req_target"]
    target_mde, p0, alpha, power = plan["target_mde"], plan["p0"], plan["alpha"], plan["power"]

    alloc_rows = _allocation_and_duration(n_req_target)
    randomization, demo_A = _randomization_scheme()
    pre_reg = _preregistration(alpha)

    summary = {
        "planning_inputs": {
            "alpha": alpha, "power": power,
            "assumed_baseline_rate": p0, "target_mde_relative": target_mde,
            "assumed_daily_traffic": C.ASSUMED_DAILY_TRAFFIC,
            "note": "以上均来自 ① 规划假设，未使用真实转化率（设计优先）",
        },
        "sample_size": {
            "target_mde_relative": target_mde,
            "p0": p0, "p1": plan["p1_target"], "cohens_h": plan["h_target"],
            "n_per_group_required": n_req_target,
            "n_total_required": n_req_target * 2,
            "n_per_group_by_lift": {
                f"{int(r*100)}%": int(v)
                for r, v in zip(mde_table.relative_lift, mde_table.n_per_group)},
        },
        "allocation_and_duration": alloc_rows,
        "randomization_scheme": randomization,
        "pre_registration": pre_reg,
        "control_app": C.CONTROL_APP, "treatment_app": C.TREATMENT_APP,
    }

    mde_table.to_csv(C.DETAILS_DIR / "02_最小可检测效应曲线.csv", index=False)
    pd.DataFrame(alloc_rows).to_csv(C.DETAILS_DIR / "02_流量分配与周期估算.csv", index=False)
    _plot(mde_table, n_req_target, target_mde, plan["n_curve"], plan["power_curve"],
          power, alloc_rows, demo_A)

    C.dump_json(summary, C.DATA_DIR / "02_实验设计.json")
    C.hr("✅ ② 实验设计完成（规划值，未用观测数据）")
    print(f"  产物: {C.DATA_DIR / '02_实验设计.json'}")
    print(f"        {C.DETAILS_DIR / '02_最小可检测效应曲线.csv'}")
    print(f"        {C.DETAILS_DIR / '02_流量分配与周期估算.csv'}")
    print(f"        {C.FIG_DIR / '02_实验设计图.png'}")
    return summary


if __name__ == "__main__":
    run()
