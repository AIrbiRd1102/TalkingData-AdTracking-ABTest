"""
06_序贯监控与安全护栏.py —— 大流程步骤 ⑥：实验运行期的监控纪律与护栏

本步是"实验设计（②）里预注册的监控策略"的落地：
  · 6.1 序贯偷看（peeking）的假阳性膨胀 + O'Brien-Fleming α 消耗边界修正
  · 6.2 护栏指标单侧劣效检验（只查容差 δ 的【下界】），守护体验不被牺牲

为什么单独成步：监控与护栏是"实验进行中就要遵守的规则"，和"事后显著性检验（⑤）"
不同；把它独立出来，便于学习"边跑边看"时最容易踩的坑（反复偷看 → 假阳性膨胀）。

【本文件怎么读】run() 只做编排：6.1 偷看模拟、6.2 护栏检验、画图各是一个函数，
各自算自己的部分并返回结果；run() 负责组装 summary 与落盘。

数据来源：03 产出的 data/prepared/（用户级面板.parquet + 数据元信息.json）。
产物：data/06_序贯监控与安全护栏.json
      + data/details/06_序贯偷看假阳性率.csv / 06_护栏指标单侧劣效检验.csv
      + figures/06_序贯监控护栏图.png

运行：python 06_序贯监控与安全护栏.py
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
from scipy import stats

import 公共工具 as C

# 序贯偷看模拟参数（预注册的监控节奏）
N_SIMS = 4000      # 蒙特卡洛重复次数
K_LOOKS = 7        # 计划查看次数（本项目：7 天，每天偷看一次）


def _peeking_simulation(p_c, daily_traffic):
    """6.1 序贯偷看假阳性膨胀 + O'Brien-Fleming 边界修正。

    【做法】在 H₀ 为真的世界里（两组真实转化率都 = 对照组的 p_c）模拟实验进程：
    每天新增一批样本，累加成 7 个"查看点"，分别统计"每天偷看一次、见 |Z|>1.96 就停"
    的假阳性率（naive），与"用 OBF 边界"的假阳性率（obf），看后者能否回到名义 α 附近。
    【返回】peeking 结果 dict（含 obf 边界序列与两种 FPR）。
    """
    C.hr("6.1 序贯偷看假阳性膨胀 + O'Brien-Fleming 修正")
    z95 = stats.norm.ppf(1 - C.ALPHA / 2)
    n_per_arm_per_day = int(round(daily_traffic / 2.0))
    _rng = np.random.default_rng(C.SEED)
    # 在 H0 下模拟：两组真实率都 = p_c，看"每天偷看一次 naive z>1.96"会把假阳性率抬多高
    draws_c = _rng.binomial(n_per_arm_per_day, p_c, size=(N_SIMS, K_LOOKS))
    draws_t = _rng.binomial(n_per_arm_per_day, p_c, size=(N_SIMS, K_LOOKS))
    cum_c = np.cumsum(draws_c, axis=1)
    cum_t = np.cumsum(draws_t, axis=1)
    cum_n = n_per_arm_per_day * np.arange(1, K_LOOKS + 1)
    ph_c = cum_c / cum_n
    ph_t = cum_t / cum_n
    pp_ = (cum_c + cum_t) / (2 * cum_n)
    se_mat = np.sqrt(pp_ * (1 - pp_) * (2 / cum_n))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_mat = (ph_t - ph_c) / se_mat
    z_mat = np.nan_to_num(z_mat)
    naive_hit = np.abs(z_mat) > z95
    C_OBF = C.calibrate_obf_constant(K_LOOKS, C.ALPHA)
    k_idx = np.arange(1, K_LOOKS + 1)
    obf_boundary = C_OBF * np.sqrt(K_LOOKS / k_idx)
    obf_hit = np.abs(z_mat) > obf_boundary
    cum_naive = np.zeros(N_SIMS, dtype=bool)
    cum_obf = np.zeros(N_SIMS, dtype=bool)
    peek_rows = []
    for k in range(K_LOOKS):
        cum_naive |= naive_hit[:, k]
        cum_obf |= obf_hit[:, k]
        peek_rows.append({
            "look_k": k + 1, "obf_boundary": float(obf_boundary[k]),
            "fpr_naive": float(cum_naive.mean()), "fpr_obf": float(cum_obf.mean()),
        })
    peek_df = pd.DataFrame(peek_rows)
    fpr_naive_final = float(peek_df.fpr_naive.iloc[-1])
    fpr_obf_final = float(peek_df.fpr_obf.iloc[-1])
    print(f"偷看 {K_LOOKS} 次：名义法假阳性 {fpr_naive_final:.1%}"
          f"（膨胀 {fpr_naive_final / C.ALPHA:.1f} 倍）；"
          f"OBF 修正后 {fpr_obf_final:.1%}（≈名义 α）")
    return {"n_sims": N_SIMS, "looks": K_LOOKS,
            "n_per_arm_per_day": n_per_arm_per_day,
            "fpr_naive_final": fpr_naive_final, "fpr_obf_final": fpr_obf_final,
            "obf_constant_C": C_OBF, "obf_z_boundary": [float(v) for v in obf_boundary],
            "inflation_times": fpr_naive_final / C.ALPHA}, peek_df


def _guardrail_test(ctrl, trt):
    """6.2 护栏指标（人均点击次数）的单侧劣效检验。

    【为什么不用普通 t 检验，也不用双侧 TOST】
      · 普通检验的 H₀ 是"没有差异"：【不显著 ≠ 没变差】（可能只是样本不够），
        且它【不含容差概念】。护栏真正要回答的是"变差是否超过可接受的容差 δ"。
      · 双侧 TOST（等价检验）会【同时】检查下界 −δ 与上界 +δ，产出
        "等价 / 劣效 / 优效 / 证据不足"四态。但护栏的目的只有一个——【防止变差】；
        上界检查（要求实验组"不能显著变好"）对保护体验而言是【冗余】的：
        护栏指标显著变好不是风险，却会被 TOST 判成"不等价"而误伤。
      · 因此采用【单侧劣效检验】，只查【下界】一个方向：
          H₀: Δ ≥ −δ（未劣于容差，安全）   vs   H₁: Δ < −δ（劣于容差，击穿）
        拒绝 H₀（判"护栏被击穿"）的等价条件是：90% CI 的【上界】< −δ。
    【为什么"90% CI 上界 < −δ"就是单侧检验】
      90% 双侧 CI 的上界 = d̂ + t_{0.95}·SE，正是 α=0.05 下单侧的 95% 置信上界
      （t_{0.95} ≈ 1.6449）。"整个区间落在 −δ 以下" ⇔ "有 95% 把握 Δ < −δ" ⇔ 拒绝 H₀。
      所以这里的 90% 不是笔误，是单侧检验的标准配置。
    【返回】guardrail 结果 dict（含 guardrail_pass 布尔，供 ⑧ 决策使用）。
    """
    C.hr("6.2 护栏指标单侧劣效检验（守护体验不被牺牲）")
    MARGIN = C.GUARDRAIL_MARGIN
    xc = ctrl["n_clicks"].to_numpy(float)
    xt = trt["n_clicks"].to_numpy(float)
    m_c, m_t = xc.mean(), xt.mean()
    diff = m_t - m_c
    rel_diff = diff / m_c
    # Welch 标准误：两组方差不等（本项目 sd 1122 vs 613），不能用合并方差。
    # 自由度用 Welch–Satterthwaite 近似（本项目 ≈ 250,954，t 临界值已≈正态 1.6449）。
    se = math.sqrt(xc.var(ddof=1) / len(xc) + xt.var(ddof=1) / len(xt))
    df_welch = (xc.var(ddof=1) / len(xc) + xt.var(ddof=1) / len(xt)) ** 2 / (
        (xc.var(ddof=1) / len(xc)) ** 2 / (len(xc) - 1)
        + (xt.var(ddof=1) / len(xt)) ** 2 / (len(xt) - 1))
    t_crit_90 = stats.t.ppf(0.95, df_welch)
    ci_lo, ci_hi = diff - t_crit_90 * se, diff + t_crit_90 * se
    # 【⚠️ 已知近似 —— 显式标注，勿当作精确量】
    #   绝对差的 CI 直接除以【点估计】m_c 得到"相对差 CI"，这是 delta method 的
    #   一阶近似：它把 m_c 当作已知常数，【忽略了分母自身的抽样方差】。
    #   严格的做法是用 Fieller 定理或 bootstrap 联合重抽样 (mean_c, m_t)。
    #   本项目 m_c 基于 15.8 万单元、SE(m_c) 远小于 m_c，分母方差对区间的贡献
    #   在小数点后第 4 位，不改变任何判定；故保留原写法以免改变既有输出。
    ci_lo_rel, ci_hi_rel = ci_lo / m_c, ci_hi / m_c

    # ---------- 单侧劣效判定（只查下界）----------
    # 拒绝 H₀: Δ ≥ −δ 的等价条件：90% CI 上界 < −δ ⇒ 判"劣效 / 击穿"。
    # 只用一个方向，不引入上界（+δ）检查——护栏不关心"是否显著变好"。
    breached = bool(ci_hi_rel < -MARGIN)
    guardrail_pass = not breached
    verdict = "inferior" if breached else "pass"
    verdict_cn = "劣效（护栏击穿）" if breached else "通过（未劣于容差）"
    # 【口径升级】字段名与判定逻辑同步为"单侧劣效"（原 equivalent_pass / TOST 四态已废弃）；
    #   ⑧ 与 汇总报告.py 已同步读取 guardrail_pass / verdict_cn。
    guardrail_row = {
        "metric": "n_clicks_per_user", "margin": MARGIN,
        "control_mean": float(m_c), "treatment_mean": float(m_t),
        "relative_diff": float(rel_diff),
        "ci90_rel_low": float(ci_lo_rel), "ci90_rel_high": float(ci_hi_rel),
        "guardrail_pass": bool(guardrail_pass),
        "verdict": verdict, "verdict_cn": verdict_cn,
    }
    print(f"护栏 G = 人均点击次数（容差 −{MARGIN:.0%}，单侧劣效）；相对差 {rel_diff:+.2%} -> "
          f"{verdict_cn}{' ✅' if guardrail_pass else ' ⚠️'}")
    print(f"  90% CI（相对）= [{ci_lo_rel:+.2%}, {ci_hi_rel:+.2%}]；上界 {ci_hi_rel:+.2%} "
          f"vs −δ={-MARGIN:+.0%} → {'击穿' if breached else '通过'}")
    return guardrail_row, m_c, m_t


def _plot(peek_df, m_c, m_t, guardrail_pass):
    """画 ⑥ 的两张图：偷看 FPR 膨胀曲线 + 护栏指标对比。返回图片路径。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.plot(peek_df.look_k, peek_df.fpr_naive * 100, "o-", color="#C44E52", lw=2, ms=6,
            label="Naive (z>1.96 every look)")
    ax.plot(peek_df.look_k, peek_df.fpr_obf * 100, "s-", color="#55A868", lw=2, ms=6,
            label="O'Brien-Fleming boundary")
    ax.axhline(C.ALPHA * 100, color="gray", ls="--", lw=1.4, label="nominal 5%")
    ax.set_xlabel("Number of interim looks (days)")
    ax.set_ylabel("Cumulative false positive rate (%)")
    ax.set_title(f"Peeking inflates FPR (A/A, {N_SIMS} reps)")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.bar(["control", "treatment"], [m_c, m_t], color=["#4C72B0", "#55A868"])
    ax.set_ylabel("mean n_clicks / user")
    ax.set_title(f"Guardrail: n_clicks (one-sided, -{C.GUARDRAIL_MARGIN:.0%} "
                 f"{'pass' if guardrail_pass else 'triggered'})")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig_path = C.FIG_DIR / "06_序贯监控护栏图.png"
    fig.savefig(fig_path, dpi=130)
    print(f"\nSaved figure: {fig_path}")
    return fig_path


def run():
    """⑥ 主流程：6.1 偷看模拟 → 6.2 护栏检验 → 落盘。

    本函数不做具体计算，只负责编排与汇总。
    """
    C.hr("⑥ 序贯监控与安全护栏（OBF 偷看修正 / 单侧劣效护栏）")
    prepared, ctrl, trt = C.load_arms()
    A, B = C.CONTROL_APP, C.TREATMENT_APP
    n_c, x_c = len(ctrl), int(ctrl.converted.sum())
    p_c = x_c / n_c

    _day = getattr(prepared, "new_users_per_day_est", None)
    # full_csv 模式 meta 的窗口字段为 NaN；NaN 在 Python 里是【真值】，不能用
    # `or` 兜底（`_day or default` 会把合法的 0.0 也换掉，且 NaN 不会被拦截），
    # 必须显式判缺失。用 pd.isna 同时覆盖 None / NaN / NaT / pd.NA。
    # 回退到设计期预注册的规划常量——日流量是【规划假设】，不是观测量。
    if _day is None or pd.isna(_day):
        daily_traffic = float(C.ASSUMED_DAILY_TRAFFIC)
    else:
        daily_traffic = float(_day)

    peeking, peek_df = _peeking_simulation(p_c, daily_traffic)
    guardrail_row, m_c, m_t = _guardrail_test(ctrl, trt)

    summary = {
        "peeking": peeking,
        "guardrail": guardrail_row,
        "control_app": A, "treatment_app": B,
    }
    peek_df.to_csv(C.DETAILS_DIR / "06_序贯偷看假阳性率.csv", index=False)
    pd.DataFrame([guardrail_row]).to_csv(
        C.DETAILS_DIR / "06_护栏指标单侧劣效检验.csv", index=False)

    _plot(peek_df, m_c, m_t, guardrail_row["guardrail_pass"])

    C.dump_json(summary, C.DATA_DIR / "06_序贯监控与安全护栏.json")
    C.hr("✅ ⑥ 序贯监控与安全护栏完成")
    print(f"  产物: {C.DATA_DIR / '06_序贯监控与安全护栏.json'}")
    print(f"        {C.DETAILS_DIR / '06_序贯偷看假阳性率.csv'}")
    print(f"        {C.DETAILS_DIR / '06_护栏指标单侧劣效检验.csv'}")
    print(f"        {C.FIG_DIR / '06_序贯监控护栏图.png'}")
    return summary


if __name__ == "__main__":
    run()
