"""
08_报告可视化与决策.py —— 大流程步骤 ⑧：汇总主报告 + 兜底仪表盘 + 决策建议

把 ②~⑦ 各步产出的 JSON 汇总成一份主报告（08_报告可视化与决策.json），打印关键结论，
生成整体 + 分段转化率的兜底仪表盘（figures/08_汇总仪表盘图.png）；并依据 ① 的决策规则
与 ④ 的有效性闸门，给出实验结论建议（可采纳 / 建议复核 / 不采纳）。

【本文件怎么读】run() 只做编排：读 JSON → 调 _decision() 出结论 → 组装 master →
落盘 → 画兜底仪表盘 → 调 _print_summary() 打控制台摘要。
_decision()（决策树）与 _print_summary()（控制台摘要）各自独立，可单独读。

数据来源：02_design / 04_validity / 05_significance / 06_monitoring / 07_causal_inference
          的 JSON + 03 产出的 用户级面板.parquet（仅画图）。

注：原第 ⑧ 步「异动归因」已移除——其分析粒度为"一天内小时桶"而非时序，
在本数据的 6 小时窗口下无法支撑"异动"结论，详见 README「方法取舍」。
产物：data/08_报告可视化与决策.json + figures/08_汇总仪表盘图.png

运行：python 08_报告可视化与决策.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import 公共工具 as C


def _build_dashboard(prepared):
    """整体转化率 + Top OS 分段转化率（从用户级面板即时计算）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    users = prepared.users
    A, B = C.CONTROL_APP, C.TREATMENT_APP
    ctrl = users[users.app == A]
    trt = users[users.app == B]
    rc = ctrl.converted.mean(); rt = trt.converted.mean()

    seg = users.groupby(["app", "os"]).agg(n=("converted", "size"),
                                           c=("converted", "sum")).reset_index()
    seg["rate"] = seg["c"] / seg["n"]
    top_os = (seg.groupby("os")["n"].sum().sort_values(ascending=False).head(10).index.tolist())
    piv = seg[seg.os.isin(top_os)].pivot_table(index="os", columns="app", values="rate")
    piv = piv.reindex(top_os)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle("A/B Test Dashboard - TalkingData AdTracking (full app12/18, 4-day)",
                 fontsize=14, fontweight="bold")
    bars = axes[0].bar(["Control (app12)", "Treatment (app18)"], [rc * 100, rt * 100],
                      color=["#4C72B0", "#DD8452"], width=0.55)
    axes[0].set_ylabel("Conversion Rate (%)"); axes[0].set_title("Overall Conversion Rate")
    for b, r, u, c in zip(bars, [rc * 100, rt * 100],
                          [len(ctrl), len(trt)], [int(ctrl.converted.sum()), int(trt.converted.sum())]):
        axes[0].text(b.get_x() + b.get_width() / 2, r + 0.05,
                     f"{r:.3f}%\n({c}/{u})", ha="center", va="bottom", fontsize=10)
    axes[0].set_ylim(0, max(rc, rt) * 100 * 1.35)
    rel_lift = (rt - rc) / rc * 100
    axes[0].text(0.5, max(rc, rt) * 100 * 1.18, f"Relative lift +{rel_lift:.1f}%",
                 ha="center", fontsize=11, color="#C44E52", fontweight="bold")

    x = np.arange(len(piv)); w = 0.38
    axes[1].bar(x - w / 2, piv.get(A, pd.Series(0, index=piv.index)) * 100, w,
                label=f"Control app{A}", color="#4C72B0")
    axes[1].bar(x + w / 2, piv.get(B, pd.Series(0, index=piv.index)) * 100, w,
                label=f"Treatment app{B}", color="#DD8452")
    axes[1].set_xticks(x); axes[1].set_xticklabels(piv.index, rotation=0)
    axes[1].set_xlabel("OS (Top 10 by users)"); axes[1].set_ylabel("Conversion Rate (%)")
    # 【⚠️ 多重检验注记】分段图是【探索性】视图：这里并未对每个 OS 段单独做假设检验
    # （没有 p 值、也没有置信区间），因此【不存在】多重比较问题；但一旦有人拿这张图
    # 去宣称"某 OS 段提升显著"，就落进了多重检验陷阱——10 个段各自 α=0.05，
    # 在全是 null 的情况下至少有一假阳性的概率 ≈ 1-0.95^10 = 40.1%。
    # 正式结论必须：① 分段检验用 Bonferroni/Holm 校正，或 ② 只保留 ① 里预注册的
    # 少数几个分段，或 ③ 用交互项检验（异质性检验）代替逐段比较。
    # 这里把提示直接画在图上，避免读者误读。
    axes[1].set_title("Segmented Conversion Rate (Top 10 OS)\n"
                      "[exploratory only - no per-segment test, no multiplicity correction]",
                      fontsize=9.5)
    axes[1].legend()
    fig.text(0.5, 0.012,
             "Note: segmented view is exploratory. No per-segment hypothesis test or "
             "multiplicity correction is applied; do not read any single segment as significant.",
             ha="center", fontsize=8, color="#555555")
    plt.tight_layout(rect=[0, 0.035, 1, 0.95])
    fig_path = C.FIG_DIR / "08_汇总仪表盘图.png"
    fig.savefig(fig_path, dpi=130)
    print(f"Saved dashboard: {fig_path}")
    return fig_path


def _load(name):
    with open(C.DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def _decision(significant, guard_pass, split_valid):
    """依据 ① 的决策规则 + ④ 的有效性闸门，给出实验结论建议。

    【三种结果】结论可采纳 / 建议复核 / 不采纳。
    【为什么用"多原因拼接"】原实现是四选一的【单分支字符串】，一旦同时命中多个
    否决条件（如本项目：护栏击穿 + 分流非随机），只有第一个被写进结论，其余被吞掉，
    读者会误以为"只有这一个问题"。这里改为：先给主判定，再把【所有】命中的否决项
    逐条列全（_issues），两者用分号拼接。
    """
    _reasons = []
    _reasons.append("主指标显著" if significant else "主指标未达显著")
    _reasons.append("护栏通过" if guard_pass else "护栏被击穿（体验指标显著变差）")
    _issues = []
    if not significant:
        _issues.append("主指标未达显著")
    if not guard_pass:
        _issues.append("护栏被击穿（体验指标显著变差）")
    if not split_valid:
        _issues.append("分流非随机（SRM 失败/协变量失衡）")

    if significant and guard_pass:
        if split_valid:
            decision = "结论可采纳（研究结论成立）：" + "、".join(_reasons)
        else:
            decision = ("建议复核（重做随机分流实验）：主指标显著且护栏通过，但分流非随机"
                        "（SRM 失败/协变量失衡），结果只能作强相关性线索；"
                        "建议先做真正随机分流的 A/B 实验再下结论")
    elif significant and not guard_pass:
        decision = "不采纳为因果结论（实验结论限定为关联）：主指标显著但护栏被击穿（体验指标显著变差）"
    else:
        decision = "不采纳（实验结论未达显著）：方向为正也不直接下结论，延长周期或扩流量重测"

    # 命中多个否决项时，把【主判定里没提到】的原因补在后面（避免只暴露第一条）。
    # ⚠️ 用 `x not in decision` 去重：主判定文案里已经写过"护栏被击穿（…）"，
    # 直接拼接会让同一条原因在结论里出现两次，反而更像笔误。
    if len(_issues) > 1:
        _extra = [x for x in _issues if x not in decision]
        if _extra:
            decision += "；同时存在：" + "、".join(_extra)
    return decision


def _print_summary(m):
    """把 master 的关键结论打到控制台（含"或 0 兜底"，防止缺字段时格式说明符抛错）。"""
    ed = m.get("experiment_design_and_test", {})
    vg = m.get("validity_gate", {})
    mon = m.get("monitoring", {})
    ci = m.get("causal_inference", {})
    print("【实验设计与检验】")
    print(f"  · 对照 {ed.get('control_rate') or 0:.4%} vs 实验 {ed.get('treatment_rate') or 0:.4%}，"
          f"相对提升 {ed.get('relative_lift') or 0:+.1%}")
    print(f"  · Z = {ed.get('z_stat') or 0:.4f}, p = {ed.get('p_value') or 0:.3e}；"
          f"Cohen's h = {ed.get('cohens_h') or 0:.4f}；显著 = {ed.get('significant')}")
    print(f"  · 设计阶段目标 MDE = +{ed.get('target_mde_relative') or 0:.0%}，"
          f"所需 ≈ {ed.get('n_per_group_required') or 0:,}/组")
    print("【④ 有效性闸门】")
    print(f"  · 真实分流 SRM = {'失败（观察性对比，非随机）⚠️' if vg.get('real_split_srm_failed') else '通过 ✅'}")
    print(f"  · 协变量均衡 = {'通过 ✅' if vg.get('covariate_balanced') else '失衡 ⚠️'}")
    print(f"  · A/A 测试 p = {vg.get('aa_p') or 0:.4f}（流程自检，非数据闸门）")
    print("【⑥ 监控与护栏】")
    print(f"  · 序贯偷看假阳性：名义 {mon.get('peeking_fpr_naive') or 0:.1%} → "
          f"OBF 修正 {mon.get('peeking_fpr_obf') or 0:.1%}")
    print(f"  · 护栏（人均点击 单侧劣效） = {'通过 ✅' if mon.get('guardrail_pass') else '击穿 ⚠️'}"
          f"；判定 = {mon.get('guardrail_verdict_cn') or '—'}")
    print("【⑦ 因果推断】")
    print(f"  · SUTVA 重叠率 = {ci.get('sutva_overlap_rate') or 0:.1%}（硬伤）")
    print(f"  · PSM 可行 = {ci.get('psm_feasible')}；核心结论：{ci.get('headline_finding')}")
    print("【⑧ 决策】")
    print(f"  · {m.get('decision')}")


def run():
    """⑧ 主流程：读各步 JSON → 决策 → 组装 master → 落盘 → 画仪表盘 → 打摘要。"""
    C.hr("⑧ 报告可视化与决策（汇总主报告 + 兜底仪表盘 + 决策建议）")
    des = _load("02_实验设计.json")
    validity = _load("04_分流有效性与质量校验.json")
    sig = _load("05_显著性检验与效应量.json")
    monitoring = _load("06_序贯监控与安全护栏.json")
    causal_summary = _load("07_因果推断与稳健性.json")
    prepared = C.load_prepared()

    causal_head = causal_summary.get("recommended_headline", {})
    valid = validity.get("validity_gate", {})
    # ⚠️ 用 C.dig 而不是 d.get("a", {}).get("b")：本项目的 JSON 由 to_python 生成，
    #    NaN 会被统一写成 null，"字段存在但值为 null"是真实会出现的，
    #   朴素链式 .get 在那种情况下会抛 AttributeError。
    guard_pass = C.dig(monitoring, "guardrail", "guardrail_pass", default=False)
    significant = sig.get("significant", False)

    decision = _decision(significant, guard_pass, bool(valid.get("split_valid")))

    master = {
        "meta": {
            "data_source": prepared.source,
            "unit": "(ip, app) 用户级 / (ip, app, hour) 用户-小时级",
            "generated_by": "运行全流程.py（平铺大步骤管道，设计优先）",
        },
        "experiment_design_and_test": {
            "control_app": C.CONTROL_APP, "treatment_app": C.TREATMENT_APP,
            "control_rate": sig.get("control_rate"),
            "treatment_rate": sig.get("treatment_rate"),
            "relative_lift": sig.get("relative_lift"),
            "z_stat": sig.get("z_stat"), "p_value": sig.get("p_value"),
            "cohens_h": sig.get("cohens_h"),
            "ci95": [sig.get("ci95_low"), sig.get("ci95_high")],
            "significant": significant,
            "achieved_power": sig.get("achieved_power"),
            "target_mde_relative": des.get("sample_size", {}).get("target_mde_relative"),
            "n_per_group_required": des.get("sample_size", {}).get("n_per_group_required"),
        },
        "validity_gate": {
            "real_split_srm_failed": C.dig(validity, "real_split_srm", "srm_failed"),
            "covariate_balanced": C.dig(validity, "covariate_balance", "balanced"),
            "aa_p": C.dig(validity, "aa_test", "p"),
            "causal_interpretable": valid.get("causal_interpretable"),
        },
        "monitoring": {
            "peeking_fpr_naive": C.dig(monitoring, "peeking", "fpr_naive_final"),
            "peeking_fpr_obf": C.dig(monitoring, "peeking", "fpr_obf_final"),
            "guardrail_pass": guard_pass,
            # 【口径】护栏采用【单侧劣效检验】（只查是否劣于 −δ 容差），判定为
            # pass / inferior 两态：pass=未劣于容差；inferior=显著劣于容差（击穿）。
            "guardrail_verdict": C.dig(monitoring, "guardrail", "verdict"),
            "guardrail_verdict_cn": C.dig(monitoring, "guardrail", "verdict_cn"),
        },
        "causal_inference": {
            "sutva_overlap_rate": C.dig(causal_summary, "sutva_overlap", "overlap_rate"),
            "psm_feasible": causal_head.get("psm_feasible"),
            "headline_finding": causal_head.get("finding"),
            "hard_limit": causal_head.get("hard_limit"),
            "estimates": [
                {"method": e["method"], "net_lift_pp": e["estimate_risk_diff"] * 100,
                 "p_value": e["p_value"]}
                for e in causal_summary.get("estimates", [])],
        },
        "decision": decision,
    }
    C.dump_json(master, C.DATA_DIR / "08_报告可视化与决策.json")

    _build_dashboard(prepared)

    # 控制台摘要
    C.hr("✅ 主报告（08_报告可视化与决策.json）")
    print("【① 问题 → ② 设计 → ③ 接入 → ④ 校验 → ⑤ 检验 → ⑥ 监控 → ⑦ 因果】")
    # 说明：以下打印统一用 `or 0` 兜底。若某个 JSON 缺字段（例如只重跑了部分
    # 步骤），None 直接进 f-string 的格式说明符（:.4% / :,）会抛 TypeError，
    # 导致整段摘要打不出来。与 汇总报告.py「一切 .get() 兜底」的原则保持一致。
    _print_summary(master)
    print(f"\n产物: {C.DATA_DIR / '08_报告可视化与决策.json'} | {C.FIG_DIR / '08_汇总仪表盘图.png'}")
    return master


if __name__ == "__main__":
    run()
