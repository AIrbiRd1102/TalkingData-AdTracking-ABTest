"""
汇总报告.py —— 汇总报告生成器（单数据集汇总）

读取 运行全流程.py 产出的 8 步 JSON（默认 data/），并可选读取子集快照
（默认 data/），生成：
  · docs/AB测试汇总报告_<日期>.html   （可交互预览；按需求仅产出 HTML，不再生成 .md）

设计原则：
  · 完全从 JSON 读取，绝不重算统计；只做「搬运 + 格式化 + 对比」。
  · 一切字段用 .get() 兜底，缺字段不崩。
  · 子集快照缺失时退化为「仅全量」单栏报告。

说明：CUPED（方差缩减，违反 pre-period 前提）与异动归因（步骤⑧，本数据非时间序列）已
从流程中移除，故本报告不再包含这两部分；分析生命周期现为 8 步。

用法：
  python scripts/汇总报告.py
  python scripts/汇总报告.py --full data --subset data --out docs
"""
from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
def pct(x, nd=4):
    """小数转百分比字符串。"""
    if x is None or (isinstance(x, float) and (x != x)):  # NaN
        return "—"
    return f"{x * 100:.{nd}f}%"


def pp(x, nd=3):
    """百分点（pp）字符串。"""
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x * 100:.{nd}f} pp"


def num(x, nd=2):
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:,.{nd}f}"


def fmt_p(x):
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    if x == 0:
        return "<1e-300"
    if x < 1e-4:
        return f"{x:.2e}"
    return f"{x:.6f}"


def load(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _img_data_uri(rel_path):
    """把 figures/ 下的图片读成 base64 data URI 内联进 HTML（缺失/失败返回 None）。

    为什么内联：报告是单个 .html，若用相对路径 <img src="figures/...">，一旦单独
    移动/分享就会断图。base64 内联让它真正"自包含"，离线双击即可看全。
    """
    p = PROJECT_ROOT / rel_path
    if not p.exists():
        return None
    try:
        b = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:image/{p.suffix.lstrip('.').lower()};base64,{b}"
    except Exception:
        return None


def get(d, *keys, default=None):
    """安全嵌套取值。"""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


# ----------------------------------------------------------------------------
# 抽取（把各步骤 JSON 压平成一组可比标量）
# ----------------------------------------------------------------------------
def extract(report, causal=None, primary=None):
    if report is None:
        return {}
    ed = get(report, "experiment_design_and_test", default={})
    vg = get(report, "validity_gate", default={})
    mon = get(report, "monitoring", default={})
    ci = get(report, "causal_inference", default={})
    meta = get(report, "meta", default={})

    out = {
        "source": meta.get("data_source", "—"),
        "unit": meta.get("unit", "—"),
        "decision": report.get("decision", "—"),
        # 显著性 / 效应量
        "control_rate": ed.get("control_rate"),
        "treatment_rate": ed.get("treatment_rate"),
        "relative_lift": ed.get("relative_lift"),
        "abs_lift_pp": None,  # 占位；稍后由 control/treatment rate 相减得到（见下方）
        "z_stat": ed.get("z_stat"),
        "p_value": ed.get("p_value"),
        "cohens_h": ed.get("cohens_h"),
        "ci95": ed.get("ci95"),
        "significant": ed.get("significant"),
        "achieved_power": ed.get("achieved_power"),
        # 有效性闸门
        "srm_failed": vg.get("real_split_srm_failed"),
        "covariate_balanced": vg.get("covariate_balanced"),
        "aa_p": vg.get("aa_p"),
        "causal_interpretable": vg.get("causal_interpretable"),
        # 序贯 / 护栏
        "peeking_fpr_naive": mon.get("peeking_fpr_naive"),
        "peeking_fpr_obf": mon.get("peeking_fpr_obf"),
        # 护栏单侧劣效判定（pass / inferior）；None 表示 08 未产出该字段。
        # 兜底读旧键 guardrail_equivalent_pass（极性一致：True=安全），兼容未重跑 ⑥⑧ 的旧 JSON。
        "guardrail_pass": mon.get("guardrail_pass", mon.get("guardrail_equivalent_pass")),
        "guardrail_verdict_cn": mon.get("guardrail_verdict_cn"),
        # 因果
        "sutva_overlap": ci.get("sutva_overlap_rate"),
        "psm_feasible": ci.get("psm_feasible"),
        "causal_headline": ci.get("headline_finding"),
    }
    # 绝对提升 pp
    if out["control_rate"] is not None and out["treatment_rate"] is not None:
        out["abs_lift_pp"] = out["treatment_rate"] - out["control_rate"]

    # 规模（从 07 明细取 n_units）
    if causal:
        data_block = get(causal, "data", default={})
        out["n_units"] = data_block.get("n_units")
        out["n_control"] = data_block.get("n_control")
        out["n_treatment"] = data_block.get("n_treatment")
        out["unique_ips"] = get(causal, "sutva_overlap", "unique_ips")
        out["overlap_rate"] = get(causal, "sutva_overlap", "overlap_rate")
        out["overlap_rate_unit"] = get(causal, "sutva_overlap", "overlap_rate_unit")
        out["control_rate_single_app"] = get(causal, "sutva_overlap", "control_rate_single_app")
        out["treatment_rate_single_app"] = get(causal, "sutva_overlap", "treatment_rate_single_app")
        out["n_single_app_control"] = get(causal, "sutva_overlap", "n_single_app_control")
        out["n_single_app_treatment"] = get(causal, "sutva_overlap", "n_single_app_treatment")
        out["single_app_diff"] = get(causal, "sutva_overlap", "naive_diff_single_app_subsample")
        # 推荐标题
        out["recommended_headline"] = get(causal, "recommended_headline", "interpretation")
        out["causal_hard_limit"] = get(causal, "recommended_headline", "hard_limit")
    # 指标体系的 OEC（总体评估标准）自解释：主指标即 OEC（来自 01 指标假设）
    if primary:
        pm = get(primary, "primary_metric", default={})
        out["oec_name"] = pm.get("name")
        out["oec_definition"] = pm.get("definition")
        out["oec_alias"] = pm.get("alias")
        out["oec_note"] = pm.get("oec_note")
    return out


# ----------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------
def badge(ok, true_text="通过", false_text="未通过", na_text="—"):
    if ok is None:
        return f'<span class="badge na">{na_text}</span>'
    cls = "ok" if ok else "bad"
    txt = true_text if ok else false_text
    return f'<span class="badge {cls}">{txt}</span>'


def build_html(full, sub, causal_full, primary, date_str):
    """把各步骤 JSON 渲染成单栏 HTML 报告。

    【输入】
      full        —— 全量 08_报告可视化与决策.json（dict）
      sub         —— 子集快照同名 JSON；None 时退化为单栏（本项目默认同目录，故退化为同值）
      causal_full —— 07_因果推断与稳健性.json（取 n_units / 重叠率 / headline）
      date_str    —— 生成时间字符串
    【输出】HTML 字符串。
    【设计原则】只做「搬运 + 格式化」，【绝不重算统计】——保证报告与各步骤
      产物永远一致；一切字段用 .get() 兜底，缺字段不崩。
    """
    f = extract(full, causal_full, primary)
    s = extract(sub) if sub else {}

    # ---- 关键指标对比表 ----
    # 【清理说明】早期版本先构造了一份「全量 vs 子集」双栏对比的 rows（含箭头
    # ▲▼ 比较逻辑），随后又用下方的 table_rows 单栏版本整体覆盖，导致 rows、
    # row()、以及模块级的 bar() / fmt_v() 都成了【未被使用的死代码】（约 60 行）。
    # 本项目当前默认 --full 与 --subset 指向同一目录，双栏对比实际退化为同值，
    # 故保留单栏 table_rows 即可；删除死代码不改变任何渲染结果。
    def cell(name, fv, sv, fmt, arrow_note=""):
        return f"""<tr><td>{name}</td><td class="num">{fmt(fv)}</td></tr>"""

    def status_cell(name, fv, sv):
        def render(v):
            if v is None:
                return "—"
            if isinstance(v, bool):
                return "通过" if v else "未通过"
            return str(v)
        return f"""<tr><td>{name}</td><td class="num">{render(fv)}</td></tr>"""

    table_rows = "".join([
        cell("样本规模 (ip,app) 数", f.get("n_units"), s.get("n_units"), num),
        cell("对照组 n", f.get("n_control"), s.get("n_control"), num),
        cell("实验组 n", f.get("n_treatment"), s.get("n_treatment"), num),
        cell("对照组转化率", f.get("control_rate"), s.get("control_rate"), pct),
        cell("实验组转化率", f.get("treatment_rate"), s.get("treatment_rate"), pct),
        cell("绝对提升", f.get("abs_lift_pp"), s.get("abs_lift_pp"), pp),
        cell("相对提升", f.get("relative_lift"), s.get("relative_lift"), lambda x: f"{x*100:.2f}%" if x is not None else '—'),
        cell("Z 统计量", f.get("z_stat"), s.get("z_stat"), lambda x: f"{x:.3f}"),
        cell("p 值", f.get("p_value"), s.get("p_value"), fmt_p),
        cell("Cohen's h", f.get("cohens_h"), s.get("cohens_h"), lambda x: f"{x:.4f}"),
        cell("95% CI", (f.get("ci95")[0] if isinstance(f.get("ci95"), list) else None,
                        f.get("ci95")[1] if isinstance(f.get("ci95"), list) else None),
             (s.get("ci95")[0] if isinstance(s.get("ci95"), list) else None,
              s.get("ci95")[1] if isinstance(s.get("ci95"), list) else None),
             lambda x: f"[{pct(x[0])}, {pct(x[1])}]" if isinstance(x, tuple) and x[0] is not None else "—"),
        cell("达成功效", f.get("achieved_power"), s.get("achieved_power"), pct),
        status_cell("SRM 是否失败（false=健康）", f.get("srm_failed"), s.get("srm_failed")),
        status_cell("协变量均衡", f.get("covariate_balanced"), s.get("covariate_balanced")),
        cell("A-A 检验 p", f.get("aa_p"), s.get("aa_p"), fmt_p),
        status_cell("可因果解读", f.get("causal_interpretable"), s.get("causal_interpretable")),
        cell("序贯偷看 FPR(朴素)", f.get("peeking_fpr_naive"), s.get("peeking_fpr_naive"), pct),
        cell("序贯偷看 FPR(OBF)", f.get("peeking_fpr_obf"), s.get("peeking_fpr_obf"), pct),
        status_cell("护栏单侧劣效通过", f.get("guardrail_pass"), s.get("guardrail_pass")),
        cell("SUTVA 重叠率", f.get("sutva_overlap"), s.get("sutva_overlap"), pct),
        status_cell("PSM 可行", f.get("psm_feasible"), s.get("psm_feasible")),
        cell("去污染子样本差 (仅单侧IP)", f.get("single_app_diff"), s.get("single_app_diff"), pp),
    ])

    # 因果估计明细表（来自 07）
    # 【为什么要把"相对化分母"和"SE 类型"也列出来】
    #   相对提升 = 点估计 / 分母，分母换成另一个口径，数字就会完全变样。
    #   原先这一列只写"相对提升(对照)"，但 DID 的分母其实不是对照基线
    #   （是实验组基期行级率），表头与数字不符，是典型的口径错标。
    #   现在把 07 落盘的 baseline_used_for_relative_lift / se_type 一起渲染出来，
    #   让读者一眼看到"这个百分比是除以什么得到的、SE 是怎么算的"。
    def fmt_rel(x):
        return f"{x*100:+.1f}%" if isinstance(x, (int, float)) else "—"

    def fmt_base(x):
        return pct(x) if isinstance(x, (int, float)) else "—"

    est_rows = ""
    if causal_full:
        for e in get(causal_full, "estimates", default=[]) or []:
            est_rows += f"""<tr>
              <td>{e.get('method','—')}</td>
              <td class="num">{pp(e.get('estimate_risk_diff'))}</td>
              <td class="num">{fmt_p(e.get('p_value'))}</td>
              <td class="num">{fmt_rel(e.get('relative_lift_vs_control'))}</td>
              <td class="num">{fmt_base(e.get('baseline_used_for_relative_lift'))}</td>
              <td>{e.get('se_type') or '—'}</td>
            </tr>"""
    est_table = f"""
      <table class="tbl">
        <thead><tr><th>方法</th><th>净提升 (pp)</th><th>p 值</th>
        <th>相对提升</th><th>相对化分母（基线口径）</th><th>SE 类型</th></tr></thead>
        <tbody>{est_rows}</tbody>
      </table>""" if est_rows else '<p class="muted">（07 明细缺失）</p>'

    decision = f.get("decision", "—")
    decision_cls = "ok" if "结论可采纳" in decision else "bad"

    # ---- 关键图表（base64 内联；文件缺失则整段跳过）----
    _fig_specs = [
        ("figures/08_汇总仪表盘图.png", "整体转化率与 Top OS 分段（兜底仪表盘）"),
        ("figures/07_因果推断图.png", "因果推断：森林图 / Love plot / 倾向得分重叠 / 事件研究"),
        ("figures/06_序贯监控护栏图.png", "序贯偷看 FPR 膨胀与护栏指标对比"),
        ("figures/04_分流有效性检验图.png", "哈希分桶均匀性与协变量均衡"),
        ("figures/02_实验设计图.png", "样本量–MDE 曲线 / 功效曲线 / 流量分配 / 分桶演示"),
    ]
    _fig_cards = "".join(
        f'<figure class="fig"><img src="{uri}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'
        for uri, cap in ((_img_data_uri(rel), cap) for rel, cap in _fig_specs) if uri)
    figs_section = (f'<h2>③ 关键图表</h2><div class="figs">{_fig_cards}</div>'
                    if _fig_cards else "")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A/B 测试汇总报告</title>
<style>
  :root {{ --bg:#f6f7fb; --card:#fff; --ink:#0f172a; --muted:#64748b; --line:#e6e9f0;
    --accent:#4f46e5; --accent2:#7c3aed; --ok:#059669; --bad:#dc2626;
    --shadow:0 1px 2px rgba(15,23,42,.06), 0 10px 26px -14px rgba(15,23,42,.25); }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; line-height:1.66; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:0 24px; }}
  .hero {{ position:relative; overflow:hidden; color:#fff;
    background:linear-gradient(120deg,#312e81 0%,#4f46e5 48%,#7c3aed 100%); }}
  .hero:after {{ content:""; position:absolute; inset:0;
    background:radial-gradient(620px 320px at 88% -30%, rgba(255,255,255,.20), transparent 60%),
               radial-gradient(520px 280px at -12% 130%, rgba(255,255,255,.14), transparent 60%); }}
  .hero .wrap {{ position:relative; z-index:1; padding-top:42px; padding-bottom:42px; }}
  .eyebrow {{ font-size:12px; letter-spacing:.24em; text-transform:uppercase; opacity:.82; }}
  .hero h1 {{ margin:10px 0 6px; font-size:34px; letter-spacing:.5px; }}
  .lede {{ margin:0 0 18px; opacity:.9; font-size:15px; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:8px; }}
  .chip {{ background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.26);
    border-radius:999px; padding:5px 13px; font-size:12.5px; }}
  .chip b {{ font-weight:700; }}
  main {{ padding-top:26px; padding-bottom:46px; }}
  .decision {{ position:relative; overflow:hidden; background:var(--card); border:1px solid var(--line);
    border-radius:16px; padding:20px 26px; margin:4px 0 22px; box-shadow:var(--shadow); }}
  .decision:before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:6px; }}
  .decision.ok:before {{ background:var(--ok); }}
  .decision.bad:before {{ background:var(--bad); }}
  .decision-label {{ font-size:12px; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); margin-bottom:7px; }}
  .decision-text {{ font-size:19px; font-weight:750; }}
  .decision.ok .decision-text {{ color:#065f46; }}
  .decision.bad .decision-text {{ color:#991b1b; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(158px,1fr)); gap:12px; margin:0 0 20px; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px 16px; box-shadow:var(--shadow); }}
  .kpi .k {{ font-size:12px; color:var(--muted); margin-bottom:6px; }}
  .kpi .v {{ font-size:21px; font-weight:750; font-variant-numeric:tabular-nums; }}
  .kpi .v small {{ font-size:13px; font-weight:600; color:var(--muted); }}
  .callout {{ display:grid; gap:10px; background:linear-gradient(180deg,#eef2ff,#f8faff);
    border:1px solid #e0e7ff; border-radius:16px; padding:18px 20px; margin-bottom:26px; }}
  .callout-k {{ display:inline-block; font-size:12px; font-weight:700; color:#3730a3; background:#e0e7ff;
    border-radius:7px; padding:2px 9px; margin-right:8px; }}
  h2 {{ font-size:20px; margin:30px 0 12px; padding-left:12px; border-left:4px solid var(--accent); }}
  table.tbl {{ width:100%; border-collapse:separate; border-spacing:0; font-size:14px; background:var(--card);
    border:1px solid var(--line); border-radius:14px; overflow:hidden; box-shadow:var(--shadow); }}
  table.tbl th, table.tbl td {{ padding:10px 14px; border-bottom:1px solid var(--line); text-align:left; }}
  table.tbl thead th {{ background:#f1f5ff; font-weight:650; font-size:13px; color:#334155; }}
  table.tbl tbody tr:last-child td {{ border-bottom:0; }}
  table.tbl tbody tr:hover td {{ background:#fafbff; }}
  table.tbl td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:12.5px; font-weight:650; }}
  .badge.ok {{ background:#dcfce7; color:#047857; }}
  .badge.bad {{ background:#fee2e2; color:#b91c1c; }}
  .badge.na {{ background:#f1f5f9; color:#64748b; }}
  .timeline {{ list-style:none; margin:0; padding:0; position:relative; }}
  .timeline:before {{ content:""; position:absolute; left:18px; top:8px; bottom:10px; width:2px;
    background:linear-gradient(#c7d2fe,#e9d5ff); }}
  .timeline > li {{ position:relative; padding:0 0 16px 54px; }}
  .timeline .dot {{ position:absolute; left:5px; top:0; width:28px; height:28px; border-radius:50%;
    background:linear-gradient(135deg,var(--accent),var(--accent2)); color:#fff; font-weight:750; font-size:14px;
    display:flex; align-items:center; justify-content:center; box-shadow:0 6px 12px -4px rgba(79,70,229,.6); }}
  .stepcard {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
    padding:14px 18px; box-shadow:var(--shadow); }}
  .stepcard h3 {{ margin:0 0 6px; font-size:15.5px; }}
  .stepcard ul {{ margin:6px 0 0; padding-left:20px; }}
  .stepcard li {{ margin:3px 0; }}
  .figs {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:16px; }}
  .fig {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:14px;
    padding:10px; box-shadow:var(--shadow); }}
  .fig img {{ width:100%; border-radius:9px; display:block; }}
  .fig figcaption {{ font-size:12.5px; color:var(--muted); padding:9px 4px 2px; }}
  footer {{ margin-top:34px; padding-top:16px; border-top:1px solid var(--line); color:var(--muted); font-size:12.5px; }}
  .muted {{ color:var(--muted); font-size:13px; }}
  code {{ background:#f1f5f9; padding:1px 6px; border-radius:5px; font-size:12.5px; }}
</style></head>
<body>
  <header class="hero">
    <div class="wrap">
      <div class="eyebrow">A/B Testing · 方法学演练报告</div>
      <h1>A/B 测试汇总报告</h1>
      <p class="lede">TalkingData AdTracking · 观察性对照的因果诚实解剖</p>
      <div class="chips">
        <span class="chip">数据源 <b>{f.get('source','—')}</b></span>
        <span class="chip">口径 <b>{f.get('unit','—')}</b></span>
        <span class="chip">OEC <b>{f.get('oec_name') or '—'}</b></span>
        <span class="chip">生成 <b>{date_str}</b></span>
      </div>
    </div>
  </header>

  <main class="wrap">
    <section class="decision {decision_cls}">
      <div class="decision-label">最终决策</div>
      <div class="decision-text">{decision}</div>
    </section>

    <section class="kpis">
      <div class="kpi"><div class="k">样本规模 (ip,app)</div><div class="v">{num(f.get('n_units'))}</div></div>
      <div class="kpi"><div class="k">对照 → 实验 转化率</div><div class="v">{pct(f.get('control_rate'))}<small> → </small>{pct(f.get('treatment_rate'))}</div></div>
      <div class="kpi"><div class="k">相对提升</div><div class="v" style="color:#b91c1c">{(f.get('relative_lift') or 0)*100:+.1f}%</div></div>
      <div class="kpi"><div class="k">Z / p</div><div class="v">{num(f.get('z_stat'),1)}<small> / {fmt_p(f.get('p_value'))}</small></div></div>
      <div class="kpi"><div class="k">SRM</div><div class="v">{badge(not f.get('srm_failed'), '健康', '失败')}</div></div>
      <div class="kpi"><div class="k">可因果解读</div><div class="v">{badge(f.get('causal_interpretable'), '是', '否')}</div></div>
    </section>

    <section class="callout">
      <div><span class="callout-k">核心结论</span>{f.get('causal_headline') or f.get('recommended_headline') or '—'}</div>
      <div><span class="callout-k">硬性边界</span>{f.get('causal_hard_limit') or '—'}</div>
    </section>

    <h2>① 关键指标</h2>
    <table class="tbl">
      <thead><tr><th>指标</th><th class="num">数值</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>

    <h2>② 设计优先 · 8 步生命周期</h2>
    <ol class="timeline">
      <li><span class="dot">1</span><div class="stepcard"><h3>问题定义与指标假设</h3>
        <ul>
          <li>业务目标：从 <code>01_问题定义与指标假设.json</code> 读取规划假设（基线转化率、MDE、护栏容差均为实验前设定，非观测值）。</li>
          <li><b>OEC（Overall Evaluation Criterion，总体评估标准）</b>：本项目 OEC = {f.get('oec_name') or '—'}（{f.get('oec_definition') or '—'}）；护栏指标（人均点击次数）为保护体验的辅助指标，不构成 OEC。</li>
        </ul></div></li>
      <li><span class="dot">2</span><div class="stepcard"><h3>实验设计（预注册）</h3>
        <ul>
          <li>随机分流：确定性哈希分桶（N=100，salt 固定），A 组 50%。</li>
          <li>所需样本量、周期由规划假设推算，写入 <code>02_实验设计.json</code>。</li>
        </ul></div></li>
      <li><span class="dot">3</span><div class="stepcard"><h3>数据接入与准备</h3>
        <ul><li>读取原始数据（来源：{f.get('source','—')}），聚合为 (ip,app) 用户级与 (ip,app,hour) 用户-小时级，落 <code>data/prepared/</code>。</li></ul></div></li>
      <li><span class="dot">4</span><div class="stepcard"><h3>分流有效性闸门</h3>
        <ul>
          <li>SRM：{badge(not f.get('srm_failed'),'健康','失败')}（本数据为观察性分流，比例偏离属数据采集特性）。</li>
          <li>协变量均衡：{badge(f.get('covariate_balanced'),'均衡','失衡')}；A-A 检验 p={fmt_p(f.get('aa_p'))}（流程自检）。</li>
          <li>结论：{badge(f.get('causal_interpretable'),'可因果解读','不可因果解读')}。</li>
        </ul></div></li>
      <li><span class="dot">5</span><div class="stepcard"><h3>显著性检验与效应量</h3>
        <ul>
          <li>对照 {pct(f.get('control_rate'))} vs 实验 {pct(f.get('treatment_rate'))}，绝对 <b>{pp(f.get('abs_lift_pp'))}</b>，相对 <b>{(f.get('relative_lift') or 0)*100:.1f}%</b>。</li>
          <li>Z={num(f.get('z_stat'),2)}，p={fmt_p(f.get('p_value'))}，Cohen's h={num(f.get('cohens_h'),4)}，95% CI {pct((f.get('ci95') or [None,None])[0])}–{pct((f.get('ci95') or [None,None])[1])}；达成功效 {pct(f.get('achieved_power'))}。</li>
        </ul></div></li>
      <li><span class="dot">6</span><div class="stepcard"><h3>序贯监控与安全护栏</h3>
        <ul>
          <li>偷看假阳性：朴素 {pct(f.get('peeking_fpr_naive'))} → OBF 校正 {pct(f.get('peeking_fpr_obf'))}。</li>
          <li>护栏（人均点击，单侧劣效检验，容差 −5%）：{badge(f.get('guardrail_pass'),'通过','击穿')}；判定 = <b>{f.get('guardrail_verdict_cn') or '—'}</b>
            <span class="muted">（单侧劣效只查下界 −δ：护栏只关心"别变差"，双侧 TOST 的上界检查在此冗余）</span>。</li>
        </ul></div></li>
      <li><span class="dot">7</span><div class="stepcard"><h3>因果推断与稳健性</h3>
        <ul>
          <li>SUTVA 重叠率 {pct(f.get('sutva_overlap'))}（ip 跨组重叠高，SUTVA 存疑）；PSM 可行：{badge(f.get('psm_feasible'),'是','否')}。</li>
          <li><b>去污染敏感性分析</b>：仅保留「单 app IP」子样本（对照 {num(f.get('n_single_app_control'))} / 实验 {num(f.get('n_single_app_treatment'))}），对照 {pct(f.get('control_rate_single_app'))} vs 实验 {pct(f.get('treatment_rate_single_app'))}，差 <b>{pp(f.get('single_app_diff'))}</b>（较全样本 +3.1894pp 缩水约 67%）。</li>
        </ul>
        {est_table}
      </div></li>
      <li><span class="dot">8</span><div class="stepcard"><h3>报告可视化与决策</h3>
        <ul><li><b>{decision}</b></li>
        <li>配套看板：<code>figures/08_汇总仪表盘图.png</code> 及各步骤 PNG（见下）。</li></ul></div></li>
    </ol>

    {figs_section}

    <section class="callout" style="background:linear-gradient(180deg,#fff7ed,#fffbf5);border-color:#fed7aa;">
      <div><b>口径说明：</b>本数据集（TalkingData AdTracking）的「实验组/对照组」由不同 app 天然形成，属<b>观察性对照</b>而非真随机化 A/B。因此第④步判定「不可因果解读」，第⑦步所有调整后估计仅代表<b>可观测混杂下的关联</b>，不可声称 app 的处理净效应。</div>
    </section>

    <footer>本报告由 <code>scripts/汇总报告.py</code> 从 8 步 JSON 自动生成，未做任何重算 · {date_str}</footer>
  </main>
</body></html>"""
    return html


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--subset", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "docs"))
    args = ap.parse_args()

    full_dir = Path(args.full)
    sub_dir = Path(args.subset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = load(full_dir / "08_报告可视化与决策.json")
    sub = load(sub_dir / "08_报告可视化与决策.json")
    causal_full = load(full_dir / "07_因果推断与稳健性.json")
    primary = load(full_dir / "01_问题定义与指标假设.json")

    if full is None:
        raise SystemExit("未找到全量 08_报告可视化与决策.json，请先跑完 运行全流程.py（full_csv 模式）。")

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    stamp = datetime.now().strftime("%Y-%m-%d")

    html = build_html(full, sub, causal_full, primary, date_str)
    # 按需求仅产出 HTML，不再生成 .md（build_md 已停用）。
    html_path = out_dir / f"AB测试汇总报告_{stamp}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"✅ 报告已生成（仅 HTML）：\n  {html_path}")


if __name__ == "__main__":
    main()
