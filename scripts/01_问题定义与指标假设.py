"""
01_问题定义与指标假设.py —— 大流程步骤 ①：在动手看任何结果之前，先把"测什么、怎么算赢"定清楚

这是 A/B 测试最容易被跳过、却最关键的一步。正确顺序必须是：
  ① 问题/假设（本步）→ ② 实验设计（下一步，用本步的规划假设算样本量）→ ③ 才接入数据。

本步【不读取任何实验数据】，只做"实验前该定的事"（即预注册 pre-registration）：
  · 业务目标与北极星
  · 主指标 / 护栏指标 / 次要指标
  · 原假设 H0 与备择假设 H1（单侧 / 双侧）
  · 实验前就要锁定的规划假设：α、功效(1-β)、假设基线转化率、目标 MDE、日均流量
  · 决策规则（方法学演示：实验结论的可采纳 / 建议复核 / 不采纳，非真实业务上线决策）

【本文件怎么读】run() 只做编排：1-A~1-D 四段各是一个函数（各自构建并打印自己那部分），
run() 负责组装成 spec 并落盘。每段都可以单独读。

产物：data/01_问题定义与指标假设.json （被 ② 实验设计 直接消费）

运行：python 01_问题定义与指标假设.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import 公共工具 as C


def _business_and_metrics():
    """1-A 业务目标与指标体系：主指标（OEC）/ 护栏指标 / 次要指标。"""
    C.hr("1-A) 业务目标与指标")
    A, B = C.CONTROL_APP, C.TREATMENT_APP
    business_goal = (
        f"评估把广告点击后激活/安装（is_attributed=1）的转化率，从对照组 app{A} "
        f"切到实验组 app{B}，是否能带来统计显著且业务可接受的提升。"
    )
    primary_metric = {
        "name": "conversion_rate",
        # 口径修正：早期文案误写为 sum(is_attributed)（会算成"归因点击次数"），
        # 与 ② 的预注册条款、③ 的实际实现（MAX）不一致。这里统一为 MAX，
        # 语义是"该用户是否【曾】转化过"这一 0/1 结果。
        "definition": ("用户级 (ip,app) 去重后，"
                       "converted = MAX(is_attributed)（是否曾转化） / n_users"),
        "direction": "higher_is_better",
        "rationale": "直接反映广告投放的变现效率，是本次实验的成败判定依据",
        "alias": "OEC",
        "oec_note": "本项目的 OEC（Overall Evaluation Criterion，总体评估标准）即此主指标：唯一决定实验成败的判定指标；护栏指标为保护体验的辅助指标，不构成 OEC。",
    }
    guardrail_metrics = [{
        "name": "n_clicks_per_user",
        "definition": "人均点击次数（用户级）",
        "direction": "non_inferior",
        "margin": C.GUARDRAIL_MARGIN,
        "rationale": "防止新 app 提升转化的同时损害了用户互动/体验（护栏：不能显著变差）",
    }]
    secondary_metrics = [{
        "name": "activation_latency",
        "definition": "点击到激活的时延（若有埋点）",
        "direction": "lower_is_better",
        "rationale": "辅助观察体验侧影响，不作为主判定",
    }]
    print(f"业务目标: {business_goal}")
    print(f"主指标: {primary_metric['name']}（{primary_metric['direction']}）")
    print(f"护栏指标: {guardrail_metrics[0]['name']}（非劣效容差 −{C.GUARDRAIL_MARGIN:.0%}）")
    return business_goal, primary_metric, guardrail_metrics, secondary_metrics


def _hypothesis():
    """1-B 原假设与备择假设（含单侧/双侧的选择理由）。"""
    C.hr("1-B) 原假设与备择假设")
    hypothesis = {
        "H0": "p_treatment - p_control = 0  （app 切换不改变转化率）",
        "H1": "p_treatment - p_control != 0  （双侧：可能变好也可能变差）",
        "alternative": "two-sided",
        "note": "业务真正关心的是'是否变好'，但用双侧可同时捕捉'变差'（安全/护栏）；"
                "若只看提升也可改 one-sided(>0)，此处选双侧更稳健。",
    }
    print(hypothesis["H0"])
    print(hypothesis["H1"])
    return hypothesis


def _planning_assumptions():
    """1-C 实验前锁定的规划假设（预注册，严禁看数据后修改）。

    ⚠️ 以下全部是【规划/假设值】，绝非本实验的观测值；② 只用它们算样本量。
    """
    C.hr("1-C) 实验前锁定的规划假设（预注册，严禁看数据后修改）")
    planning = {
        "alpha": C.ALPHA,
        "power": C.POWER,
        "assumed_baseline_rate": C.ASSUMED_BASELINE_RATE,
        "target_mde_relative": C.TARGET_MDE_REL,
        "assumed_daily_traffic": C.ASSUMED_DAILY_TRAFFIC,
        "guardrail_margin": C.GUARDRAIL_MARGIN,
        "warning": "以上 baseline / traffic 均为规划假设值（领域经验或历史试点），"
                   "非本实验观测值；步骤②只用这些值算样本量，绝不偷看真实转化。",
    }
    print(f"α = {C.ALPHA}（一类错误上限）")
    print(f"目标功效 1-β = {C.POWER}")
    print(f"假设基线 p0 = {C.ASSUMED_BASELINE_RATE:.4%}（规划假设，非观测）")
    print(f"目标 MDE = +{C.TARGET_MDE_REL:.0%} 相对提升（业务可接受的最小增益）")
    print(f"假设日均流量 = {C.ASSUMED_DAILY_TRAFFIC:,} (ip,app)/天（估周期用）")
    return planning


def _decision_rule():
    """1-D 决策规则：满足什么条件才采纳 / 建议复核 / 不采纳。"""
    C.hr("1-D) 决策规则（ship / no-ship / iterate）")
    decision_rule = {
        "ship_if": "主指标 Z 检验 p<α 且相对提升 ≥ 目标 MDE 且 护栏指标单侧劣效通过（未劣于容差）",
        "no_ship_if": "主指标 p≥α，或提升 < 目标 MDE，或护栏被击穿（n_clicks 显著变差）",
        "iterate_if": "方向为正但不显著（功效不足）→ 延长周期/扩流量重测，而非直接下结论",
        # ⚠️ 数据有效性闸门只有 SRM + 协变量均衡；A/A 是【流程自检】（检验工具链
        #    是否正常工作），不构成数据闸门——详见 ④ 的 4.4 说明。
        "validity_gate": "步骤④ SRM/协变量均衡 任一不通过 → 分流无效，结论不可信，先修分流（A/A 为流程自检，不构成数据有效性闸门）",
    }
    for k, v in decision_rule.items():
        print(f"  - {k}: {v}")
    return decision_rule


def run():
    """① 主流程：依次构建 1-A~1-D 四段，组装成 spec 落盘。本步不读取任何实验数据。"""
    C.hr("① 问题定义与指标假设（看数据之前先定：测什么、怎么算赢）")

    business_goal, primary_metric, guardrail_metrics, secondary_metrics = _business_and_metrics()
    hypothesis = _hypothesis()
    planning = _planning_assumptions()
    decision_rule = _decision_rule()

    spec = {
        "business_goal": business_goal,
        "primary_metric": primary_metric,
        "guardrail_metrics": guardrail_metrics,
        "secondary_metrics": secondary_metrics,
        "hypothesis": hypothesis,
        "planning_assumptions": planning,
        "decision_rule": decision_rule,
        "control_app": C.CONTROL_APP, "treatment_app": C.TREATMENT_APP,
    }
    C.dump_json(spec, C.DATA_DIR / "01_问题定义与指标假设.json")
    C.hr("✅ ① 问题定义与指标假设完成（已写 data/01_问题定义与指标假设.json，供 ② 实验设计消费）")
    print(f"  产物: {C.DATA_DIR / '01_问题定义与指标假设.json'}")
    return spec


if __name__ == "__main__":
    run()
