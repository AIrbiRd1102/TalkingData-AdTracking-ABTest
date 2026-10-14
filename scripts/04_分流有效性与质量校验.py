"""
04_分流有效性与质量校验.py —— 大流程步骤 ④：实验有效性闸门（先看分流是否可靠）

本步在"接入数据（③）"之后、"显著性检验（⑤）"之前：
如果分流本身不可信（样本比例失衡 SRM、协变量不均衡、随机化没生效），
那么后面的任何显著/不显著结论都不可信。这就是数据科学里强调的
"validity check before inference"。

与 ② 的"分流方案设计"呼应：② 只定方案，本步在真实数据上验证方案质量，
并额外检查"对照组 vs 实验组"是否真的近似随机（本数据集是观察性对比，会暴露问题）。

覆盖：
  · 4.1 真实分流 SRM 检查（对照组/实验组用户数是否 ≈ 预期比例）
  · 4.2 哈希分桶方案质量（确定性 / 均匀性 / 正交性 / 分桶本身是否 ≈50/50）
  · 4.3 协变量均衡（os / device / channel 在两组间分布是否一致）
  · 4.4 A/A 测试（对照组随机二分，验证检验流程不会误判显著）

【本文件怎么读】run() 只做编排：4.1~4.4 各是一个函数（自己读输入、自己打印、
返回自己的结果），最后 run() 汇总判定与落盘。每一检都可以单独读、单独改。

数据来源：03 产出的 data/prepared/（用户级面板.parquet + IP众数属性.parquet）。
产物：data/04_分流有效性与质量校验.json + data/details/04_协变量均衡检验.csv
      + figures/04_分流有效性检验图.png

运行：python 04_分流有效性与质量校验.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest

import 公共工具 as C


def _max_share_gap(a, b):
    """分类协变量在两组间的"最大水平占比差"，衡量协变量均衡度（越小越均衡）。

    【输入】a, b —— 同一分类列在实验组 / 对照组的取值数组（如 os、device、channel）。
    【输出】标量 gap ∈ [0, 1]。
    【统计原理】
        gap = max_l | P(X = l | T=1) − P(X = l | T=0) |
      即逐个水平（channel 的每个取值、os 的每个取值…）算两组占比之差，
      取绝对值最大的那个。
      ⚠️ 为什么分类变量不用 SMD：SMD 的定义依赖"均值"和"方差"，
      对名义分类变量（channel=123 与 channel=456 之间没有大小关系）没有意义。
      所以分类变量改用"各水平占比差的最大值"这一更直观的尺度。
    【阈值】本项目取 C.CATEGORICAL_BALANCE_GAP = 0.01（占比差尺度）。⚠️ 与 ⑦ 里 PSM 用的
      "|SMD| < 0.1"（标准差尺度）**不是同一个指标**，报数时必须说明用哪个。
    """
    a = np.asarray(a)
    b = np.asarray(b)
    levels = np.unique(np.concatenate([a, b]))   # 两组出现过的全部水平
    pa = np.array([(a == l).mean() for l in levels])   # 实验组各水平占比
    pb = np.array([(b == l).mean() for l in levels])   # 对照组各水平占比
    return float(np.max(np.abs(pa - pb)))


def _real_split_srm(ctrl, trt):
    """4.1 真实分流 SRM 检查：对照/实验两组用户数是否 ≈ 预期比例（50/50）。

    【返回】SRM 结果 dict（含 srm_failed 布尔，供有效性闸门使用）。
    """
    C.hr("4.1 真实分流 SRM 检查（对照 vs 实验 用户数是否 ≈ 50/50）")
    # 卡方拟合优度检验：χ² = Σ (O_k − E_k)² / E_k，df = 组数 − 1 = 1
    # 分子惩罚偏离幅度（平方），除以 E 做"相对化"——同样偏 1000 人，
    # 在期望 100 人的组里是灾难，在期望 10 万人的组里无足轻重。
    n_c, n_t = len(ctrl), len(trt)
    obs = np.array([n_c, n_t])
    chi2_app, p_srm_app = chisquare(obs, [obs.sum() * 0.5, obs.sum() * 0.5])
    print(f"观测用户数: 对照={n_c:,} ({n_c/(n_c+n_t):.2%}), 实验={n_t:,} ({n_t/(n_c+n_t):.2%})")
    print(f"卡方 = {chi2_app:.2f}, p = {p_srm_app:.3e}")
    # 【阈值口径】SRM 业界常用比 0.05 更严的判据（如 p<0.001，或直接看相对偏离幅度），
    #   因为超大规模下极小的比例漂移也会"显著"。本项目两组 53.3% vs 46.7%，偏离幅度
    #   极大（χ² 达数千量级），用 0.05 还是 0.001 都判失败，故阈值选择不影响结论。
    srm_failed = p_srm_app < C.ALPHA
    print("结论: " + ("SRM 失败 —— 两组非 50/50，分流不是随机的（观察性对比的典型特征）"
                       if srm_failed else "SRM 通过"))
    return {"control_n": n_c, "treatment_n": n_t,
            "control_share": n_c / (n_c + n_t), "treatment_share": n_t / (n_c + n_t),
            "chi2": float(chi2_app), "p": float(p_srm_app), "srm_failed": bool(srm_failed)}


def _hash_scheme_quality(ip_attr):
    """4.2 哈希分桶方案质量：② 设计的方案，在真实 ip 上验证。

    检查四件事：① 确定性（同 ip 永远同桶）② 均匀性（100 个桶是否均等）
    ③ 分桶本身是否 ≈50/50 ④ 多实验正交性（换 salt 后是否近似独立）。
    【返回】(scheme dict, counts, exp_n, p_u)——后两项仅供画图使用。
    """
    C.hr("4.2 哈希分桶方案质量（② 设计的方案，在真实 ip 上验证）")
    ips = ip_attr["ip"].to_numpy()
    b_main = np.array([C.bucket_of(ip) for ip in ips])
    b_main_again = np.array([C.bucket_of(ip) for ip in ips])
    determinism_ok = bool(np.array_equal(b_main, b_main_again))
    counts = np.bincount(b_main, minlength=C.N_BUCKETS)
    exp_n = len(ips) / C.N_BUCKETS
    chi2_u, p_u = chisquare(counts)
    grp = np.where(b_main < C.N_BUCKETS * C.SPLIT_A_FRACTION, "A", "B")
    nA, nB = int((grp == "A").sum()), int((grp == "B").sum())
    chi2_srm_scheme, p_srm_scheme = chisquare([nA, nB], [len(ips) / 2, len(ips) / 2])
    b_side = np.array([C.bucket_of(ip, "exp_reco_side_v1") for ip in ips])
    ct = pd.crosstab(pd.Series(b_main, name="main"), pd.Series(b_side, name="side"))
    chi2_o, p_o, _, _ = chi2_contingency(ct.values)
    print(f"(1) 确定性 {'一致 ✅' if determinism_ok else '不一致 ❌'}")
    print(f"(2) 均匀性 p={p_u:.4f} {'均匀 ✅' if p_u > C.ALPHA else '不均匀 ❌'}")
    print(f"(3) 分桶本身 SRM p={p_srm_scheme:.4f} "
          f"{'通过 ✅（≈50/50）' if p_srm_scheme > C.ALPHA else '失败 ❌'}")
    print(f"(4) 正交性(双实验) p={p_o:.4f} {'✅' if p_o > C.ALPHA else '❌'}")
    # 四项全过才算"分桶机制健康"。⚠️ 其中均匀性检验的 n 是 17 万个 ip、df=99，
    # 大样本下卡方检验对极小偏差也极其敏感（本项目 uniformity_p≈0.023 < 0.05），
    # 因此 scheme_ok=False 只说明"均匀性边际不达标"，【不等于分桶方案坏掉】——
    # 必须结合实际偏离幅度与 scheme_srm_p / orthogonality_p 一起读。
    # 这与 4.1 SRM 用 α=0.05 是同一类"大样本阈值敏感性"问题。
    scheme_ok = (determinism_ok and (p_u > C.ALPHA)
                 and (p_srm_scheme > C.ALPHA) and (p_o > C.ALPHA))
    if scheme_ok:
        print("→ 分桶机制本身健康；下面 4.1/4.3 的失衡来自 app12/app18 不是随机分配。")
    else:
        print("→ 分桶机制有未达标项（见上表逐项标注）；下面 4.1/4.3 的失衡来自"
              "app12/app18 不是随机分配，与分桶方案本身无关。")

    scheme = {"n_ips": int(len(ips)), "n_buckets": C.N_BUCKETS,
              "determinism_ok": determinism_ok, "uniformity_p": float(p_u),
              "scheme_srm_p": float(p_srm_scheme), "orthogonality_p": float(p_o),
              "scheme_A": nA, "scheme_B": nB}
    return scheme, scheme_ok, counts, exp_n, p_u


def _covariate_balance(ctrl, trt):
    """4.3 协变量均衡：os / device / channel 在两组间分布是否一致。

    用户级面板在 ③ 数据接入时已把 os/device/channel 按 (ip,app) 关联进来，
    直接按真实分组切分即可（无需再 merge，否则同名列会冲突成 _x/_y）。
    【返回】(balance_df, balance_ok)。
    """
    C.hr("4.3 协变量均衡（os / device / channel 在两组间是否一致）")
    bal_rows = [{"covariate": c, "max_level_share_gap": _max_share_gap(ctrl[c], trt[c])}
                for c in ["os", "device", "channel"]]
    balance_df = pd.DataFrame(bal_rows)
    balance_ok = bool((balance_df.max_level_share_gap < C.CATEGORICAL_BALANCE_GAP).all())
    for _, r in balance_df.iterrows():
        print(f"  {r['covariate']}: 最大水平占比差 = {r['max_level_share_gap']:.4f} "
              f"{'✅' if r['max_level_share_gap'] < C.CATEGORICAL_BALANCE_GAP else '⚠️ 失衡'}")
    print("注：观察性数据下协变量常失衡（设备/渠道构成不同），会混淆效应估计。")
    return balance_df, balance_ok


def _aa_test(ctrl):
    """4.4 A/A 测试：把对照组随机切成两半互相检验，验证检验流程不会误判显著。

    【统计原理】A/A 是【同一组】随机二分，真实效应为 0。H₀ 成立时 p 值服从
    U(0,1)，所以单次 A/A 的 p 本来就有 5% 概率掉进 [0, 0.05]——【不能拿单次
    p < 0.05 就判"管线误判"】。正确用法是重复多次、检查 p 值分布是否均匀。
    本项目已在配套文档中补充做了 300 次重复：p 值均值 0.4746、<0.05 占 6.33%、
    KS 检验 p=0.0976（不能拒绝均匀）→ 管线无系统性偏差。
    这里按脚本口径只做一次，得到的 p=0.0444 属正常波动，不构成 A/A 失败。

    ⚠️ 定位说明：A/A 检验的是【检验流程/工具链】是否正常工作，属于流程自检；
    它【不构成】数据层面的有效性闸门（那一条是 SRM + 协变量均衡，见 run()）。
    """
    C.hr("4.4 A/A 测试（对照组随机二分，验证检验流程不误判显著）")
    rng = np.random.default_rng(C.SEED)
    idx = rng.permutation(ctrl.index)
    half = len(idx) // 2
    aa_a = ctrl.loc[idx[:half]]
    aa_b = ctrl.loc[idx[half:]]
    na, xa = len(aa_a), int(aa_a.converted.sum())
    nb, xb = len(aa_b), int(aa_b.converted.sum())
    z_aa, p_aa = proportions_ztest([xa, xb], [na, nb], alternative="two-sided")
    print(f"A 组 n={na:,} 率={xa/na:.6%}；B 组 n={nb:,} 率={xb/nb:.6%}")
    print(f"Z = {z_aa:.4f}, p = {p_aa:.4f} -> "
          f"{'未误判(正常) ✅' if p_aa >= C.ALPHA else '单次落 5% 区间（正常波动，需多次重复判断，见上方注释）⚠️'}")
    return {"p": float(p_aa), "ok": bool(p_aa >= C.ALPHA)}


def _plot(counts, exp_n, p_u, balance_df):
    """画 ④ 的两张图：哈希分桶均匀性 + 协变量均衡。返回图片路径。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.bar(np.arange(C.N_BUCKETS), counts, color="#4C72B0", width=0.9)
    ax.axhline(exp_n, color="#C44E52", ls="--", lw=1.5, label=f"expected = {exp_n:.0f}")
    ax.set_xlabel("Hash bucket (0-99)")
    ax.set_ylabel("Number of users")
    ax.set_title(f"Bucket uniformity  p={p_u:.3f}")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    ax = axes[1]
    bars = ax.bar(balance_df.covariate, balance_df.max_level_share_gap, color="#55A868")
    ax.axhline(C.CATEGORICAL_BALANCE_GAP, color="#C44E52", ls="--", lw=1.5,
               label=f"balance threshold {C.CATEGORICAL_BALANCE_GAP}")
    ax.set_ylabel("max level share gap")
    ax.set_title("Covariate balance (ctrl vs trt)")
    for b, v in zip(bars, balance_df.max_level_share_gap):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig_path = C.FIG_DIR / "04_分流有效性检验图.png"
    fig.savefig(fig_path, dpi=130)
    print(f"\nSaved figure: {fig_path}")
    return fig_path


def run():
    """④ 主流程：依次跑 4.1~4.4 四项检查，汇总有效性闸门判定并落盘。

    本函数不做具体检验，只负责编排与汇总。每一项检查的实现见对应函数。
    """
    C.hr("④ 分流有效性与质量校验（SRM / 分桶质量 / 协变量均衡 / A-A）")
    prepared, ctrl, trt = C.load_arms()
    ip_attr = prepared.ip_attr

    real_srm = _real_split_srm(ctrl, trt)
    scheme, scheme_ok, counts, exp_n, p_u = _hash_scheme_quality(ip_attr)
    balance_df, balance_ok = _covariate_balance(ctrl, trt)
    aa = _aa_test(ctrl)

    # ---------- 有效性总判定 ----------
    # ⚠️ 数据层面的闸门只有两条：真实分流 SRM + 协变量均衡。
    #    A/A 是【流程自检】（检验工具链是否正常），不参与本判定。
    #    scheme_ok 的"大样本阈值敏感性"说明见 _hash_scheme_quality 内的注释。
    split_valid = (not real_srm["srm_failed"]) and balance_ok
    print(f"\n有效性闸门：分桶机制={'健康 ✅' if scheme_ok else '异常 ❌'}；"
          f"真实分流={'有效 ✅' if split_valid else '无效 ⚠️（观察性，结论不可作因果）'}")
    if not split_valid:
        print("⚠️ 由于 SRM 失败 / 协变量失衡，⑤ 及后续的显著结果只能作为"
              "相关性线索，不能解释为 app 的处理效应（因果需 ⑦ 进一步讨论）。")

    summary = {
        "real_split_srm": real_srm,
        "hash_scheme_quality": scheme,
        "covariate_balance": {
            "max_gap": {r["covariate"]: float(r["max_level_share_gap"])
                        for _, r in balance_df.iterrows()},
            "balanced": balance_ok,
        },
        "aa_test": aa,
        "validity_gate": {
            "scheme_ok": scheme_ok, "split_valid": bool(split_valid),
            "causal_interpretable": bool(split_valid),
        },
        "control_app": C.CONTROL_APP, "treatment_app": C.TREATMENT_APP,
    }
    balance_df.to_csv(C.DETAILS_DIR / "04_协变量均衡检验.csv", index=False)
    _plot(counts, exp_n, p_u, balance_df)

    C.dump_json(summary, C.DATA_DIR / "04_分流有效性与质量校验.json")
    C.hr("✅ ④ 分流有效性与质量校验完成")
    print(f"  产物: {C.DATA_DIR / '04_分流有效性与质量校验.json'}")
    print(f"        {C.DETAILS_DIR / '04_协变量均衡检验.csv'}")
    print(f"        {C.FIG_DIR / '04_分流有效性检验图.png'}")
    return summary


if __name__ == "__main__":
    run()
