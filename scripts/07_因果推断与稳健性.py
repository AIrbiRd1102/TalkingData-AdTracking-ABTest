"""
07_因果推断与稳健性.py —— 大流程步骤 ⑦：当随机化不可信时，如何更稳健地估计处理效应

前置：④ 有效性闸门已显示本数据为观察性对比（SRM 失败、协变量失衡），
所以 ⑤ 的显著结果只能当相关性线索。本步演示"若想逼近因果，该怎么剥离可观测混杂"：

  · 7.0 SUTVA / 重叠诊断（同一 ip 同时进入两组的污染程度）
  · 7.1 朴素差异（基准线）
  · 7.2 倾向得分模型（逻辑回归）+ 5-fold CV AUC / Brier —— 量化"是否可识别"
  · 7.3 PSM 可行性诊断（1:1 最近邻 + caliper）—— 仅诊断，不报告估计值
  · 7.4 IPW 逆概率加权（ATT + 尾部截断 + bootstrap SE）
  · 7.5 DID 双重差分（⚠️ 方法演示，无真实政策冲击 + 安慰剂检验）
  · 7.6 E-value 敏感性分析

方法取舍说明：
  · DML 已移除——本数据重叠崩溃（PS 的 CV-AUC 极高、共同支撑内样本极少），
    已判定不具备因果识别条件，再估计 ATE 无意义；且 DML 的价值在于高维/非线性
    协变量下的 nuisance 建模，本项目协变量仅 os/device/channel/hour 四个分类变量，
    逻辑回归足够。
  · PSM 只做"可行性诊断"不报估计值——匹配率过低且匹配后协变量仍未达平衡
    （|SMD| > 0.1），此时任何匹配估计都不具代表性，报告数字反而误导。

【本文件怎么读】run() 只做"编排"：按 7.0→7.6 的顺序调用下面各自的函数。
每个函数负责一个分析块——读自己的输入、算自己的东西、自己打印、把结果【返回】；
块与块之间只通过参数/返回值传递数据，【不使用模块级全局变量】。
这样每一块都能单独读、单独改、单独验证，不会被别的块的状态污染
（改造前 run() 是约 500 行的单函数，且靠 global _keep_levels 在函数间隐式传参）。

分析单元：(ip, app)。T = 1(app==18)，Y = 用户是否转化。
数据来源：03 产出的 data/prepared/（用户级面板.parquet + 用户小时面板.parquet）。
产物：data/07_因果推断与稳健性.json + data/details/07_因果估计明细.csv /
      07_匹配前后协变量均衡.csv / 07_双重差分事件研究.csv / 07_双重差分单元格明细.csv
      + figures/07_因果推断图.png

运行：python 07_因果推断与稳健性.py
"""
import math
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import 公共工具 as C

# DID 的"伪干预时点"。⚠️ 本数据无真实政策冲击，这里是【按小时虚拟切出来的】
# 方法演示时点，不是真实上线时间；结论只能读作"两组日内节律差异"。
PSEUDO_POST_START_HOUR = 18


def _build_X(df, keep_levels, include_clicks=False):
    """构造倾向得分模型的特征矩阵（one-hot + 数值），列顺序固定。

    【输入】
      df             —— 用户级面板（含 os/device/channel/hour/n_clicks）
      keep_levels    —— 各分类变量保留的水平集合（由 _adaptive_top_k 决定，【显式传入】）
      include_clicks —— 是否把 log1p(n_clicks) 也作为协变量。
                        ⚠️ 默认 False 且【不应打开】：n_clicks 是实验期【同期】
                        变量、受处理 T 影响，把它当协变量会把处理效应本身
                        调掉（over-adjustment / 控制中介变量偏误）。
                        这也是本项目【不用 CUPED】的同一个原因。
    【输出】DataFrame，行数 = 分析单元数（本项目 296,709）；列数随 K 与 hour 水平数
            动态变化。
    【特征工程要点】
      · 高频分类变量（os/device/channel 各有几十到几百个水平）若全量 one-hot，
        会产生大量稀疏列、且低频水平的系数估计极不稳定。
        做法是只保留出现频次 Top-K，其余一律归入 "__other__" ——
        这是标准的【低频水平合并】。
        ⚠️ K 不固定为 8，而是由 _adaptive_top_k 按"覆盖率 ≥ 80%"自适应确定
        （三个变量集中度差异极大：实测 os 47.6% / channel 76.2% / device 100%，
        固定 K 会让 os 丢掉一半样本的区分度）。
      · hour 是 0~23 的整数，直接按类别 one-hot（24 列）而不当作连续变量，
        因为小时与转化率不是线性关系（夜间与白天模式不同）。
        ⚠️ 全量（full_csv）路径下 hour 曾被写死为 0，导致这 24 列退化为 1 列
        常数列、该协变量被静默丢弃；已在 03 修复为 min(hour)。
    """
    CAT_COLS = ["os", "device", "channel"]
    parts = {}
    for c in CAT_COLS:
        enc = np.where(df[c].isin(keep_levels[c]), df[c].astype(str), "__other__")
        d = pd.get_dummies(pd.Series(enc, index=df.index), prefix=c, dtype=float)
        parts[c] = d
    hour = pd.get_dummies(df["hour"].astype(str), prefix="hour", dtype=float)
    parts["hour"] = hour.set_axis(df.index)
    if include_clicks:
        parts["nclicks"] = pd.DataFrame(
            {"nclicks_log1p": np.log1p(df["n_clicks"].to_numpy(float))}, index=df.index)
    X = pd.concat([parts[c] for c in CAT_COLS] + [parts["hour"]]
                  + ([parts["nclicks"]] if include_clicks else []), axis=1)
    return X


def _adaptive_top_k(series, target=None, k_min=None, k_max=None):
    """自适应决定一个分类变量保留多少个水平 K（低频水平合并的 K 选择）。

    【为什么需要自适应】固定 K（如 8）在不同集中度的变量上代表完全不同的
      信息保留度：长尾变量会丢掉大半样本的区分度，集中变量又会保留过多
      近乎无用的稀疏列。用【覆盖率】作为统一尺子，K 的选择才有可解释依据。

    【输入】
      series —— 分类变量 Series（本项目：users["os"] / ["device"] / ["channel"]）
      target —— 目标覆盖率（默认 C.TOP_K_TARGET_COVERAGE = 0.80）
      k_min  —— K 的下限（默认 C.TOP_K = 8，保证不低于改造前的分辨率）
      k_max  —— K 的上限（默认 C.TOP_K_MAX = 30，防止 one-hot 维度爆炸、
                 也防止低频水平系数估计不稳）
    【输出】int，最终采用的 K。
    【确定性】只依赖 value_counts 的频次排序，与行序无关，可复现。
    """
    target = C.TOP_K_TARGET_COVERAGE if target is None else target
    k_min = C.TOP_K if k_min is None else k_min
    k_max = C.TOP_K_MAX if k_max is None else k_max
    vc = series.value_counts()
    n = len(series)
    k = k_min
    while k < k_max and float(vc.head(k).sum()) / n < target:
        k += 1
    return k


def _group_overview(users):
    """两组基础量：样本量 / 转化数 / 转化率。

    【为什么用 treat / Y 列】③ 在构建面板时就已经写好 `treat`（app==实验组 ? 1 : 0）
    与 `Y`（= converted，转成 int）。这里用布尔掩码一次性取两组，比反复
    `users[users.app == A]` 切片更快，也避免同一个语义出现两套写法。
    """
    m_ctrl = (users["treat"] == 0).to_numpy()
    m_trt = (users["treat"] == 1).to_numpy()
    Y_all = users["Y"].to_numpy(dtype=float)
    n_c, x_c = int(m_ctrl.sum()), int(Y_all[m_ctrl].sum())
    n_t, x_t = int(m_trt.sum()), int(Y_all[m_trt].sum())
    return {"n_c": n_c, "x_c": x_c, "n_t": n_t, "x_t": x_t,
            "p_c": x_c / n_c, "p_t": x_t / n_t}


def _sutva_diag(users):
    """7.0 SUTVA / 重叠诊断：同一 ip 是否同时进入两组。

    【返回】(sutva 字典, 去污染子样本字典)。sutva 字典同时落盘【两个口径】：
      · overlap_rate       —— 分母是【唯一 ip 数】
      · overlap_rate_unit  —— 换算成【分析单元 (ip,app)】口径
    两个数字都对但含义不同，必须同时给出、且报数时说明用的是哪个。
    """
    C.hr("7.0 SUTVA 诊断：同一 ip 是否同时进入两组")
    apps_per_ip = users.groupby("ip")["app"].nunique()
    n_both = int((apps_per_ip == 2).sum())
    n_only = int((apps_per_ip == 1).sum())
    overlap_rate = n_both / len(apps_per_ip)
    print(f"唯一 ip 数 = {len(apps_per_ip):,}；同时出现在两 app 的 ip = {n_both:,} ({overlap_rate:.2%})")
    # 口径说明：overlap_rate 的分母是【唯一 ip 数】；若要换算成【分析单元】口径，
    # 因每个重叠 ip 贡献 2 个 (ip, app) 单元，故需再乘 2 后除以单元总数。
    _unit_share = 2 * n_both / len(users)
    print(f"⚠️ SUTVA 违反：唯一 ip 中 {overlap_rate:.1%} 同时出现在两个 app"
          f"（换算为分析单元口径：受污染单元占比 {_unit_share:.1%}），两组非独立样本。")

    # ---- 去污染敏感性分析：只保留"单 app IP"子样本 ----
    only_ips = set(apps_per_ip[apps_per_ip == 1].index)
    sub = users[users.ip.isin(only_ips)]
    sc, st = sub[sub["treat"] == 0], sub[sub["treat"] == 1]
    # 【去污染敏感性：两组各自率也落盘，避免只存差值而丢失对比维度】
    #   sc/st 已于上行定义；这里取均值与样本量。下游报告/教程需要「对照 vs 实验」
    #   并排对比，只存差值会让报告失去可追溯到 data/07 的数据源。
    sc_rate, st_rate = float(sc.Y.mean()), float(st.Y.mean())
    n_sc, n_st = int(len(sc)), int(len(st))
    sub_diff = st_rate - sc_rate

    sutva = {"unique_ips": int(len(apps_per_ip)),
             "ips_in_both_apps": n_both, "ips_in_one_app": n_only,
             "overlap_rate": float(overlap_rate),
             "overlap_rate_unit": float(_unit_share),
             "naive_diff_single_app_subsample": float(sub_diff),
             "control_rate_single_app": sc_rate,
             "treatment_rate_single_app": st_rate,
             "n_single_app_control": n_sc,
             "n_single_app_treatment": n_st}
    return sutva


def _fit_ps_model(users):
    """7.2 倾向得分模型：自适应 Top-K one-hot → 逻辑回归 → 5-fold CV 评估。

    【返回】dict，含特征矩阵 X、处理/结果向量 T/Y、倾向得分 ps、logit(ps)、
    匹配卡尺、共同支撑区间、CV 指标，以及 Top-K 自适应的结果（供落盘）。
    """
    C.hr("7.2 倾向得分模型（逻辑回归）+ 5-fold CV 评估")

    # ---- 特征矩阵：低频水平合并，K 按覆盖率自适应 ----
    # 【修复：Top-K 固定 → 自适应】
    #   原实现对 os/device/channel 一律只保留频次前 8 个水平。问题在于三个变量的
    #   【集中度完全不同】：实测覆盖率 os 仅 47.6%、channel 76.2%、device 100%。
    #   固定 K 会让 os 有 52.4% 的用户被塞进 "__other__"，等于【丢掉了一半样本
    #   在该维度上的区分度】，而 device 却有大量几乎无用的稀疏列 —— 同一套 K
    #   在三个变量上完全不是同一个"信息保留度"。
    #   改为按【覆盖率目标】自适应：K 从 C.TOP_K(8) 起增大，直到覆盖率达到
    #   C.TOP_K_TARGET_COVERAGE(80%) 或触顶 C.TOP_K_MAX(30)。
    #   这样三个变量的信息保留度可比，且 K 的选择有了可解释的依据。
    keep_levels = {}
    for c in ["os", "device", "channel"]:
        k = _adaptive_top_k(users[c])
        keep_levels[c] = users[c].value_counts().head(k).index.tolist()
        print(f"  {c}: 自适应保留前 {k} 个水平（覆盖 "
              f"{users[c].isin(keep_levels[c]).mean():.1%} 用户，"
              f"目标 ≥ {C.TOP_K_TARGET_COVERAGE:.0%}）")

    X_all = _build_X(users, keep_levels)
    X = X_all.to_numpy(dtype=float)
    T = users["treat"].to_numpy(dtype=float)
    Y = users["Y"].to_numpy(dtype=float)
    print(f"\n特征矩阵: {X.shape[0]:,} 行 × {X.shape[1]} 列")

    # 样本外（交叉验证）倾向得分，用于评估"分组可被观测特征预测的程度"
    ps_model = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"))
    cv = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
    ps_cv = cross_val_predict(ps_model, X, T, cv=cv, method="predict_proba")[:, 1]
    ps_auc = float(roc_auc_score(T, ps_cv))
    ps_brier = float(brier_score_loss(T, ps_cv))
    print(f"5-fold CV: AUC = {ps_auc:.4f}，Brier = {ps_brier:.6f}")
    print("⚠️ 因果推断中 PS 的 AUC 越高越糟：AUC 高 = 两组可由观测特征近乎完全区分")
    print("   = 重叠（正值性）不足 = 因果不可识别。本例的高 AUC 正是")
    print("   『本数据不可因果解读』这一结论的量化证据，而非模型效果好的证明。")

    ps_model.fit(X, T)
    ps = np.clip(ps_model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    logit_ps = np.log(ps / (1 - ps))
    overlap_lo = max(ps[T == 0].min(), ps[T == 1].min())
    overlap_hi = min(ps[T == 0].max(), ps[T == 1].max())
    on_support = float(((ps >= overlap_lo) & (ps <= overlap_hi)).mean())
    caliper = C.CALIPER_SD * float(np.std(logit_ps))
    print(f"共同支撑区间 [ps] = [{overlap_lo:.3f}, {overlap_hi:.3f}]，"
          f"落在区间内样本占比 {on_support:.2%}")

    return {"X": X, "T": T, "Y": Y, "ps": ps, "logit_ps": logit_ps, "caliper": caliper,
            "keep_levels": keep_levels,
            "ps_auc": ps_auc, "ps_brier": ps_brier,
            "overlap_lo": float(overlap_lo), "overlap_hi": float(overlap_hi),
            "on_support": on_support,
            "n_features": int(X_all.shape[1]), "n_rows": int(X_all.shape[0])}


def _register_estimate(estimates, name, est, se, p_c, note="", base_rate=None, se_type=None):
    """登记一个估计量（append 到 estimates 并打印一行）。

    【输入】
      name/est/se/note —— 方法名 / 点估计（风险差）/ 标准误 / 备注。
      p_c              —— 用户级对照转化率，"相对提升"的默认分母。
      base_rate        —— 计算"相对提升"所用的【分母基线率】。默认 None 表示
                          回退到 p_c。
    【⚠️ 为什么 base_rate 必须可以显式指定 —— 修复了一处口径混用】
      relative_lift = est / base_rate 的分母必须与被估量【同口径】：
        · 朴素差异 / IPW 的 est 是【用户级】风险差 → 分母用用户级 p_c ✓
        · DID 的 est 是【用户-小时级（行级）】风险差 → 分母必须是同为该行级
          口径的"对照组基期率" _cell_rate(0,0)（本项目 ≈ 0.001041）。
      改动前所有估计量一律除以用户级 p_c（0.014939），DID 因此得到
      "-0.1557pp = -10.4%" —— 把一个【行级】效应除以【用户级】基线，
      量纲不匹配，该数字没有任何可解释含义（且会被误读成"转化下降 10%"）。
    【输出】无返回值（副作用：append 到 estimates）。
    """
    base = p_c if base_rate is None else base_rate
    lo, hi = est - 1.96 * se, est + 1.96 * se
    z = est / se if se > 0 else float("nan")
    p = 2 * (1 - stats.norm.cdf(abs(z))) if se > 0 else float("nan")
    estimates.append({
        "method": name, "estimate_risk_diff": est, "se": se,
        "ci_low": lo, "ci_high": hi, "z": z, "p_value": p,
        "relative_lift_vs_control": est / base, "note": note,
        # 【显式化】SE 是怎么来的写进产物：不同估计量的 SE 性质差别很大
        # （解析式 / bootstrap / 聚类稳健），放在一起比较时不能默认同质。
        "se_type": se_type, "baseline_used_for_relative_lift": float(base)})
    print(f"  {name:<28} 净增量 = {est * 100:+.4f} pp "
          f"(相对基线 {est / base:+.1%}) 95%CI [{lo*100:+.4f}, {hi*100:+.4f}] pp p={p:.4g}")


def _naive_diff(estimates, ov):
    """7.1 朴素差异（未调整）—— 对照基准线。返回点估计（供 _headline 用）。"""
    C.hr("7.1 朴素差异（未调整）—— 对照基准")
    p_t, p_c, n_t, n_c = ov["p_t"], ov["p_c"], ov["n_t"], ov["n_c"]
    diff_naive = p_t - p_c
    se_naive = math.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    _register_estimate(estimates, "Naive difference", diff_naive, se_naive, p_c,
                       "未调整，含选择偏差", se_type="解析式（非合并 SE）")
    return diff_naive


def _psm_diagnose(users, keep_levels, T, Y, logit_ps, caliper):
    """7.3 PSM 可行性诊断（1:1 最近邻 + caliper，无放回）—— 只诊断，不报告估计值。

    【为什么只诊断】匹配率过低 + 匹配后协变量仍未达平衡阈值 ⇒ 匹配样本不具
    代表性（ATT ≠ ATE，外推失效），此时报告任何匹配估计都会误导。故本函数
    只输出"可行/不可行"与诊断量，不产出效应估计。
    """
    C.hr("7.3 PSM 可行性诊断（1:1 最近邻 + caliper，无放回）")

    # ---- 匹配前后要比哪些协变量 ----
    bal_cols = {}
    for c in ["os", "device", "channel"]:
        for lv in keep_levels[c][:5]:
            bal_cols[f"{c}={lv}"] = (users[c] == lv).astype(float).to_numpy()
    bal_cols["hour"] = users["hour"].to_numpy(float)
    bal_cols["log1p(n_clicks)"] = np.log1p(users["n_clicks"].to_numpy(float))

    t_idx = np.where(T == 1)[0]
    c_idx = np.where(T == 0)[0]
    nn = NearestNeighbors(n_neighbors=min(25, len(c_idx)), algorithm="kd_tree")
    nn.fit(logit_ps[c_idx].reshape(-1, 1))
    dist, cand = nn.kneighbors(logit_ps[t_idx].reshape(-1, 1))
    used = np.zeros(len(c_idx), dtype=bool)
    # 【可复现性】用固定种子的随机排列决定"谁先挑"：贪心匹配的结果依赖于
    # 处理顺序，不固定顺序则每次跑出的匹配对不同，结论无法复现。
    # ⚠️【性能热点】下面是 138,534 × 最多 25 的纯 Python 双层循环（约 346 万次
    # 迭代），是全脚本最慢的一段。之所以【不向量化】：无放回贪心匹配依赖
    # used[] 的实时状态，向量化会改变抢占顺序从而改变匹配结果。
    # 若要优化，必须保证输出完全一致（可考虑 numba 保持串行语义）。
    order = np.random.default_rng(C.SEED).permutation(len(t_idx))
    pairs_t, pairs_c = [], []
    for i in order:
        # 候选按距离升序排列；一旦超出卡尺立即停止（后面的只会更远）
        for d_, j in zip(dist[i], cand[i]):
            if d_ > caliper:
                break
            if not used[j]:          # 无放回：每个对照最多被用一次
                used[j] = True
                pairs_t.append(t_idx[i]); pairs_c.append(c_idx[j])
                break
    pairs_t = np.array(pairs_t); pairs_c = np.array(pairs_c)
    n_pairs = len(pairs_t)
    print(f"匹配成功 {n_pairs:,} 对（实验组 {n_pairs/len(t_idx):.2%} 被匹配）")

    # ---- 匹配前后协变量均衡（Love plot 的数据来源）----
    rows = []
    for name, v in bal_cols.items():
        rows.append({"covariate": name, "smd_before": C.smd(v[T == 1], v[T == 0]),
                     "smd_after": C.smd(v[pairs_t], v[pairs_c])})
    rows.append({"covariate": "logit(ps)", "smd_before": C.smd(logit_ps[T == 1], logit_ps[T == 0]),
                 "smd_after": C.smd(logit_ps[pairs_t], logit_ps[pairs_c])})
    balance_df = pd.DataFrame(rows)
    balance_df["abs_before"] = balance_df.smd_before.abs()
    balance_df["abs_after"] = balance_df.smd_after.abs()
    smd_after_max = float(balance_df.abs_after.max())
    matched_rate = n_pairs / len(t_idx)
    print(f"匹配质量：匹配前 |SMD| 最大 {balance_df.abs_before.max():.4f}，"
          f"匹配后 {smd_after_max:.4f}")

    # 可行性判定：匹配率过低、或匹配后协变量仍未达平衡阈值 → 判定不可行。
    # 【阈值提常量】0.50 / 0.10 原先是硬编码在表达式里的"魔法数"，已提到
    # 公共工具的 PSM_MIN_MATCH_RATE / BALANCE_SMD_THRESHOLD，便于全局统一口径。
    # 【修复：判定理由必须如实反映"是哪一条不达标"】
    #   原实现只要判定不可行，就无条件打印"且匹配后 |SMD| 最大仍有 X（> 0.1 平衡阈值）"，
    #   把"匹配率过低"与"协变量未平衡"两个独立原因捆成一句，且【无论 SMD 实际多大
    #   都断言它超过了阈值】。本项目就是活例子：匹配率 3.98% 不达标（⇒ 判定不可行），
    #   但匹配后 |SMD| 最大只有 0.044（≤ 0.1，协变量其实已经平衡了），
    #   原文案会输出"0.044（> 0.1 平衡阈值）"这种自相矛盾的话。
    #   现在按【实际不达标的条件】逐条列原因；feasible 的布尔判定与改动前完全一致
    #   （matched_rate ≥ 0.5 且 smd_after_max ≤ 0.1），只修文案不改变判定。
    _reasons = []
    if matched_rate < C.PSM_MIN_MATCH_RATE:
        _reasons.append(f"实验组仅 {matched_rate:.2%} 被匹配（匹配 {n_pairs:,} 对，"
                        f"门槛 {C.PSM_MIN_MATCH_RATE:.0%}）")
    if smd_after_max > C.BALANCE_SMD_THRESHOLD:
        _reasons.append(f"匹配后 |SMD| 最大 {smd_after_max:.3f}"
                        f"（> {C.BALANCE_SMD_THRESHOLD} 平衡阈值）")
    psm_feasible = bool(not _reasons)
    if psm_feasible:
        psm_reason = (f"匹配率 {matched_rate:.2%}，匹配后 |SMD| 最大 {smd_after_max:.3f}"
                      f"（≤ {C.BALANCE_SMD_THRESHOLD}）")
    else:
        psm_reason = "；".join(_reasons)
    print(f"PSM 可行性判定 = {psm_feasible}；依据：{psm_reason}")
    if not psm_feasible:
        print("→ 匹配样本不具代表性（ATT ≠ ATE，外推失效），故【不报告 PSM 估计值】。")

    return {"n_pairs": n_pairs, "balance_df": balance_df,
            "psm_feasible": psm_feasible, "psm_reason": psm_reason,
            "treated_matched_rate": float(n_pairs / len(t_idx)),
            "control_used_rate": float(n_pairs / len(c_idx)),
            "smd_max_before": float(balance_df.abs_before.max()),
            "smd_max_after": smd_after_max}


def _ipw_att(T, Y, ps):
    """7.4 IPW 逆概率加权（ATT + 尾部截断 + bootstrap SE）。返回 (att_w, se_ipw)。

    【权重为什么是这个形式】目标估计量是 ATT（处理组的平均处理效应）：
      · 实验组权重恒为 1 ——保持原样，因为 ATT 问的就是"这批人"的效应；
      · 对照组权重 = e(x)/(1 − e(x)) ——即"进入实验组的几率"。
        对照组里 e(x) 越大（长得越像实验组）的单元，权重越高。
    效果：把对照组【重加权】成实验组的协变量分布，两组就可比了——
    这就是"用加权代替匹配"的核心直觉。
    【为什么要截断 trimming】当 e(x) → 1 时 e/(1−e) → ∞，极少数单元会
    主导整个加权和，方差爆炸。按 1%/99% 分位截断，用【一点偏倚】换
    【大幅降方差】（bias-variance tradeoff 的标准操作）。
    【为什么用 bootstrap 求 SE】加权后的估计量没有简洁的解析方差，
    直接对 (w, Y, T) 联合重抽样最稳妥。
    """
    C.hr("7.4 IPW 逆概率加权（ATT + 尾部截断 + bootstrap SE）")
    w = np.where(T == 1, 1.0, ps / (1.0 - ps))
    lo_q, hi_q = np.quantile(w, [0.01, 0.99])
    w_trim = np.clip(w, lo_q, hi_q)

    def att_ipw(weights, y, t):
        """Hájek 型（归一化）IPW-ATT 估计量。

        【公式】
            ATT = [ Σ_{i:T=1} w_i Y_i / Σ_{i:T=1} w_i ] − [ Σ_{i:T=0} w_i Y_i / Σ_{i:T=0} w_i ]
        【为什么叫 Hájek 型】分母用的是【权重之和】而不是样本量 n_g —— 这叫
          归一化加权（Hájek estimator）；若不除以权重和、直接除以 n_g 则是
          Horvitz–Thompson 型。Hájek 型的优点：权重整体乘以常数时估计值不变，
          且估计结果必然落在 [0,1] 内（Y 是 0/1 时），数值更稳；代价是引入了
          轻微的小样本偏倚。本项目两组权重和被用作分母，属 Hájek 型。
        【输入】weights/y/t —— 一维等长数组（权重、结果、处理指示）。
        【输出】float，ATT 的风险差（risk difference，比例单位）。
        """
        mt = (y[t == 1] * weights[t == 1]).sum() / weights[t == 1].sum()
        mc = (y[t == 0] * weights[t == 0]).sum() / weights[t == 0].sum()
        return float(mt - mc)

    att_w = att_ipw(w_trim, Y, T)
    _rng = np.random.default_rng(C.SEED)
    boot = np.empty(C.N_BOOT)
    n_all = len(Y)
    # 【⚠️ bootstrap 的边界 —— 显式说明，不装作完整】
    #   这里的重抽样只扰动 (w_trim, Y, T) 的【行】，权重是【固定】的：
    #   每个 bootstrap 样本里既没有重新拟合倾向得分模型，也没有重算 1/99 截断分位。
    #   因此这个 SE 【只反映抽样变异】，【未计入】① PS 模型参数的不确定性、
    #   ② 截断点的抽样变异 → 属于【保守性不足】的 SE（真实不确定性更大）。
    #   完整的做法是嵌套 bootstrap（外层重抽样、内层重估 PS），代价是 N_BOOT² 次
    #   逻辑回归拟合（400×400 = 16 万次），在 29.7 万×25 维上不可行。
    #   由于本项目结论是"不可因果识别"，SE 略微偏小不改变任何定性结论，
    #   故保留此权衡；但在正式报告中应注明"SE 未含 PS 估计不确定性"。
    for b in range(C.N_BOOT):
        idx = _rng.integers(0, n_all, n_all)
        boot[b] = att_ipw(w_trim[idx], Y[idx], T[idx])
    se_ipw = float(boot.std(ddof=1))
    return att_w, se_ipw


def _did_analysis(panel, treatment_app):
    """7.5 DID 双重差分（⚠️ 方法演示：无真实政策冲击）。

    【返回】(did 字典, 带 T/post 列的 panel, 事件研究表 ev)。DID 的效应登记
    （写入 estimates）由调用方完成，因为 estimates 列表由 run() 统一持有。
    """
    C.hr("7.5 DID 双重差分（⚠️ 方法演示：无真实政策冲击）")
    panel = panel.copy()
    panel["T"] = (panel["app"] == treatment_app).astype(int)
    panel["post"] = (panel["hour"] >= PSEUDO_POST_START_HOUR).astype(int)
    panel["y"] = panel["y"].astype(int)
    cell = panel.groupby(["T", "post"], as_index=False).agg(n_obs=("y", "size"),
                                                           n_conv=("y", "sum"))
    cell["rate"] = cell["n_conv"] / cell["n_obs"]
    print(cell.to_string(index=False))

    def _cell_rate(t, post):
        r = cell[(cell["T"] == t) & (cell.post == post)]
        return float(r.rate.iloc[0])

    did_2x2 = (_cell_rate(1, 1) - _cell_rate(1, 0)) - (_cell_rate(0, 1) - _cell_rate(0, 0))
    Xr = sm.add_constant(panel[["T", "post"]].assign(interact=panel["T"] * panel.post))
    # 【为什么必须按单元聚类，HC1 不够 —— 修复了一处 SE 低估】
    #   panel 是【长表】：同一个 (ip, app) 单元在多个 hour 上重复出现（本项目
    #   4,857,118 行 / 296,709 单元 ≈ 每行单元出现 16 次）。同一单元的残差必然
    #   相关（同一用户这一小时点了，下一小时大概率也点）。
    #   HC1 只修正【异方差】，仍假设【行间独立】；在长表面板上它会把 SE 严重
    #   低估（改动前给出 p ≈ 1.5e-52，就是低估 SE 的典型症状）。
    #   正规做法是 cluster-robust：允许【簇内任意相关】，簇间独立。
    #   聚类键取 (ip, app) —— 与 ① 声明的分析单元【严格一致】；
    #   （若改成按 ip 聚类会更保守，因为会把同一 ip 在两个 app 的相关性也计入，
    #    但那等于默认两个 app 的行属于同一单元，与本项目"分析单元 = (ip,app)"矛盾。）
    _cluster_id, _ = pd.factorize(
        panel["ip"].astype(str) + "|" + panel["app"].astype(str))
    ols = sm.OLS(panel["y"].astype(float), Xr).fit(
        cov_type="cluster", cov_kwds={"groups": _cluster_id})
    did_coef = float(ols.params["interact"])
    did_se = float(ols.bse["interact"])
    did_p = float(ols.pvalues["interact"])
    # 同时保留 HC1 结果【仅作对照展示】（不参与任何判定），便于直观看出
    # "忽略簇内相关会把 SE 低估多少"——这是面板数据最经典的 SE 陷阱之一。
    _se_hc1 = float(sm.OLS(panel["y"].astype(float), Xr)
                    .fit(cov_type="HC1").bse["interact"])
    print(f"DID SE 对照：HC1（假设行间独立）= {_se_hc1:.6e} → "
          f"按 (ip,app) 聚类 = {did_se:.6e}（放大 {did_se / _se_hc1:.2f}×）；"
          f"簇数 = {len(np.unique(_cluster_id)):,}，面板行数 = {len(panel):,}")

    # ---- 事件研究（供画图 + 落 CSV）----
    hourly = panel.groupby(["hour", "T"], as_index=False).agg(n=("y", "size"),
                                                             conv=("y", "sum"))
    hourly["rate"] = hourly["conv"] / hourly["n"]
    ev = hourly.pivot(index="hour", columns="T", values="rate")
    ev.columns = ["control", "treatment"]
    ev["gap"] = ev["treatment"] - ev["control"]
    ev = ev.reset_index()
    both_apps = hourly.groupby("hour")["T"].nunique() == 2
    base_hour = int(both_apps[both_apps].index.min())
    ev["gap_vs_base"] = ev["gap"] - ev.loc[ev.hour == base_hour, "gap"].iloc[0]

    did = {"disclaimer": "方法演示：无真实政策冲击，虚拟干预时点=18:00",
           "pseudo_post_start_hour": PSEUDO_POST_START_HOUR,
           "did_2x2_manual": float(did_2x2), "did_coef": did_coef,
           "did_se": did_se, "did_p": did_p,
           "did_se_type": "cluster((ip,app))",
           "did_se_hc1_for_reference": _se_hc1,
           "did_baseline_for_relative_lift": float(_cell_rate(1, 0)),
           "did_baseline_meaning": "实验组基期(post=0)行级转化率（ATT 相对化的分母）",
           "cell": cell, "base_hour": base_hour}
    return did, panel, ev


def _placebo_test(panel, base_hour):
    """7.5b 安慰剂（伪干预）置换检验：在 pre 期虚拟一个干预时点，看 DID 是否显著。

    【修复：观测统计量与置换分布必须同一设计】
      改动前是【错配】的：
        · 观测统计量 obs_placebo 只用【2 个小时】(base_hour, base_hour+1)；
        · 置换分布却在【整个 pre 期 16~18 个小时】上打乱标签。
      两者压根不是同一个估计量 —— 拿一个 2 小时的 DID 去和一个 18 小时的
      DID 分布比大小，得到的 p 值没有意义（这正是置换检验最容易犯的错：
      置换必须复现【原统计量的完整计算过程】，只打乱那一个变量）。
      修复：统一用"伪干预时点 = base_hour+1"把整个 pre 期切成两段，
      pre = [base_hour, base_hour+1)、post = [base_hour+1, POST_START_HOUR)，
      观测与置换都在【同一段数据、同一套切分】上算同一个 DID 公式。
    【返回】(placebo_coef, placebo_p)。
    """
    pre_panel = panel[(panel.hour >= base_hour) & (panel.hour < PSEUDO_POST_START_HOUR)].copy()
    _fake_post = (pre_panel["hour"] >= base_hour + 1)
    pre_panel["fp"] = _fake_post.astype(int)
    _rates_obs = pre_panel.groupby(["T", "fp"])["y"].mean()

    def _r(t, p):
        return float(_rates_obs.get((t, p), float("nan")))

    obs_placebo = ((_r(1, 1) - _r(1, 0)) - (_r(0, 1) - _r(0, 0)))
    true_fake = _fake_post.astype(int).to_numpy()
    boot_p = 0
    for _ in range(C.N_BOOT):
        perm = np.random.default_rng(C.SEED + _).permutation(true_fake)
        g = pre_panel.assign(fp=perm)
        rates = g.groupby(["T", "fp"])["y"].mean()
        d = (rates.get((1, 1), 0.0) - rates.get((1, 0), 0.0)) - (
            rates.get((0, 1), 0.0) - rates.get((0, 0), 0.0))
        if abs(d) >= abs(obs_placebo):
            boot_p += 1
    placebo_p = (boot_p + 1) / (C.N_BOOT + 1)
    placebo_coef = float(obs_placebo)
    print(f"安慰剂检验（pre 期 {base_hour}-{PSEUDO_POST_START_HOUR} 点，伪干预设在 {base_hour+1} 点）："
          f"假 DID = {placebo_coef*100:+.4f} pp, 置换 p = {placebo_p:.4f} -> "
          f"{'不显著 ✅（支持平行趋势）' if placebo_p > C.ALPHA else '显著 ⚠️'}")
    print("  （观测与置换已对齐：同一段数据、同一伪干预切分、同一 DID 公式）")
    return placebo_coef, placebo_p


def _evalue_rows(estimates, p_c):
    """7.6 E-value 敏感性分析：把每个估计量换算成"需要多强的未观测混杂才能解释掉"。

    【E 值回答什么】"需要一个多强的【未观测混杂】——它同时与分组 T 和结果 Y
    相关——才能把当前观测到的关联完全解释掉？"
       E = RR + √[RR(RR − 1)]
    【怎么读】本项目朴素 RR = 3.1349 → E = 5.7220，意味着需要 RR ≥ 5.72 的
    未观测混杂才能抹平关联，门槛很高。
    【⚠️ 关键边界 —— 面试高频】E 值只处理【混杂强度】问题，它【默认其他识别
    假设都成立】。而本项目崩掉的是【可识别性】本身：SUTVA 违反（83.98% 单元
    来自双侧 IP）+ 重叠不足（共同支撑内仅 16.15%、PS 的 CV-AUC 0.9894）。
    这两类问题让估计目标（estimand）本身没有定义，混杂再弱也救不了。
    所以正确结论是："关联不易被混杂解释掉"，但【不能升级为因果】。
    """
    C.hr("7.6 E-value 敏感性分析")

    def e_value(rr):
        if rr is None or rr <= 1 or not np.isfinite(rr):
            return None
        return float(rr + math.sqrt(rr * (rr - 1.0)))

    rows = []
    for est in estimates:
        rr = (p_c + est["estimate_risk_diff"]) / p_c
        rows.append({"method": est["method"], "risk_ratio": rr, "e_value": e_value(rr)})
    return rows


def _headline(diff_naive, att_w, p_c):
    """根据 IPW 相对朴素差异的方向，生成 recommended_headline 的 finding / interpretation。"""
    _frac = att_w / diff_naive if diff_naive != 0 else float("nan")
    if _frac >= 1:
        _adj_pct = _frac - 1
        _finding = (f"朴素相对提升 +{diff_naive/p_c:.1%} 经 IPW 调整 device/OS/channel/hour 后"
                    f"不减反增、升至 +{att_w/p_c:.1%}（净增量 {diff_naive*100:+.4f}pp → {att_w*100:+.4f}pp，仍显著），"
                    f"说明可观测混杂未削弱、反而放大了表观关联（约 {_adj_pct:.0%} 的放大）")
        _interp = (f"控制 device/OS/channel 后关联不降反升，表明朴素估计其实低估了表观关联；"
                   f"两组特征分布差异会『掩盖』而非『制造』差异，叠加 SUTVA 违反，更不可断言为 app 本身的处理效应")
    else:
        _adj_pct = 1 - _frac
        _finding = (f"朴素相对提升 +{diff_naive/p_c:.1%} 经 IPW 调整 device/OS/channel/hour 后"
                    f"收缩至 +{att_w/p_c:.1%}（净增量 {diff_naive*100:+.4f}pp → {att_w*100:+.4f}pp，仍显著），"
                    f"约 {_adj_pct:.0%} 的原始差异可由可观测混杂解释")
        _interp = (f"原始相对提升 +{diff_naive/p_c:.1%} 中有 {_adj_pct:.0%} "
                   "可由 device/OS/channel 构成的【可观测混杂】解释；"
                   "调整后残余关联仍显著，叠加 SUTVA 违反，不可断言为 app 本身的处理效应")
    return _finding, _interp


def _draw_figure(est_df, balance_df, ps, T, ev, did, placebo, diff_naive, p_c,
                 control_app, treatment_app):
    """画 2×2 因果推断图：森林图 / Love plot / 倾向得分重叠 / 事件研究。"""
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    fig.suptitle(
        "Causal Inference - From Naive Gap to Adjusted Net Lift\n"
        f"(TalkingData full app12/18 | 4-day | unit = (ip, app) | baseline p_c = {p_c:.4%})",
        fontsize=13.5)
    ax = axes[0, 0]
    d = est_df.iloc[::-1].reset_index(drop=True)
    ypos = np.arange(len(d))
    ax.errorbar(d.estimate_risk_diff * 100, ypos, xerr=1.96 * d.se * 100, fmt="o",
                color="#4C72B0", ecolor="#4C72B0", capsize=4, ms=6, lw=1.6)
    ax.axvline(0, color="#C44E52", ls="--", lw=1.4, label="no effect")
    ax.axvline(diff_naive * 100, color="gray", ls=":", lw=1.4, label="naive gap")
    ax.set_yticks(ypos); ax.set_yticklabels(d.method, fontsize=8.5)
    ax.set_xlabel("Net lift (percentage points, 95% CI)")
    ax.set_title("(1) Forest plot: estimates of the net lift")
    ax.legend(fontsize=8.5); ax.grid(axis="x", alpha=0.3)
    ax = axes[0, 1]
    bp = balance_df.sort_values("abs_before", ascending=False)
    ypos = np.arange(len(bp))
    ax.scatter(bp.abs_before, ypos, s=42, color="#C44E52", marker="o", label="before", zorder=3)
    ax.scatter(bp.abs_after, ypos, s=42, color="#55A868", marker="s", label="after", zorder=3)
    for i in range(len(bp)):
        ax.plot([bp.abs_before.iloc[i], bp.abs_after.iloc[i]], [i, i],
                color="gray", lw=1, alpha=0.5, zorder=1)
    ax.axvline(0.1, color="black", ls="--", lw=1.3, label="|SMD|=0.1")
    ax.set_yticks(ypos); ax.set_yticklabels(bp.covariate, fontsize=7.5)
    ax.set_xlabel("|Standardized Mean Difference|")
    ax.set_title("(2) Love plot: covariate balance")
    ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=0.3)
    ax = axes[1, 0]
    bins = np.linspace(0, 1, 51)
    ax.hist(ps[T == 0], bins=bins, alpha=0.6, color="#4C72B0", label=f"control app{control_app}", density=True)
    ax.hist(ps[T == 1], bins=bins, alpha=0.6, color="#DD8452", label=f"treatment app{treatment_app}", density=True)
    ax.set_xlabel("Propensity score"); ax.set_ylabel("Density")
    ax.set_title("(3) Propensity score overlap")
    ax.legend(fontsize=8.5)
    ax = axes[1, 1]
    ax.plot(ev.hour, ev.gap * 100, "o-", color="#8172B2", lw=2, ms=6, label="treatment - control")
    ax.axhline(0, color="gray", lw=1)
    ax.axvline(PSEUDO_POST_START_HOUR - 0.5, color="#C44E52", ls="--", lw=1.6,
               label=f"pseudo shock @ {PSEUDO_POST_START_HOUR}:00")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Gap in conversion rate (pp)")
    ax.set_title("(4) Event study (DID demo)\nDID=%+.3f pp (p=%.3f) | placebo=%+.3f pp (p=%.3f)"
                 % (did["did_coef"] * 100, did["did_p"],
                    placebo["coef"] * 100, placebo["p"]))
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = C.FIG_DIR / "07_因果推断图.png"
    fig.savefig(fig_path, dpi=130)
    print(f"\nSaved figure: {fig_path}")
    return fig_path


def run():
    """⑦ 因果推断主流程：按 7.0→7.6 编排各分析块，汇总落盘。

    本函数【不做具体计算】，只负责：调用各块 → 收集结果 → 组装 summary →
    落盘 JSON/CSV/图。要看某一步怎么算，直接跳到对应的函数即可。
    """
    C.hr("⑦ 因果推断（PSM 诊断 / IPW / DID / E-value）")
    prepared = C.load_prepared()
    users = prepared.users.copy()
    panel = prepared.panel.copy()
    A, B = C.CONTROL_APP, C.TREATMENT_APP

    # ---------- 7.0 SUTVA / 重叠诊断 ----------
    sutva = _sutva_diag(users)
    ov = _group_overview(users)
    n_c, n_t = ov["n_c"], ov["n_t"]
    p_c, p_t = ov["p_c"], ov["p_t"]

    # ---------- 7.1 朴素差异 ----------
    estimates = []
    diff_naive = _naive_diff(estimates, ov)

    # ---------- 7.2 倾向得分模型 ----------
    ps_pack = _fit_ps_model(users)
    T, Y, ps, logit_ps = ps_pack["T"], ps_pack["Y"], ps_pack["ps"], ps_pack["logit_ps"]

    # ---------- 7.3 PSM 可行性诊断 ----------
    psm = _psm_diagnose(users, ps_pack["keep_levels"], T, Y, logit_ps, ps_pack["caliper"])

    # ---------- 7.4 IPW ----------
    att_w, se_ipw = _ipw_att(T, Y, ps)
    _register_estimate(estimates, "IPW (ATT, trimmed 1/99)", att_w, se_ipw, p_c,
                       "bootstrap SE（未含 PS 估计与截断点的不确定性）",
                       se_type=f"bootstrap {C.N_BOOT} 次（仅行重抽样，权重固定）")

    # ---------- 7.5 DID + 安慰剂 ----------
    did, panel, ev = _did_analysis(panel, B)
    # 【口径】DID 是【行级（用户-小时级）】估计量，分母必须是同为行级的基期率，
    # 绝不能再除以用户级 p_c（0.014939）——那会把行级效应除以用户级基线（详见 _register_estimate）。
    # 【分母为什么选 (T=1, post=0) 即"实验组基期率"而不是对照组基期率】
    #   DID 的估计目标是 ATT（对处理组的效应），相对化的自然基准是
    #   【被处理组自身在干预前的水平】——这也是应用计量里报告"DiD 相当于前期均值的
    #   百分之多少"时的通行做法。若改用对照组基期率（0.1041%）作分母，会得到
    #   -0.1557pp / 0.1041% = -149.6% 这种"降幅超过 100%"的越界数字，
    #   虽算术无误但不可解释（速率最多降到 0），反而制造误导。
    #   本项目：−0.001557 / 0.003722 = −41.8%，读作"净掉对照组的自然趋势后，
    #   实验组行级转化相对其自身基期水平下降约 41.8%"。
    _register_estimate(estimates, "DID (pseudo shock @18:00)", did["did_coef"], did["did_se"], p_c,
                       "方法演示，cluster SE（按 (ip,app) 聚类）；相对基线分母=实验组基期行级率",
                       base_rate=did["did_baseline_for_relative_lift"],
                       se_type="cluster-robust（按 (ip,app) 聚类）")

    placebo_coef, placebo_p = _placebo_test(panel, did["base_hour"])
    placebo = {"coef": placebo_coef, "p": placebo_p,
               "hour": int(did["base_hour"] + 1)}

    # ---------- 7.6 E-value ----------
    ev_rows = _evalue_rows(estimates, p_c)
    est_df = pd.DataFrame(estimates)

    _finding, _interp = _headline(diff_naive, att_w, p_c)

    summary = {
        "data": {"unit": "(ip, app)", "n_units": int(len(users)),
                 "n_control": n_c, "n_treatment": n_t,
                 "control_rate": float(p_c), "treatment_rate": float(p_t)},
        # 【双口径显式化】overlap_rate 的分母是【唯一 ip 数】；overlap_rate_unit
        # 换算成【分析单元】口径（每个重叠 ip 贡献 2 个 (ip,app) 单元）。
        # 两个数字都对，但含义不同，必须同时落盘、且下游报数时要说明用的是哪个。
        "sutva_overlap": sutva,
        # 【显式化】把 7.2 里只在控制台打印过的 PS 诊断量写进产物，
        # 便于下游报告/对比脚本直接读取，不必再翻日志。
        "ps_model": {"cv_auc": ps_pack["ps_auc"], "brier": ps_pack["ps_brier"],
                     "common_support": [ps_pack["overlap_lo"], ps_pack["overlap_hi"]],
                     "on_support_share": ps_pack["on_support"]},
        # 【显式化】特征矩阵的规模与 Top-K 自适应结果（改动后 K 不再固定为 8）
        "ps_feature_info": {"n_features": ps_pack["n_features"],
                            "n_rows": ps_pack["n_rows"],
                            "top_k": {c: len(ps_pack["keep_levels"][c])
                                      for c in ["os", "device", "channel"]},
                            "coverage": {c: float(users[c].isin(ps_pack["keep_levels"][c]).mean())
                                         for c in ["os", "device", "channel"]}},
        "estimates": estimates,
        "psm": {"n_pairs": psm["n_pairs"], "caliper": float(ps_pack["caliper"]),
                "treated_matched_rate": psm["treated_matched_rate"],
                "control_used_rate": psm["control_used_rate"],
                "common_support": [ps_pack["overlap_lo"], ps_pack["overlap_hi"]],
                "smd_max_before": psm["smd_max_before"],
                "smd_max_after": psm["smd_max_after"]},
        "did": dict(disclaimer=did["disclaimer"],
                    pseudo_post_start_hour=did["pseudo_post_start_hour"],
                    did_2x2_manual=did["did_2x2_manual"], did_coef=did["did_coef"],
                    did_se=did["did_se"], did_p=did["did_p"],
                    did_se_type=did["did_se_type"],
                    did_se_hc1_for_reference=did["did_se_hc1_for_reference"],
                    did_baseline_for_relative_lift=did["did_baseline_for_relative_lift"],
                    did_baseline_meaning=did["did_baseline_meaning"],
                    placebo_pseudo_shock_hour=placebo["hour"],
                    placebo_coef=placebo["coef"], placebo_p=placebo["p"],
                    placebo_note=f"伪干预设在 {placebo['hour']}:00；观测统计量与置换分布"
                                 f"使用同一段数据（hour ∈ [{did['base_hour']}, {PSEUDO_POST_START_HOUR})）"
                                 f"与同一套切分，已修正此前'观测 2 小时 / 置换 18 小时'的错配",
                    parallel_trend_supported=bool(placebo["p"] > C.ALPHA)),
        "e_value": ev_rows,
        "recommended_headline": {
            "finding": _finding,
            "adjuster_methods": {e["method"]: e["estimate_risk_diff"] for e in estimates
                                 if e["method"] in ("IPW (ATT, trimmed 1/99)",)},
            "psm_feasible": psm["psm_feasible"],
            "psm_reason": psm["psm_reason"],
            "interpretation": _interp,
            "hard_limit": f"SUTVA 违反（ip 重叠 {sutva['overlap_rate']:.1%}）+ 重叠不足，本数据【无法】做严格因果识别；"
                         "结论须限定为'可观测混杂调整后的关联'，不可声称因果净增量"},
    }

    # ---------- 落盘 ----------
    est_df.to_csv(C.DETAILS_DIR / "07_因果估计明细.csv", index=False)
    psm["balance_df"].to_csv(C.DETAILS_DIR / "07_匹配前后协变量均衡.csv", index=False)
    ev.to_csv(C.DETAILS_DIR / "07_双重差分事件研究.csv", index=False)
    did["cell"].to_csv(C.DETAILS_DIR / "07_双重差分单元格明细.csv", index=False)

    _draw_figure(est_df, psm["balance_df"], ps, T, ev, did, placebo, diff_naive, p_c, A, B)

    C.dump_json(summary, C.DATA_DIR / "07_因果推断与稳健性.json")
    C.hr("✅ ⑦ 因果推断与稳健性完成")
    print(f"  产物: {C.DATA_DIR / '07_因果推断与稳健性.json'}")
    print(f"        {C.DETAILS_DIR / '07_因果估计明细.csv'}")
    print(f"        {C.DETAILS_DIR / '07_匹配前后协变量均衡.csv'}")
    print(f"        {C.DETAILS_DIR / '07_双重差分事件研究.csv'}")
    print(f"        {C.DETAILS_DIR / '07_双重差分单元格明细.csv'}")
    return summary


if __name__ == "__main__":
    run()
