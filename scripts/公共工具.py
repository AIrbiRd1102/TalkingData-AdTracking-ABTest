"""
公共工具.py —— 全项目共享工具（不是「流程步骤」，只是被各步骤 import 的通用函数）

集中放置：
  · 项目级常量（ALPHA / POWER / SEED / 分组 app / 各方法超参）
  · 路径（PROJECT_ROOT / DATA_DIR / FIG_DIR / PREPARED_DIR）
  · 统计工具（to_python / two_prop_z / smd / inv_cohens_h / OBF 标定 / dump_json / hr）
  · 中间文件读写（save_prepared / load_prepared）—— 各「大流程步骤」之间用 parquet 交接

各步骤（01~08）都 `import 公共工具`，从本文件取常量与工具，保证口径一致。
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import 配置  # noqa: E402  （配置.py 与本文件同在 scripts/ 目录）


# ============================================================
# 路径
# ============================================================
# 本文件位于 scripts/ 子目录，故需上溯两级到项目根：
#   scripts/公共工具.py → parent = scripts/ → parent.parent = 项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"
PREPARED_DIR = DATA_DIR / "prepared"
DETAILS_DIR = DATA_DIR / "details"   # 各步骤的明细 CSV（供核查/Tableau），与小结 JSON 分离
for _d in (DATA_DIR, FIG_DIR, PREPARED_DIR, DETAILS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 全局常量（与旧脚本口径完全一致）
# ============================================================
ALPHA = 0.05
POWER = 0.80
SEED = 42
CONTROL_APP, TREATMENT_APP = 12, 18

# 因果推断超参
TOP_K = 8            # os / device / channel 各保留前 8 个水平（自适应 Top-K 的【下限】）
N_BOOT = 400         # IPW / 匹配 / 安慰剂 的 bootstrap 次数
N_FOLDS = 5          # 交叉验证折数（倾向得分模型 CV）
CALIPER_SD = 0.20    # 匹配卡尺 = 0.2 × sd(logit ps)

# ---- 判定阈值（原散落在各步骤里的"魔法数"集中于此，便于统一口径与审阅）----
BALANCE_SMD_THRESHOLD = 0.10   # 协变量均衡阈值 |SMD|（业界通行；匹配后需 ≤ 此值）
CATEGORICAL_BALANCE_GAP = 0.01 # 分类协变量均衡阈值：各水平【占比差】的最大值（步骤④ 用）
                               # ⚠️ 与上面 |SMD|<0.1 【不是同一个尺度】（占比差 vs 标准差），报数时务必说明用的是哪个
PSM_MIN_MATCH_RATE = 0.50      # PSM 可行性门槛：实验组被匹配比例需 ≥ 此值
TOP_K_TARGET_COVERAGE = 0.80   # 低频水平合并后要求的最低覆盖率（Top-K 自适应用）
TOP_K_MAX = 30                 # Top-K 自适应上限（防止 one-hot 维度爆炸、系数不稳）

# ============================================================
# 实验规划假设（仅用于「步骤② 实验设计」）
# —— 设计阶段尚未看到实验结果，因此以下全部是「规划/假设值」，绝非观测值 ——
# 真实 A/B 测试里这些应在看数据前由业务/历史试点定好（预注册）。
# ============================================================
ASSUMED_BASELINE_RATE = 0.002    # 规划假设基线转化率（领域经验/历史试点值，非本实验观测值）
TARGET_MDE_REL = 0.10            # 最小值得检测的相对提升（+10%），实验前由业务设定
ASSUMED_DAILY_TRAFFIC = 200_000  # 规划假设日均 (ip,app) 流量（估周期用；是吞吐量，非效应）
GUARDRAIL_MARGIN = 0.05          # 护栏指标单侧劣效容差（相对降幅上限 −5%，即 δ）

# ============================================================
# 随机分流方案（步骤②设计 / 步骤④有效性校验 共用同一套参数）
# ============================================================
N_BUCKETS = 100
BUCKET_SALT = "exp_abtest_main_v1"
SPLIT_A_FRACTION = 0.5           # 桶编号 < N_BUCKETS*SPLIT_A_FRACTION 归 A 组（其余 B 组）


# ============================================================
# 通用工具
# ============================================================
def hr(title):
    """打印大步骤标题分隔线。"""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def to_python(o):
    """递归清洗 numpy / pandas / 浮点 NaN，保证 json 可序列化。

    【输入】任意 Python / numpy / pandas 对象（本项目：各步骤的 summary 字典）。
    【输出】只含 str / int / float / bool / None / list / dict 的结构，可直接 json.dump。

    【为什么必须有这个函数 —— 两个真实的坑】
      ① numpy 标量（np.int64 / np.float64 / np.bool_）不是 JSON 原生类型，
         不转换会抛 `TypeError: Object of type int64 is not JSON serializable`。
      ② NaN / Inf 在【标准 JSON 里没有字面量】。Python 的 json.dump 默认会写出
         `NaN` / `Infinity`，那是 JSON 的方言扩展，严格解析器（以及前端
         JSON.parse）会直接报错。所以这里统一转成 None（→ JSON 的 null）。

    【已修正的缺陷】原实现把 float 的 NaN 判断放在
    `isinstance(o, (str, int, float, bool))` 之后，导致内置 float 的 NaN
    【永远走不到】那个分支、被原样返回，写出非法 JSON。本版把 NaN 判断提到
    最前面，使 numpy 与内置 float 行为一致。
    """
    # ⚠️ NaN 判断必须放在"原生类型直接返回"之前，否则内置 float 的 NaN 会漏出
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: to_python(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_python(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 8)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    return str(o)


def two_prop_z(c1, n1, c0, n0):
    """两比例 Z 检验（【合并标准误】版本），返回 (diff, z, p)。

    【输入】
      c1, n1 —— 实验组（treatment）的转化单元数 / 单元总数
      c0, n0 —— 对照组（control）的转化单元数 / 单元总数
    【输出】
      diff —— 绝对提升 p1 - p0（点估计）
      z    —— 标准化统计量 = diff / SE
      p    —— 双侧 p 值 = 2[1 - Φ(|z|)]
    【统计原理】
      H0 假设两组真实比例相等（p1 = p0 = p）。既然假设同率，把所有数据并起来
      估计这个共同比例 p̄ = (c1+c0)/(n1+n0) 最有效，因此分母用【合并】标准误
          SE_pooled = √[ p̄(1-p̄) · (1/n1 + 1/n0) ]
      ⚠️ 与构造置信区间的【非合并】标准误
          SE_unpooled = √[ p1(1-p1)/n1 + p0(1-p0)/n0 ]
      是不同的两个数（本项目 6.25998e-4 vs 6.44412e-4）。一句话记忆：
      **检验站在 H0 的世界里算，区间站在 H1 的世界里算**。
    【注意】本项目 05 步实际调用的是 statsmodels 的 proportions_ztest，
      与这里公式一致；07 步的朴素差异用的是非合并 SE，故两者 Z 值略有差异
      （50.95 vs 49.49），属正常。
    """
    if n1 <= 0 or n0 <= 0:
        return float("nan"), float("nan"), float("nan")
    p1, p0 = c1 / n1, c0 / n0
    diff = p1 - p0
    pp = (c1 + c0) / (n1 + n0)                      # 合并比例 p̄（H0 下的共同率）
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))  # 合并标准误
    if se <= 0:
        return diff, float("nan"), float("nan")
    z = diff / se
    from scipy import stats
    return diff, z, 2 * (1 - stats.norm.cdf(abs(z)))


def bucket_of(value, salt=BUCKET_SALT, n_buckets=N_BUCKETS):
    """确定性哈希分桶：同一 (salt, value) 永远落到同一桶，保证分流可复现。

    【输入】
      value     —— 分流键（本项目是 ip；真实业务常用 device_id / user_id）
      salt      —— 实验盐值。不同实验用不同 salt ⇒ 分流互相正交（近似独立）
      n_buckets —— 桶数（本项目 100）
    【输出】[0, n_buckets) 的整数桶号。
    【为什么这样设计 —— 分流的三条硬要求】
      ① 确定性：同 ip 每次算都落同桶（不能每次刷新就换组），靠 md5 保证；
      ② 均匀性：md5 输出近似均匀，取前 16 位十六进制（64 bit）再取模，
         比直接用 Python 内置 hash() 更稳（内置 hash 对字符串加了随机种子，
         跨进程不一致，会毁掉可复现性）；
      ③ 正交性：换 salt 相当于重新洗牌，两个实验的分流互不相关，
         可并行跑多实验而互不干扰。
    用于步骤②（描述分流方案）与步骤④（在真实 ip 上验证分桶质量）。
    """
    key = f"{salt}:{value}".encode("utf-8")
    # 取 hexdigest 前 16 位（=64 bit）转整数再取模；不用内置 hash() 是因为
    # Python 3 起 str 的 hash 带随机种子（PYTHONHASHSEED），跨进程不可复现。
    return int(hashlib.md5(key).hexdigest()[:16], 16) % n_buckets


def smd(x_a, x_b):
    """标准化均值差 SMD（Standardized Mean Difference，连续/二元变量）。

    【输入】x_a, x_b —— 同一协变量在两组中的取值数组（本项目：匹配前后的协变量）。
    【输出】SMD 标量。
    【统计原理】
        SMD = (mean_a - mean_b) / √[(s²_a + s²_b) / 2]
      分母是"合并标准差的近似"，因此 SMD 是以【标准差】为尺子的差异，
      **与样本量无关**，可以跨变量、跨实验横向比较——这正是它优于
      "直接比均值差"和"直接做 t 检验"的原因（后两者都随 n 放大）。
    【阈值】业界通行 |SMD| < 0.1 视为均衡（本项目 ⑦ 的 PSM 用此阈值）。
      ⚠️ 注意本项目 ④ 用的是另一套尺度："分类变量各水平占比差 < 0.01"。
      两个指标量纲不同、阈值不同，报数时务必说明用的是哪个。
    """
    x_a, x_b = np.asarray(x_a, float), np.asarray(x_b, float)
    sa, sb = x_a.var(ddof=1), x_b.var(ddof=1)
    pooled = math.sqrt((sa + sb) / 2.0)
    if pooled < 1e-12:
        # 常数列（方差为 0）时无法标准化，定义为无差异，避免除零
        return 0.0
    return float((x_a.mean() - x_b.mean()) / pooled)


def inv_cohens_h(h, p0):
    """由 Cohen's h 与基线 p0 反解 p1。h = 2asin(sqrt(p1)) - 2asin(sqrt(p0))。

    【输入】h —— 效应量 Cohen's h；p0 —— 基线转化率。
    【输出】对应的 p1。
    【统计原理】这是 Cohen's h 定义的【逆变换】：
        h = 2·arcsin(√p1) − 2·arcsin(√p0)
      ⇒ arcsin(√p1) = (h + 2·arcsin(√p0)) / 2
      ⇒ p1 = sin( (h + 2·arcsin(√p0)) / 2 )²
      arcsin 是方差稳定化变换：变换后方差 ≈ 1/n，与 p 本身无关，
      这样效应量才能脱离基线率、跨实验比较（比例类指标用 h 而非 Cohen's d
      的根本原因）。
    """
    phi = (h + 2 * math.asin(math.sqrt(p0))) / 2.0
    return math.sin(phi) ** 2


def calibrate_obf_constant(k_looks, alpha, n_paths=200_000, seed=7):
    """蒙特卡洛标定 O'Brien-Fleming 常数 C。

    【输入】
      k_looks —— 计划查看次数 K（本项目 7 天，每天偷看一次）
      alpha   —— 目标【整体】（family-wise）一类错误上限（0.05）
      n_paths —— 模拟的布朗运动路径数（20 万）
    【输出】常数 C，使第 k 次查看的拒绝边界为 |Z_k| ≥ C·√(K/k)。
    【统计原理】
      把整个实验看成一条标准布朗运动 B(t)，t∈[0,1] 是"信息分数"。
      第 k 次查看对应时刻 t_k = k/K，此时累计 Z 统计量 Z_k = B(t_k)/√t_k。
      因为 √t_k · |Z_k| = |B(t_k)|，所以"任一次越界"等价于
          max_k |B(t_k)| ≥ C
      于是 C 就是 max_k|B(t_k)| 的 (1-α) 分位数——用蒙特卡洛直接求出来即可，
      不必推导解析解。
    【为什么 C 随 K 增大而增大】K=1 时 C=1.96（就是普通 Z 检验）；看得越频繁，
      "历史最大值"越容易触碰边界，门槛就必须越高（K=7 → 2.0605，
      连续监控极限约 2.24）。这正是 α 消耗函数要为"偷看机会"买单的体现。
    【实现细节】增量取 N(0, 1/K)，故 B(1) = Σ 增量 ~ N(0,1)，cumsum 即路径。
    """
    _rng = np.random.default_rng(seed)
    inc = _rng.normal(0.0, 1.0 / math.sqrt(k_looks), size=(n_paths, k_looks))
    brow = np.cumsum(inc, axis=1)          # 沿时间轴累加 → 布朗运动路径
    max_abs = np.abs(brow).max(axis=1)     # 每条路径的历史最大偏离
    return float(np.quantile(max_abs, 1 - alpha))


def dump_json(obj, path):
    """把（已 to_python 清洗过的）对象写盘。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_python(obj), f, ensure_ascii=False, indent=2)


# ============================================================
# 中间文件读写（各「大流程步骤」之间用 parquet 交接）
# ============================================================
def save_prepared(users, panel, df, ip_attr, meta):
    """把 ③ 数据接入与准备 的产物存盘，供 04~08 读取。

    【输入】
      users   —— 用户级 (ip, app) 面板，本项目的主分析单元（296,709 行）
      panel   —— 用户-小时级 (ip, app, hour)，DID / 事件研究用
      df      —— panel 加上 os/device/channel 属性（下钻用）
      ip_attr —— ip 级众数属性（④ 分桶质量校验用）
      meta    —— dict，含 window_start / window_end / window_hours /
                 new_users_total / new_users_per_hour / new_users_per_day_est / source
    【输出】无返回值；落盘 4 个 parquet + 1 个 json 到 data/prepared/。
    【架构意义】本项目【刻意用硬盘文件做步骤间交接】，而不是在内存里传 DataFrame。
      好处：任一步骤可独立重跑、可断点续跑、可单独调试；坏处是多了 IO。
      这就是"平铺大步骤管道"的实现方式（见 运行全流程.py）。
    【为什么用 parquet】列式压缩，比 CSV 小 5~10 倍、快数倍，且保留 dtype
      （CSV 会把 int 读成 float、把分类读成 object，容易静默改变口径）。
    """
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    users.to_parquet(PREPARED_DIR / "用户级面板.parquet", index=False)
    panel.to_parquet(PREPARED_DIR / "用户小时面板.parquet", index=False)
    df.to_parquet(PREPARED_DIR / "用户小时属性面板.parquet", index=False)
    ip_attr.to_parquet(PREPARED_DIR / "IP众数属性.parquet", index=False)
    dump_json(meta, PREPARED_DIR / "数据元信息.json")
    print(f"已保存中间文件到 {PREPARED_DIR}")


class Prepared:
    """轻量数据容器（只装 DataFrame + 窗口信息，不是框架）。load_prepared() 返回。"""

    def __init__(self, users, panel, df, ip_attr, meta):
        self.users = users
        self.panel = panel
        self.df = df
        self.ip_attr = ip_attr
        self.window_start = meta.get("window_start")
        self.window_end = meta.get("window_end")
        self.window_hours = meta.get("window_hours")
        self.new_users_total = meta.get("new_users_total")
        self.new_users_per_hour = meta.get("new_users_per_hour")
        self.new_users_per_day_est = meta.get("new_users_per_day_est")
        self.source = meta.get("source", "full_csv")


_PREPARED_FILES = ("用户级面板.parquet", "用户小时面板.parquet",
                   "用户小时属性面板.parquet", "IP众数属性.parquet", "数据元信息.json")


def load_prepared():
    """读取 ③ 产出的中间文件，返回 Prepared 容器。

    【健壮性】缺文件时给出"请先跑 ③"的明确提示，而不是把 pandas 的
    FileNotFoundError 堆栈直接抛给用户——让首次 clone 的人知道该跑哪一步。
    """
    missing = [n for n in _PREPARED_FILES if not (PREPARED_DIR / n).exists()]
    if missing:
        raise SystemExit(
            f"⚠️ 缺少中间文件 {missing}（目录：{PREPARED_DIR}）\n"
            f"   请先运行 ③ 数据接入与准备：python scripts/03_数据接入与准备.py"
        )
    users = pd.read_parquet(PREPARED_DIR / "用户级面板.parquet")
    panel = pd.read_parquet(PREPARED_DIR / "用户小时面板.parquet")
    df = pd.read_parquet(PREPARED_DIR / "用户小时属性面板.parquet")
    ip_attr = pd.read_parquet(PREPARED_DIR / "IP众数属性.parquet")
    with open(PREPARED_DIR / "数据元信息.json", encoding="utf-8") as f:
        meta = json.load(f)
    return Prepared(users, panel, df, ip_attr, meta)


# ============================================================
# 各步骤共用的"起手式"与安全取值
# ============================================================
def load_arms():
    """读 ③ 的中间文件并按 app 切出对照/实验两组（④~⑧ 各分析步骤的统一入口）。

    【为什么要有这个函数】④/⑤/⑥/⑦/⑧ 每一步开头原本都要重复同一组起手式：
        prepared = load_prepared(); users = prepared.users
        ctrl = users[users.app == CONTROL_APP]; trt = users[users.app == TREATMENT_APP]
    这属于"接口契约级"重复：一旦分析单元口径要调整（例如从 (ip,app) 改成 ip），
    漏改任何一步都会造成【静默的口径漂移】。收敛到一处，改一次即全局一致。

    【返回】(prepared, ctrl, trt)。两组对应的 app 号取 C.CONTROL_APP / C.TREATMENT_APP
    （仅在需要打印标签时使用）；需要完整 users 表时用 prepared.users。

    ⚠️ 本函数按 app 切分，是为了"对照组就是 app12"一眼可读；⑦ 因果推断为了向量化
    使用的是 ③ 预生成的 treat / Y 列（语义完全等价，赋值处见 03）。两条路径口径
    相同、只是写法不同，这是有意的可读性取舍，不是二义性。
    """
    prepared = load_prepared()
    users = prepared.users
    ctrl = users[users.app == CONTROL_APP].reset_index(drop=True)
    trt = users[users.app == TREATMENT_APP].reset_index(drop=True)
    return prepared, ctrl, trt


def dig(d, *keys, default=None):
    """安全嵌套取值：链路上任意一层不是 dict 或值为 None 时返回 default，绝不抛异常。

    【为什么需要】本项目的 JSON 由 to_python 生成，而 to_python 会把 NaN/Inf 统一
    转成 None 写盘（这是产出合法 JSON 的既定约定）。因此"字段存在但值为 null"是会
    真实出现的情况；朴素的 `d.get("a", {}).get("b")` 在 `"a": null` 时会抛
    AttributeError。汇总报告.py 内部已有一份等价实现，此处下沉到公共层供各步骤复用。
    """
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur
