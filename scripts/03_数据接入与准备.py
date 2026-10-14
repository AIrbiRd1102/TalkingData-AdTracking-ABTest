"""
03_数据接入与准备.py —— 大流程步骤 ③：把原始数据变成可直接分析的面版

职责：
  · _prepare_full_csv()：全量直读模式，Spark 直接读本地原始 CSV（或 Parquet 缓存），
    按 (ip, app) 去重、聚合、构建面板
  · _prepare_csv_direct()：pandas 直读本地 CSV 子集（冒烟调试用，无需 Spark）
  · 落盘到 data/prepared/*.parquet，供 04~08 直接读取

产出的面板：
  · users   : 用户级 (ip, app) 口径
              列 = ip, app, converted, n_clicks, first_click, hour, os, device, channel, treat, Y
  · panel   : 用户-小时级 (ip, app, hour) 原始列 ip, app, hour, y, n  （DID / 事件研究用）
  · df      : 用户-小时级 (ip, app, hour) 含属性 os/device/channel
              列 = ip, app, hour, converted, n_clicks, os, device, channel
              （可选下钻面板；主流程步骤未消费，保留供探索性分析）
  · ip_attr : ip 级众数属性 os/device/channel
              （可选下钻；主流程用 users 级口径做协变量均衡，ip_attr 未消费）

口径约定（与 Spark 阶段一致）：按 (ip, app) 去重；converted = MAX(is_attributed)；
os/device/channel 取众数（确定性）。

运行：
  python 03_数据接入与准备.py                 # 默认 full_csv 源（Spark 直读全量 CSV）
  ABTEST_DATA_SOURCE=csv python 03_...       # pandas 直读本地 CSV 子集（冒烟调试）

  可选环境变量：
  ABTEST_PARQUET_DIR=<dir>   # 原始层 Parquet 缓存的存放位置（默认与 CSV 同目录）
  SAMPLE_ROWS=<n>            # 只读前 n 行冒烟调试（不走缓存）
"""
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import 公共工具 as C


# ============================================================
# (b) 数据准备：分块聚合（csv 直读 / 全量 Spark 直读共用）
# ============================================================
def _mode_attr(cnt_list, keys, col):
    """从「(ip, app, level) 计数表」列表中还原每个 (ip, app) 的众数属性。

    【输入】
      cnt_list —— 每个元素是一个 chunk 的 groupby(["ip","app",col]).size() 结果
                  （MultiIndex Series，索引为 (ip, app, level)，值为出现次数）
      keys     —— 分组键列表，本项目为 ["ip", "app"]
      col      —— 目标分类列名（os / device / channel）
    【输出】DataFrame，列 = keys + [col]，每行一个 (ip, app) 的众数取值。
    【统计/工程原理】
      分类变量【不能跨 chunk 直接聚合】——众数不满足"分块可合并"性质
      （两块各自众数的众数 ≠ 整体众数）。正确做法是先把每块的
      (ip, app, level) 出现【次数】算出来，次数是可加的，相加后再取最大
      次数对应的 level，就等价于全量众数。这是典型的
      **map（局部计数）→ reduce（计数相加）** 思路。
    【确定性】并列时用 idxmax 取索引最靠前的那个，结果不依赖行序，可复现。
    """
    # 把各 chunk 的计数表纵向拼接后按 (ip, app, level) 求和 —— 计数是可加的
    s = pd.concat(cnt_list).groupby(level=[0, 1, 2]).sum()
    s = s.reset_index(name="n")
    # idxmax：每个 (ip, app) 内取出现次数最多的 level；并列时取首个（确定性）
    idx = s.groupby(keys)["n"].idxmax()
    return s.loc[idx, keys + [col]]


def _aggregate_one_chunk(chunk):
    """把【一个 chunk】的原始行聚合成"跨块可合并"的局部结果。

    【返回】(u, p, os_cnt, dev_cnt, ch_cnt)
      u/p       —— 用户级 / 用户-小时级的局部聚合表
      os_cnt 等 —— 三个分类列的 (ip, app, level) 计数表
    【为什么局部结果要落成这三种形态】
      · converted → max、n_clicks → sum、first_click → min：三者都【可跨块合并】；
      · os/device/channel 的众数【不可跨块合并】（两块各自众数的众数 ≠ 整体众数），
        故先落成计数表——计数是可加的，跨块相加后再取最大即等价于全量众数
        （详见 _mode_attr 的 map→reduce 说明）。
    ⚠️ 为什么不用 sum(converted)：同一用户可能有多条被归因的点击，sum 得到的是
       "归因点击次数"，而我们要的是"该用户是否转化过"这一 0/1 结果，sum 会虚高。
    """
    chunk["click_time"] = pd.to_datetime(chunk["click_time"], errors="coerce")
    chunk["hour"] = chunk["click_time"].dt.hour.astype("Int64")
    chunk["is_attributed"] = chunk["is_attributed"].astype(int)

    u = chunk.groupby(["ip", "app"], sort=False).agg(
        converted=("is_attributed", "max"),
        n_clicks=("is_attributed", "size"),
        first_click=("click_time", "min"),
    ).reset_index()

    p = chunk.groupby(["ip", "app", "hour"], sort=False).agg(
        y=("is_attributed", "max"),
        n=("is_attributed", "size"),
    ).reset_index()

    os_cnt = chunk.groupby(["ip", "app", "os"], sort=False).size()
    dev_cnt = chunk.groupby(["ip", "app", "device"], sort=False).size()
    ch_cnt = chunk.groupby(["ip", "app", "channel"], sort=False).size()
    return u, p, os_cnt, dev_cnt, ch_cnt


def _aggregate_chunks(chunk_iter, source):
    """对原始行 chunk 迭代器做分块聚合（csv 直读路径使用）。

    【输入】
      chunk_iter —— 逐块产出的原始行 DataFrame 迭代器（每行 = 一次点击）
      source     —— 来源标记字符串（"csv"），仅用于日志与 meta
    【输出】_finalize 的结果（Prepared 容器）。
    【统计/工程原理】
      每块内向量化 groupby 出【局部聚合】（见 _aggregate_one_chunk），
      最后跨块【合并】。三类指标用了三种不同的合并算子，这是本函数的核心，务必分清：
        · converted   → max   ："是否曾转化"（只要有一次 is_attributed=1 即 1）
                                局部 max 再全局 max = 全局 max ✓
        · n_clicks    → sum   ：点击次数，局部 sum 再全局 sum ✓
        · first_click → min   ：最早点击时间 ✓
        · os/device/channel → 众数：不可直接合并，走 _mode_attr 的计数相加
      ⚠️ 为什么不用 sum(converted)：同一用户可能有多次被归因的点击，
         sum 会得到"归因点击次数"，而我们要的是"该用户是否转化过"这一
         0/1 结果——两者口径完全不同，sum 会虚高。
      【为什么在客户端做聚合】数据量可达千万行级，聚合放在 pandas 侧分块进行，
        压力分散、口径可控，且不依赖任何数据库服务。
    """
    C.hr(f"③ 数据准备（{source} 源，pandas 客户端分块聚合）")

    users_partial, panel_partial = [], []
    os_cnt, dev_cnt, ch_cnt = [], [], []
    for chunk in chunk_iter:
        u, p, oc, dc, cc = _aggregate_one_chunk(chunk)
        users_partial.append(u)
        panel_partial.append(p)
        os_cnt.append(oc); dev_cnt.append(dc); ch_cnt.append(cc)

    users = pd.concat(users_partial, ignore_index=True)
    users = users.groupby(["ip", "app"], sort=False).agg(
        converted=("converted", "max"),
        n_clicks=("n_clicks", "sum"),
        first_click=("first_click", "min"),
    ).reset_index()

    panel = pd.concat(panel_partial, ignore_index=True)
    panel = panel.groupby(["ip", "app", "hour"], sort=False).agg(
        y=("y", "max"), n=("n", "sum")
    ).reset_index()
    # 【防御】hour 来自 click_time.dt.hour，解析失败时为 pd.NA。
    # groupby 默认 dropna=True 会顺手丢掉 NA 键，所以这里通常已无 NA；
    # 但显式 dropna 能保证 `.astype(int)` 永不撞上
    # "Cannot convert non-finite values (NA or inf) to integer"。
    # 代价：极端情况下（某用户所有点击时间都解析失败）会少几行面板，
    # 而不是整条管线崩溃——对统计量无实质影响。
    _n_bad_hour = int(panel["hour"].isna().sum())
    if _n_bad_hour:
        print(f"  ⚠️ 面板中有 {_n_bad_hour:,} 行 hour 缺失（click_time 解析失败），已剔除")
        panel = panel.dropna(subset=["hour"]).copy()
    panel["hour"] = panel["hour"].astype(int)

    # 三个分类协变量各自按 (ip, app) 取众数（见模块级 _mode_attr 的原理说明）
    _keys = ["ip", "app"]
    os_m = _mode_attr(os_cnt, _keys, "os")
    dev_m = _mode_attr(dev_cnt, _keys, "device")
    ch_m = _mode_attr(ch_cnt, _keys, "channel")

    users = (
        users.merge(os_m, on=["ip", "app"], how="left")
        .merge(dev_m, on=["ip", "app"], how="left")
        .merge(ch_m, on=["ip", "app"], how="left")
    )
    # 【防御】first_click 为 NaT（click_time 解析失败）时 .dt.hour 得到 NA，
    # 直接 astype(int) 会抛 "Cannot convert non-finite values to integer"。
    # 统一先转 Float64（对 int32 / float64 / Int64 输入都安全）再填充哨兵。
    # 哨兵 -1 只影响 ⑦ 的 hour one-hot（多出一个 level），不影响任何比率计算。
    _hour = users["first_click"].dt.hour.astype("Float64")
    _n_bad_first = int(_hour.isna().sum())
    if _n_bad_first:
        print(f"  ⚠️ {_n_bad_first:,} 个 (ip, app) 首次点击时间缺失，hour 以 -1 哨兵填充")
    users["hour"] = _hour.fillna(-1).astype(int)
    users["treat"] = (users["app"] == B).astype(int)
    users["Y"] = users["converted"].astype(int)
    users = users.reset_index(drop=True)

    panel["y"] = panel["y"].astype(int)
    panel["n"] = panel["n"].astype(int)

    df = panel.merge(
        users[["ip", "app", "os", "device", "channel"]], on=["ip", "app"], how="left"
    ).rename(columns={"y": "converted", "n": "n_clicks"})

    print(f"  users(用户级) = {len(users):,} 行 | "
          f"panel(用户-小时) = {len(panel):,} 行 | "
          f"df(用户-小时·带属性) = {len(df):,} 行")
    return _finalize(users, panel, df, source)


def _prepare_csv_direct(csv_path=None):
    """csv 直读模式：pandas 直接读本地 CSV（train_5m.csv 子集），无需 Spark。

    复用 _aggregate_chunks 的同一套分块聚合口径，并同样只保留 app IN (12,18)，
    因此与 full_csv 全量路径完全同构。默认读 D:/AB测试数据/train_5m.csv，
    可用 ABTEST_RAW_CSV 覆盖。
    """
    csv_path = csv_path or os.environ.get("ABTEST_RAW_CSV", "D:/AB测试数据/train_5m.csv")
    cols = ["ip", "app", "device", "os", "channel", "click_time", "is_attributed"]
    A, B = C.CONTROL_APP, C.TREATMENT_APP
    raw = pd.read_csv(csv_path, chunksize=200_000, usecols=cols)

    def _keep_ab(it):
        for ch in it:
            yield ch[ch["app"].isin([A, B])]

    return _aggregate_chunks(_keep_ab(raw), "csv")


# ============================================================
# 原始层 Parquet 缓存：一次性把 CSV 固化成列式格式，供重复实验直接读
# ============================================================
def _parquet_cache_path(csv_path):
    """原始层 Parquet 缓存路径（默认与 CSV 同目录、同名不同后缀）。

    可用环境变量 ABTEST_PARQUET_DIR 覆盖存放位置。
    """
    p = Path(csv_path)
    cache_dir = Path(os.environ.get("ABTEST_PARQUET_DIR", str(p.parent)))
    return cache_dir / (p.stem + ".parquet")


def _csv_to_parquet(spark, df, pq_path):
    """把原始 CSV 的 DataFrame 写成 Parquet 缓存（一次性，可复用）。

    【为什么要这一层】CSV 每次读都要做 1.85 亿行的文本解析 + 逐行类型转换；
    固化成 Parquet 后这部分只付一次。后续"重新生成 prepared"（改聚合口径、
    换对比组）时直接读缓存，跳过 CSV 解析。体积上 `click_time` 的字典基数极低
    （4 天 × 24h × 60min 才几千个不同值）、`is_attributed` 几乎全 0，
    列式压缩率很高。

    ⚠️【本机限制 —— 失败不阻断】Spark 写文件走 Hadoop FileOutputCommitter，
    Windows 上 D:/hadoop/bin 缺 hadoop.dll 会抛 UnsatisfiedLinkError
    （README「已知环境坑」已记录）。因此本函数失败时只打警告并返回 False：
    本次仍读 CSV 跑完，功能完全不受影响；补齐 hadoop.dll 后无需改任何代码
    即自动生效。

    ⚠️【为什么写完还要读回一次】Spark 的 `overwrite` 是"先建目录再写文件"，
    写失败时会留下一个【空目录 / 损坏目录】。若只看 `exists()` 会把它误判成
    "缓存已建成"，下次运行就会误走读缓存分支然后崩溃（实测踩到过）。
    所以必须读回一行确认可用，并在失败时删掉残留目录。

    【什么时候不值得建】如果只重新生成一次 prepared，这次转换是纯增加成本；
    缓存的价值建立在"会重跑"上。
    """
    try:
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        df.write.mode("overwrite").parquet(str(pq_path))
        # 读回一行验证：确认缓存真的可用，而不是一个写失败留下的空目录
        spark.read.schema(df.schema).parquet(str(pq_path)).limit(1).collect()
        print(f"✅ 已建 Parquet 缓存：{pq_path}（后续重跑将直接读它，跳过 CSV 解析）")
        return True
    except Exception as e:
        print(f"⚠️ Parquet 缓存写入失败（不影响本次运行，本次仍读 CSV）：{type(e).__name__}")
        print(f"   {str(e)[:160]}")
        print("   提示：把与 Hadoop 版本匹配的 hadoop.dll 放入 D:/hadoop/bin 后自动生效，")
        print("        无需改代码（见 README「已知环境坑」）。")
        # 必须清理残留：空目录会让下次 exists() 误判为"已建成"，进而误走读缓存分支
        shutil.rmtree(pq_path, ignore_errors=True)
        return False


def _read_raw(spark, csv_path, schema, sample_rows=None):
    """读原始数据：优先 Parquet 缓存；没有则读 CSV 并【尝试】建一份缓存。

    【为什么读 CSV 只读一次】若缓存建成，直接返回"读缓存"的 DataFrame——
    后续 filter / 聚合全部基于缓存，CSV 全程只被读一遍（即写缓存那一次），
    不会出现"写缓存读一次 + 聚合又读一次"的重复 I/O。

    【sample_rows】是调试用的冒烟开关，故意不走缓存（数据量小，建缓存没意义）。
    """
    pq_path = _parquet_cache_path(csv_path)
    if pq_path.exists() and not sample_rows:
        try:
            print(f"读取 Parquet 缓存：{pq_path}（跳过 1.85 亿行 CSV 的解析与类型转换）")
            return spark.read.schema(schema).parquet(str(pq_path))
        except Exception as e:
            # 缓存损坏（例如上次写入中断留下的空目录）→ 清理并回退读 CSV，不阻断
            print(f"⚠️ Parquet 缓存不可读（{type(e).__name__}），已清理并回退读 CSV")
            shutil.rmtree(pq_path, ignore_errors=True)

    reader = (spark.read.option("header", "true").schema(schema)
              .option("mode", "PERMISSIVE").csv(csv_path))
    if sample_rows:
        return reader.limit(sample_rows)
    if _csv_to_parquet(spark, reader, pq_path):
        return spark.read.schema(schema).parquet(str(pq_path))
    return reader


# ============================================================
# (b) 数据准备：全量 CSV 路径（source="full_csv"，需 Spark + 本地原始 CSV）
# 逻辑等价于 spark_full_direct.py：Spark 直接读磁盘 CSV -> 聚合 -> 收集为 pandas。
# 与 csv 直读路径产出完全同构，因此后续大步骤无需任何改动即可切换。
# ============================================================
def _prepare_full_csv():
    C.hr("③ 数据准备（全量直读模式，Spark 读原始数据，跳过 MySQL 入库）")
    config = sys.modules["配置"]
    config.configure_spark_env()  # ⚠️ 必须在 import pyspark 之前
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    raw_dir = os.environ.get("ABTEST_RAW_DIR", "D:/AB测试数据")
    csv_path = os.path.join(raw_dir, os.environ.get("ABTEST_RAW_CSV", "train.csv"))
    sample = os.environ.get("SAMPLE_ROWS", "").strip()
    sample_rows = int(sample) if sample else None

    spark = (
        SparkSession.builder
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .appName("ab-test-full-direct-csv")
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "6g"))
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions",
                os.environ.get("SPARK_SHUFFLE_PARTITIONS", "200"))
        .config("spark.local.dir", os.environ.get("SPARK_LOCAL_DIR", "D:/spark_tmp"))
        # 把 $HADOOP_HOME/bin 加进 java.library.path，否则 JVM 加载不到 hadoop.dll，
        # Spark 写文件（含 Parquet 缓存）会抛 UnsatisfiedLinkError。详见 配置.py。
        .config("spark.driver.extraJavaOptions", config.spark_extra_java_options())
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        from pyspark.sql.types import (
            StructType, StructField, LongType, IntegerType, StringType,
        )
        schema = StructType([
            StructField("ip", LongType(), True),
            StructField("app", IntegerType(), True),
            StructField("device", IntegerType(), True),
            StructField("os", IntegerType(), True),
            StructField("channel", IntegerType(), True),
            StructField("click_time", StringType(), True),
            StructField("attributed_time", StringType(), True),
            StructField("is_attributed", IntegerType(), True),
        ])
        # 读原始数据：优先 Parquet 缓存（跳过 CSV 解析）；没有则读 CSV 并尝试建一份缓存
        exp = _read_raw(spark, csv_path, schema, sample_rows).filter(
            F.col("app").isin(C.CONTROL_APP, C.TREATMENT_APP))

        # 全量模式记录全局 click_time 起止，供 _finalize 推导时间窗口
        from pyspark.sql.functions import min as _Fmin, max as _Fmax
        _bounds = exp.agg(_Fmin("click_time").alias("tmin"),
                          _Fmax("click_time").alias("tmax")).collect()[0]
        t_min_raw, t_max_raw = _bounds["tmin"], _bounds["tmax"]

        # ------------------------------------------------------------------
        # 分类协变量 os / device / channel：按 (ip, app) 取【众数】
        # 【已修复的技术债】改动前这里用 F.max(col)，即取【数值最大的取值】，
        #   而 csv 路径（_mode_attr）用的是【众数】（出现次数最多的取值）。
        #   两者对同一 (ip, app) 会给出不同结果 → 三条数据路径的协变量口径不一致。
        #   F.max 的危害是【方向性】的：它系统性地偏向编号大的水平，等于给协变量
        #   注入了一层与处理无关的确定性扭曲，会污染 ④ 的均衡检验与 ⑦ 的倾向得分。
        # 【Spark 版众数怎么写】众数不具备"分块可合并"性，所以不能直接两次 agg 求
        #   max。标准做法是两步：
        #     ① 先按 (ip, app, level) 数【次数】（次数是可加的，与 _mode_attr 同思路）；
        #     ② 再用窗口函数在每个 (ip, app) 内按次数降序取第 1 行（row_number=1）。
        # 【并列怎么办 —— 必须显式定序，否则结果不可复现】
        #   pandas 的 idxmax 在并列时取【索引最靠前】的，而计数表已按 level 升序排列，
        #   即并列时取【最小的 level】。这里用 orderBy(desc(n), asc(col)) 复刻同一规则，
        #   保证 Spark 路径与 pandas 路径在并列情形下也给出同一个值。
        # ------------------------------------------------------------------
        from pyspark.sql.window import Window

        def _mode_by_unit(col):
            """返回 (ip, app, <col>_mode) 的众数表。"""
            cnt = exp.groupBy("ip", "app", col).agg(F.count("*").alias("_n"))
            w = Window.partitionBy("ip", "app").orderBy(F.desc("_n"), F.asc(col))
            return (cnt.withColumn("_rn", F.row_number().over(w))
                       .filter(F.col("_rn") == 1)
                       .select("ip", "app", F.col(col).alias(f"{col}_mode")))

        mode_os = _mode_by_unit("os")
        mode_dev = _mode_by_unit("device")
        mode_ch = _mode_by_unit("channel")

        df = (
            exp.groupBy("ip", "app", "click_time")  # click_time 原始串，按小时需 HOUR
            .agg(
                F.max("is_attributed").alias("converted"),
                F.count("*").alias("n_clicks"),
                # ⚠️ 注意：这里【不再】聚合 os/device/channel，改由下面按 (ip, app)
                # 众数表 join 进来 —— 众数是 (ip, app) 级属性，不能逐小时聚合。
            )
        )
        # 用 click_time 前 13 字符（'YYYY-MM-DD HH'）作小时桶
        df = df.withColumn("hour", F.substring("click_time", 1, 13))
        df = (df.groupBy("ip", "app", "hour")
              .agg(
                  F.max("converted").alias("converted"),
                  F.sum("n_clicks").alias("n_clicks"),
              ))
        # 把 (ip, app) 级众数 join 回用户-小时面板（左连接保证行数不变）
        df = (df.join(mode_os, on=["ip", "app"], how="left")
                .join(mode_dev, on=["ip", "app"], how="left")
                .join(mode_ch, on=["ip", "app"], how="left"))
        df = df.toPandas()
        df["hour"] = df["hour"].str[11:13].astype(int)
        df = df.rename(columns={"os_mode": "os", "device_mode": "device",
                                "channel_mode": "channel"})

        panel = df[["ip", "app", "hour", "converted", "n_clicks"]].copy()
        panel = panel.rename(columns={"converted": "y", "n_clicks": "n"})

        # 【修复】原实现写死 `users["hour"] = 0`，是全量模式下唯一的"整组特征失效"问题：
        #   07 的倾向得分模型用 `_build_X` 对 hour 做 one-hot（24 列），hour 恒为 0
        #   ⇒ 24 列退化为【只有 hour_0 一列全 1、其余 23 列全 0】的常数列，
        #   等价于 hour 这个协变量被【静默丢弃】——不报错、只是悄悄少了一个混杂维度。
        #   修复口径：与 csv 路径的 `first_click.dt.hour` 对齐，取该 (ip, app)
        #   在窗口内【最早】出现的小时桶（df 已是 (ip,app,hour) 粒度，故 min(hour) 即可）。
        #   first_click 本身仍为 NaT —— 全量路径没有保留逐行 click_time，无法还原时间戳，
        #   这是该路径已知的信息损失，不影响 hour。
        # os/device/channel 现在在每个 (ip, app) 内是【常数】（众数本来就是单元级属性），
        # 所以这里用 "max" 只是"把那个常数取出来"，不会再引入任何口径扭曲；
        # 改动前的 F.max 是在【逐小时】粒度上取最大取值，那才是错的根源。
        users = (df.groupby(["ip", "app"])
                 .agg(converted=("converted", "max"),
                      n_clicks=("n_clicks", "sum"),
                      hour=("hour", "min"),
                      os=("os", "max"), device=("device", "max"),
                      channel=("channel", "max"))
                 .reset_index())
        users["first_click"] = pd.NaT
        users["treat"] = (users["app"] == C.TREATMENT_APP).astype(int)
        users["Y"] = users["converted"].astype(int)
    finally:
        spark.stop()

    return _finalize(users, panel, df, "full_csv", window_info=(t_min_raw, t_max_raw))


# ============================================================
# 由 users / panel / df 派生 ip_attr 与窗口信息
# ============================================================
def _mode_by_ip(frame, col):
    """每个 ip 在 col 上的众数（并列时取其一），用于协变量均衡检验。

    【输入】frame —— 含 ip 与 col 两列的 DataFrame（本项目传 users）；col —— 分类列名。
    【输出】DataFrame，列 = ["ip", col]，每个 ip 一行。
    【与 _mode_attr 的区别 —— 两套口径并存，别混用】
      · _mode_attr  ：按 **(ip, app)** 取众数 → 用于 users 表的协变量
      · _mode_by_ip ：按 **ip** 取众数（跨 app 合并）→ 用于 ip_attr 表
      两者粒度不同，是【有意为之】：users 的分析单元是 (ip, app)，
      而 ip_attr 供 ④ 的分桶质量检验使用，分桶键本身就是 ip，故须是 ip 级。
    【实现】按 (ip, level) 计数 → 按计数降序排序 → drop_duplicates("ip") 保留首个
      （即计数最大的 level；并列时取排序最靠前的，稳定可复现）。
    """
    vc = frame.groupby(["ip", col]).size().reset_index(name="n")
    vc = vc.sort_values(["ip", "n"], ascending=[True, False])
    return vc.drop_duplicates("ip")[["ip", col]]


def _derive_window(users, window_info=None):
    """推导数据窗口与吞吐量元信息。

    【窗口推导的三条分支】
      ① window_info 非空 → 用 Spark 侧全局 click_time 起止（full_csv 路径：
         first_click 未逐行保留，只能由 Spark 侧全局 min/max 推导）
      ② users.first_click 有非空值 → 用首次点击的 min/max（csv 路径）
      ③ 都没有 → 退化为 NaN，不阻断流程（保证管线不因缺字段崩溃）
    ⚠️ new_users_per_day_est = 每小时新增 × 24，这是【规划吞吐量】，
      只用于 ⑥ 的序贯模拟（决定每天新增多少单元），不是效应估计。
    【返回】meta dict（不含 source，由调用方补上）。
    """
    if window_info is not None:
        t_min = pd.Timestamp(window_info[0])
        t_max = pd.Timestamp(window_info[1])
    elif users["first_click"].notna().any():
        t_min, t_max = users["first_click"].min(), users["first_click"].max()
    else:
        t_min = t_max = None

    new_users_total = len(users)
    if t_min is not None:
        window_hours = (t_max - t_min).total_seconds() / 3600.0
        new_users_per_hour = new_users_total / window_hours if window_hours else float("nan")
        new_users_per_day_est = new_users_per_hour * 24.0
    else:
        window_hours = float("nan")
        new_users_per_hour = float("nan")
        new_users_per_day_est = float("nan")
    return {
        "window_start": str(t_min) if t_min is not None else None,
        "window_end": str(t_max) if t_max is not None else None,
        "window_hours": window_hours,
        "new_users_total": new_users_total,
        "new_users_per_hour": new_users_per_hour,
        "new_users_per_day_est": new_users_per_day_est,
    }


def _finalize(users, panel, df, source, window_info=None):
    """派生 ip_attr 与时间窗口元信息，落盘并返回 Prepared 容器。

    【输入】
      users / panel / df —— 三类面板（见本文件开头 docstring）
      source             —— "csv" / "full_csv"
      window_info        —— (t_min, t_max) 二元组；仅 full_csv 路径传入
                            （该路径未逐行保留 click_time，窗口由 Spark 侧
                            全局 min/max 推导，见 _derive_window）
    【输出】C.Prepared 容器（含 meta）。
    """
    ip_attr = _mode_by_ip(users, "os")
    for _c in ["device", "channel"]:
        ip_attr = ip_attr.merge(_mode_by_ip(users, _c), on="ip", how="left")
    ip_attr = ip_attr.reset_index(drop=True)

    meta = _derive_window(users, window_info)
    meta["source"] = source
    C.save_prepared(users, panel, df, ip_attr, meta)
    return C.Prepared(users, panel, df, ip_attr, meta)


def prepare_data(source="full_csv"):
    """统一入口：根据 source 选择数据来源，构建面板并落盘。

    source:
      "full_csv"  —— Spark 直接读本地全量 CSV（默认；需 Spark + 原始 CSV）
      "csv"       —— pandas 直读本地 CSV 子集（train_5m.csv；冒烟调试用，无需 Spark）
    """
    if source == "full_csv":
        return _prepare_full_csv()
    if source == "csv":
        return _prepare_csv_direct()
    raise ValueError(f"未知 source: {source!r}（可选 full_csv / csv）")


def main():
    # 默认 full_csv：正式口径是 Spark 直读全量 CSV（首次会建 Parquet 缓存）。
    source = os.environ.get("ABTEST_DATA_SOURCE", "full_csv")
    prepared = prepare_data(source)
    if prepared is not None:
        C.hr("✅ ③ 数据接入与准备完成")
        print(f"  产物: {C.PREPARED_DIR}/{{users,panel,df,ip_attr}}.parquet + 数据元信息.json")
        print(f"  数据来源 = {prepared.source} | 用户级 {len(prepared.users):,} 行")
    return prepared


if __name__ == "__main__":
    main()
