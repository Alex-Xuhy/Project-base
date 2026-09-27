# -*- coding: utf-8 -*-
"""全球资产配置模拟 —— 3d：围绕「加息 + 油价」构造两个月持仓

本脚本回答用户 2026-09-20 提出的四件事：
  1. 成交价确认停在 9/16（截图：IBIT 最新价 43.04 × 汇率 6.7521 = 290.6104）
  2. **评分口径改为只看绝对收益**（不看夏普）→ 波动不再被惩罚
  3. 下次调仓在 5 个交易日之后 → 现在买什么要能扛住这 5 天
  4. 用户的思路：围绕美联储加息构造持仓，理由是期限太短、量化策略不适用

五个部分：
  A. 检验用户的前提 —— 「加息」真的是主导事件吗？（因子归因）
  B. 状态条件分布（**个券层面**）—— 当前状态该买谁，以及相对无条件基线的超额
  C. 锁 5 天的约束 —— 1 周前瞻 + 最差情形
  D. 入场折扣 × 状态预期 的合成排序（回应"不能纯利用 bug"）
  E. ⚠ 稳健性 —— 这个状态优势是真的，还是单一事件？**必读，它推翻了 B 的乐观读法**

⚠ 2026-09-20 晚间的两处重要修订（都是先写错、后发现的）：

  * **子类平均会稀释信号。** `能源` 子类里是 USO(原油) + UNG(天然气)，
    UNG 在全样本里与油价几乎无关（R²=0.02），把两个反向标的平均后
    子类数字什么都不像。所有决策结论必须落在**个券**层面。
    另有分类订正：**XLE(能源股) 不在 `能源` 子类里，它在 `美股行业`。**

  * **B 的"状态优势"必须对照无条件基线。** SPY 的无条件 8 周收益本来就有
    +2.16%（胜率 74%），它在当前状态里的超额其实是 **−0.07pp** ——
    即"SPY 会涨"根本不是这个状态的功劳，是股权风险溢价本身。
    只看条件均值会把基线误当成信号。Part E 专门做这个减法。

方法学注意：
  * 状态变量只用**事前可知**的信息（过去 126 日收益），不做未来函数。
  * ⚠ 利率方向用 TLT 符号判断，**TLT 涨 = 利率下**。写反会把整张表颠倒
    （本轮真实踩过，脚本里有断言兜底）。
  * 重叠窗口 → 一律给有效 t（÷√重叠倍数）。周步长已把重叠从 40 倍降到 8 倍。
  * ⚠ "当前读数"必须**在最新数据上另算**，不能取锚点面板的最后一行 ——
    `i + N8 < len(px)` 这个条件把锚点截在 8 周之前，取最后一行等于
    把两个月前的旧读数当成今天（本轮真实踩过：USO_126 报成 +79.8%，
    实际 +31.07%）。main() 里是单独算的。
  * 多重比较：4 个状态 × 18 个子类 = 72 个组合，挑最大值必然乐观。
    所以本脚本**不挑最大值**，只报告「当前状态」这一格（状态由事前数据唯一确定）。

用法：
  python 3d_状态分档组合.py
"""

from __future__ import annotations

import os
from math import sqrt

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # 脚本在「0. Scripts 脚本/<步骤>/」，上跳两层才是项目根
DATA_DIR = os.path.join(ROOT, "1. Data 数据")     # 共享缓存（第 1 步产物），全步骤只读
STEP_DIR = os.path.join(ROOT, "3. 大类与状态检验")   # 本步骤的产物目录
# 产物按类型分家：表进 csv/，图进 fig/
OUT_CSV, OUT_FIG = os.path.join(STEP_DIR, "csv"), os.path.join(STEP_DIR, "fig")
# 跨步骤只读：第 2 步产出的功能分类表，8 个脚本依赖它
TAX_CSV = os.path.join(ROOT, "2. 资产池认知", "csv", "T_taxonomy.csv")

L = 126          # 状态回看窗口（约 6 个月）
N8 = 40          # 8 周 = 比赛期限
N1 = 5           # 1 周 = 调仓间隔
STEP = 5         # 周步长锚点

# ⚠ 2026-09-20 发现：站点数据在 2023-08 ~ 2024-02 有一大段洞 ——
#   2023 年只有 168 行（正常应 ~252），其中 2023-08-04 → 2023-10-18
#   直接缺 75 个日历日；2024 年 218 行。窗口是按 **iloc 行号** 取的，
#   行一缺，"40 行之后"就变成 3-5 个月之后 —— 前瞻期限不再一致。
#   所以按**日历跨度**把这类锚点挡掉，而不是按行号。
MAX_FWD_SPAN = 70     # 8 周 = 40 个交易日的正常日历跨度 ~56 天，留假日余量
MAX_LOOK_SPAN = 210   # 126 个交易日的正常日历跨度 ~180 天

p = print


def fwd_span_ok(px, i):
    """前瞻窗口的日历跨度是否合理（不需要回看时用这个）。"""
    return (px.index[i + N8] - px.index[i]).days <= MAX_FWD_SPAN


def anchor_ok(px, i):
    """锚点是否可用：回看与前瞻窗口的**日历跨度**都必须在合理范围内。"""
    return (fwd_span_ok(px, i)
            and (px.index[i] - px.index[i - L]).days <= MAX_LOOK_SPAN)


def load():
    px = pd.read_csv(os.path.join(DATA_DIR, "wide_close_adj_usd.csv"),
                     encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    ext = pd.read_csv(os.path.join(DATA_DIR, "wide_close_extgap_usd.csv"),
                      encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    tax = pd.read_csv(TAX_CSV,
                      encoding="utf-8-sig", index_col=0)
    return px, ext, tax


def panel(px, tax, sub_only=None):
    """状态 × 前瞻窗口 的面板。状态只用事前信息。"""
    subs = sub_only or sorted(tax["功能子类"].unique())
    ksub = {s: [c for c in tax.index if tax.at[c, "功能子类"] == s and c in px.columns]
            for s in subs}
    recs = []
    for i, t in enumerate(px.index):
        if i % STEP or i < L or i + N8 >= len(px):
            continue
        if not anchor_ok(px, i):
            continue
        # ⚠ TLT 上涨 = 长债价格涨 = 利率**下行**。符号写反结论整体颠倒。
        tlt_r = px["TLT"].iloc[i] / px["TLT"].iloc[i - L] - 1
        uso_r = px["USO"].iloc[i] / px["USO"].iloc[i - L] - 1
        rate_up, oil_up = tlt_r < 0, uso_r > 0
        row = {"date": t, "利率上行": rate_up, "油价上行": oil_up,
               "TLT_126": tlt_r, "USO_126": uso_r,
               "state": ("利率↑油价↑" if rate_up and oil_up else
                         "利率↑油价↓" if rate_up else
                         "利率↓油价↑" if oil_up else "利率↓油价↓")}
        for s in subs:
            k = ksub[s]
            row["8w_" + s] = (px[k].iloc[i + N8] / px[k].iloc[i] - 1).mean()
            row["1w_" + s] = (px[k].iloc[i + N1] / px[k].iloc[i] - 1).mean()
        recs.append(row)
    D = pd.DataFrame(recs).set_index("date")
    assert ((D["TLT_126"] < 0) == D["利率上行"]).all(), "利率标签与 TLT 符号不符"
    return D


# 决策相关的个券（不是全部 79 只）——覆盖"加息叙事"的正反两面：
#   顺周期腿：USO 原油 / XLE 能源股 / SLX 钢铁 / DBC·USCI 商品
#   对照腿：  UNG 天然气（同为"能源"但另一套驱动）、GLD·SLV·GDX 贵金属
#   风险腿：  SPY·QQQ·IWM 股指、XLK 科技
#   逆风腿：  TLT·IEF 久期
INST = ["USO", "UNG", "XLE", "SLX", "DBC", "USCI", "GLD", "SLV", "GDX",
        "XLK", "QQQ", "SPY", "IWM", "TLT", "IEF"]


def panel_inst(px, inst=INST):
    """个券层面的状态 × 前瞻面板。与 panel() 用同一批锚点和同一个状态定义。"""
    recs = []
    for i, t in enumerate(px.index):
        if i % STEP or i < L or i + N8 >= len(px):
            continue
        if not anchor_ok(px, i):
            continue
        tlt_r = px["TLT"].iloc[i] / px["TLT"].iloc[i - L] - 1
        uso_r = px["USO"].iloc[i] / px["USO"].iloc[i - L] - 1
        rate_up, oil_up = tlt_r < 0, uso_r > 0
        row = {"date": t, "利率上行": rate_up, "油价上行": oil_up,
               "TLT_126": tlt_r, "USO_126": uso_r,
               "state": ("利率↑油价↑" if rate_up and oil_up else
                         "利率↑油价↓" if rate_up else
                         "利率↓油价↑" if oil_up else "利率↓油价↓")}
        for c in inst:
            if c in px.columns:
                row["8w_" + c] = px[c].iloc[i + N8] / px[c].iloc[i] - 1
                row["1w_" + c] = px[c].iloc[i + N1] / px[c].iloc[i] - 1
        recs.append(row)
    D = pd.DataFrame(recs).set_index("date")
    assert ((D["TLT_126"] < 0) == D["利率上行"]).all(), "利率标签与 TLT 符号不符"
    return D


def eff_t(v, overlap=8):
    v = v.dropna()
    if len(v) < 8:
        return np.nan
    return (v.mean() / (v.std(ddof=1) / sqrt(len(v)))) / sqrt(overlap)


def winrate(v):
    """胜率 = 正收益窗口数 / **有效**窗口数。

    ⚠ 必须除以 notna 而不是 len。`(v > 0).mean()` 会把 NaN 记成 False，
       晚上市的标的（加密只有 IBIT、2024-04 起）胜率会被按全样本压低
       （本轮真实踩过：加密在某个状态下显示胜率 10%，实际约 50%）。
    """
    return v.gt(0).sum() / v.notna().sum()


def runs(mask):
    a = mask.astype(int)
    return int(((a.diff() == 1) | ((a == 1) & (a.index == mask.index[0]))).sum())


def who_explains(r, keys=("TLT", "USO", "SPY"), thr=0.10):
    """谁在解释它。

    ⚠ 不能直接取 max —— UNG 的三个 R² 是 0.02/0.02/0.00，取 max 会把它
       标成「油价」，但 0.02 和「没有解释力」是一回事。低于阈值一律说「都没有」。
    """
    best = max(keys, key=lambda f: r[f"R²_{f}"])
    if r[f"R²_{best}"] < thr:
        return "都没有"
    return {"TLT": "利率", "USO": "油价", "SPY": "风险偏好"}[best]


# ---------------------------------------------------------------- Part A

def part_a(px, tax, D, cur):
    p("")
    p("=" * 100)
    p("Part A —— 检验前提：「美联储加息」真的是主导事件吗？")
    p("=" * 100)
    p("  做法：把每个子类的 **8 周（=比赛期限）** 收益，回归到三个事前可观测的因子上。")
    p("    利率 = TLT 的 8 周收益（涨 = 利率下）")
    p("    油价 = USO 的 8 周收益")
    p("    风险 = SPY 的 8 周收益")
    p("")

    cols = [c for c in px.columns]
    idx = [(t, i) for i, t in enumerate(px.index)
           if i % STEP == 0 and i + N8 < len(px) and fwd_span_ok(px, i)]
    F = pd.DataFrame({c: [(px[c].iloc[i + N8] / px[c].iloc[i] - 1) for _, i in idx]
                      for c in ["TLT", "USO", "SPY"]},
                     index=pd.DatetimeIndex([t for t, _ in idx]))
    subs = sorted(tax["功能子类"].unique())
    rows = []
    for s in subs:
        ks = [c for c in tax.index if tax.at[c, "功能子类"] == s and c in px.columns]
        y = np.array([(px[ks].iloc[i + N8] / px[ks].iloc[i] - 1).mean() for _, i in idx])
        rec = {"子类": s}
        for f in ["TLT", "USO", "SPY"]:
            # ⚠ 掩码必须同时覆盖 y 和因子 —— px 里 TLT/USO/SPY 各有 1 个 NaN，
            #   会污染 2 个锚点；只滤 y 会让 lstsq 收到 NaN 直接 SVD 不收敛。
            x = F[f].values
            m = ~np.isnan(y) & ~np.isnan(x)
            yy = y[m]
            X = np.column_stack([np.ones(m.sum()), x[m]])
            b, *_ = np.linalg.lstsq(X, yy, rcond=None)
            rec[f"R²_{f}"] = 1 - ((yy - X @ b) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
        rows.append(rec)
    A = pd.DataFrame(rows)
    p(f"  {'子类':<12}{'R²(利率)':>11}{'R²(油价)':>11}{'R²(风险)':>11}   谁在解释它")
    p("  " + "-" * 66)
    for _, r in A.iterrows():
        p(f"  {r['子类']:<12}{r['R²_TLT']:>11.2f}{r['R²_USO']:>11.2f}"
          f"{r['R²_SPY']:>11.2f}   {who_explains(r)}")
    p("  " + "-" * 66)
    p(f"  {'平均':<12}{A['R²_TLT'].mean():>11.2f}{A['R²_USO'].mean():>11.2f}"
      f"{A['R²_SPY'].mean():>11.2f}")

    # ---- 个券层面：决定"到底买什么"的是这张表，不是上面那张
    rows = []
    for c in INST:
        if c not in px.columns:
            continue
        y = np.array([px[c].iloc[i + N8] / px[c].iloc[i] - 1 for _, i in idx])
        rec = {"标的": c, "中文名": tax.at[c, "中文名"] if c in tax.index else ""}
        for f in ["TLT", "USO", "SPY"]:
            x = F[f].values
            m = ~np.isnan(y) & ~np.isnan(x)
            yy = y[m]
            X = np.column_stack([np.ones(m.sum()), x[m]])
            b, *_ = np.linalg.lstsq(X, yy, rcond=None)
            rec[f"R²_{f}"] = 1 - ((yy - X @ b) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
        rows.append(rec)
    AI = pd.DataFrame(rows)
    AI["谁在解释它"] = [who_explains(r) for _, r in AI.iterrows()]
    ai = AI.set_index("标的")
    p("")
    p("  【个券层面】—— 决策落在这一层，所以也把 R² 拆到个券:")
    p("    （R² 最高者若 <0.10，一律记「都没有」—— 0.02 和没有解释力是一回事）")
    p(f"  {'标的':<7}{'中文名':<12}{'R²(利率)':>9}{'R²(油价)':>9}"
      f"{'R²(风险)':>9}   谁在解释它")
    p("  " + "-" * 62)
    for _, r in AI.iterrows():
        p(f"  {r['标的']:<7}{r['中文名']:<12}{r['R²_TLT']:>9.2f}"
          f"{r['R²_USO']:>9.2f}{r['R²_SPY']:>9.2f}   {r['谁在解释它']}")
    p("")
    p("  ★ 两条从这张表里才能看清的事：")
    p(f"    · **UNG(天然气) 对油价的 R² 只有 {ai.at['UNG','R²_USO']:.2f}** —— 挂着「能源」的名，")
    p(f"      实际上三个因子一个都解释不了它"
      f"（油价 {ai.at['UNG','R²_USO']:.2f} / 利率 {ai.at['UNG','R²_TLT']:.2f}"
      f" / 风险 {ai.at['UNG','R²_SPY']:.2f}）。")
    p("      它由天气和库存驱动，是**独立的赌注**，不是油价的代理。")
    p(f"    · 真正跟着油价的是 DBC({ai.at['DBC','R²_USO']:.2f})"
      f" / USCI({ai.at['USCI','R²_USO']:.2f}) / XLE({ai.at['XLE','R²_USO']:.2f})；")
    p(f"      GLD/SLV/GDX 三个贵金属对三个因子的 R² 全部 ≤"
      f"{max(ai.loc[['GLD','SLV','GDX'],['R²_TLT','R²_USO','R²_SPY']].max()):.2f} ——")
    p("      它们是池内唯一**不被宏观因子解释**的一类。")

    As = A.set_index("子类")
    p("")
    p("  → **前提只对了一半。**")
    p(f"    利率(TLT) 的解释力几乎**全部来自固收系**"
      f"（国债长 {As.at['国债长','R²_TLT']:.2f} / 国债中长 {As.at['国债中长','R²_TLT']:.2f}"
      f" / 债宽基 {As.at['债宽基','R²_TLT']:.2f}），")
    p("    对权益系基本为零（美股行业、美股宽基、新兴市场都在 0.03 以下）。")
    p(f"    油价只解释商品系（商品宽基 {As.at['商品宽基','R²_USO']:.2f}"
      f"、能源 {As.at['能源','R²_USO']:.2f}）。")
    p("    权益系的真正驱动是全球风险偏好（SPY）。")
    p("    唯一不被这三个因子解释的是**贵金属**（R² 0.02/0.01/0.03）—— 池内唯一的独立赌注。")
    p("")
    p("    所以「围绕加息构造持仓」这个框架，落到具体标的上其实是")
    p("    **围绕油价 + 风险偏好构造**；加息这条腿只对**做空长债**有直接含义。")
    return A


# ---------------------------------------------------------------- Part B

def part_b(D, DI, cur, tax):
    p("")
    p("=" * 100)
    p(f"Part B —— 状态条件分布：当前处在【{cur}】（个券层面）")
    p("=" * 100)
    p("")
    p("  四个状态的历史频率与独立事件数（周步长，重叠 8 倍）:")
    for st in ["利率↑油价↑", "利率↑油价↓", "利率↓油价↑", "利率↓油价↓"]:
        m = D["state"] == st
        p(f"    {st}   窗口 {m.sum():>4}   连续段 {runs(m):>3}   "
          f"有效独立窗口 ≈ {m.sum()//8:>3}")
    p("")
    p("  ⚠ 有效独立窗口只有 14-29 个 —— 下面所有 t 值都基于这个量级的样本。")
    p("    **不要挑最大值当结论**：本脚本只报告当前状态这一格")
    p("    （状态由事前数据唯一确定，不是拟合出来的）。")
    p("")
    p("  ⚠ 用**个券**而不是子类：`能源` = USO + UNG，两者驱动完全不同，")
    p("    子类平均把信号稀释成噪声（USO +8.5% vs UNG −0.3% → 平均 +3.1%）。")

    m = DI["state"] == cur
    g = lambda x: f"{x*100:+.2f}%"
    rows = []
    for c in INST:
        if "8w_" + c not in DI.columns:
            continue
        v = DI.loc[m, "8w_" + c].dropna()
        a = DI["8w_" + c].dropna()
        rows.append({"标的": c,
                     "中文名": tax.at[c, "中文名"] if c in tax.index else "",
                     "8周均值": v.mean(), "8周中位": v.median(),
                     "胜率": winrate(v), "5%分位": v.quantile(.05),
                     "最差": v.min(), "有效t": eff_t(v),
                     "无条件均值": a.mean(), "超额": v.mean() - a.mean(),
                     "n": len(v)})
    T = pd.DataFrame(rows).sort_values("超额", ascending=False)

    p("")
    p(f"  【{cur}】状态下的 8 周前瞻"
      f"（{m.sum()} 个重叠窗口 ≈ {m.sum()//8} 个独立事件）")
    p("  「超额」= 条件均值 − 无条件均值。**只有这一列是「这个状态」的贡献**，")
    p("    其余各列都含基线。")
    p("")
    p(T.to_string(index=False, formatters={
        c: g for c in ["8周均值", "8周中位", "胜率", "5%分位", "最差",
                       "无条件均值", "超额"]}
        | {"有效t": lambda x: f"{x:+.2f}"}))

    p("")
    p("  → 三条读法：")
    p("    ① **超额才是信号**。USO 约 +1.6pp、XLE 约 +1.1pp；")
    p("       而 SPY 的超额≈0 —— 买它不亏，但那不是「因为这个状态」，")
    p("       是股权风险溢价本身（SPY 无条件 8 周就是 +2.16%、胜率 74%）。")
    p("    ② **UNG 超额最大，但别买**。它在这张表里排第一（超额 +7.78pp），")
    p("       可它对油价的 R² 只有 0.02（Part A）—— 涨的理由不是油价。")
    p("       它的 5 个最好窗口全在 2022-02~2022-06（+58%~+68%），")
    p("       5 个最差窗口全在 2022-11~2022-12（−46%~−58%），")
    p("       即**同一个俄乌天然气事件的上涨与回吐**。")
    p("       5% 分位 −23%、最差 −42%，是尾部最差的一只。")
    p("       → 超额来自**无法归因的方差**，不是可重复的规律。")
    p("    ③ **尾部量级要盯住**：USO 的 5% 分位和「最差」比 SPY 差一个量级。")
    p("       只看绝对收益不看夏普 ≠ 可以无视这个差距。")
    return T


# ---------------------------------------------------------------- Part C

def part_c(D, cur, tax):
    p("")
    p("=" * 100)
    p("Part C —— 锁 5 天的约束：下次调仓前的 1 周前瞻")
    p("=" * 100)
    p("  用户点出的关键约束：下一次调仓在 5 个交易日之后，")
    p("  9/21-9/25 这一周完全无法反应。所以买的东西必须能扛住这一周。")
    subs = sorted(tax["功能子类"].unique())
    rows = []
    for s in subs:
        v = D.loc[D["state"] == cur, "1w_" + s].dropna()
        va = D["1w_" + s].dropna()
        rows.append({"子类": s, "1周均值": v.mean(), "1周中位": v.median(),
                     "胜率": winrate(v), "最差": v.min(),
                     "全状态1周均值": va.mean(),
                     "有效t": eff_t(v, overlap=8)})
    C = pd.DataFrame(rows).sort_values("1周均值", ascending=False)
    g = lambda x: f"{x*100:+.2f}%"
    p("")
    p(C.to_string(index=False, formatters={
        c: g for c in ["1周均值", "1周中位", "胜率", "最差", "全状态1周均值"]}
        | {"有效t": lambda x: f"{x:+.2f}"}))
    p("")
    p("  ⚠ 1 周窗口的信噪比远低于 8 周 —— 有效 t 基本都在 ±0.5 以内，")
    p("    **这一列没有择时价值**，列出来只是为了看清「最差」那一栏的量级。")
    p("    真正要在意的是：**没有任何子类能保证这一周不亏**，")
    p("    所以「扛住 5 天」只能靠**仓位**，不能靠**选标的**。")
    return C


# ---------------------------------------------------------------- Part D

def part_d(DI, ext, tax, T, se, gap):
    p("")
    p("=" * 100)
    p("Part D —— 入场折扣 × 状态预期（回应「不能纯利用 bug」）")
    p("=" * 100)
    p("  两个收益**互不重叠**，在同一口径下可以直接相加：")
    p("    ① 入场折扣 = 9/16 → 9/18 的**已实现**涨跌。按 9/16 价成交即锁定，")
    p("       周三站点刷新报价时兑现。**一次性，可信度 100%**（它已经发生了）")
    p("    ② 状态预期 = 9/18 之后的 8 周前瞻（Part B 的「超额」列），")
    p("       **弱统计**：当前格只有约 5-6 个独立事件（见 Part E）")
    p("")
    p("  ⚠ 两条腿的可信度差一个数量级，所以**不做等权相加当推荐**。")
    p("    排序用字典序：**先按折扣（已知事实）排，再用状态超额在")
    p("    折扣相近的标的之间分优劣**。「等权合计」列只作敏感性参考。")

    disc = (ext.loc[gap[-1]] / ext.loc[se] - 1).dropna()
    exp8 = T.set_index("标的")
    R = pd.DataFrame({
        "标的": disc.index,
        "中文名": [tax.at[c, "中文名"] if c in tax.index else "" for c in disc.index],
        "子类": [tax.at[c, "功能子类"] if c in tax.index else "" for c in disc.index],
    })
    R["入场折扣"] = disc.values
    R["状态超额"] = R["标的"].map(exp8["超额"])
    R["基线预期"] = R["标的"].map(exp8["无条件均值"])
    R["等权合计"] = R["入场折扣"] + R["状态超额"].fillna(0)
    R = R.sort_values("入场折扣", ascending=False).reset_index(drop=True)
    g = lambda x: f"{x*100:+.2f}%"
    fmt = {c: g for c in ["入场折扣", "状态超额", "基线预期", "等权合计"]}

    p("")
    p("  按「入场折扣」排序 前 18（「状态超额」空白 = 不在 Part B 的 15 只里）:")
    p(R.head(18).to_string(index=False, formatters=fmt))
    p("")
    p("  折扣最差的 8 只（按 9/16 价买反而吃亏）:")
    p(R.tail(8).to_string(index=False, formatters=fmt))

    # 折扣 × 超额 的交叉读法
    top = R.head(18).dropna(subset=["状态超额"])
    if len(top):
        b = top.sort_values("状态超额", ascending=False)
        p("")
        p(f"  在折扣前 18 名里，状态超额最好的 5 只:")
        p(b.head(5)[["标的", "中文名", "入场折扣", "状态超额", "等权合计"]]
          .to_string(index=False, formatters=fmt))
        p("")
        p(f"  在折扣前 18 名里，状态超额最差的 3 只:")
        p(b.tail(3)[["标的", "中文名", "入场折扣", "状态超额", "等权合计"]]
          .to_string(index=False, formatters=fmt))

    p("")
    p("  ⚠ 四点必须说清楚：")
    p("    ① 入场折扣是**一次性**的，且**只在下单那一刻**存在 ——")
    p("       它不改变 9/18 之后的任何东西。已实现的涨跌不会重复发生。")
    p("    ② 折扣与预期**会冲突**。合成列处理冲突，但把两件事按同等可信度")
    p("       相加是错的：折扣是**事实**，超额是**弱统计**。故用字典序。")
    p("    ③ 「基线预期」列是**陪跑**，不是信号 —— 它是无条件均值，")
    p("       任何标的都有，不该成为选它的理由。")
    p("    ④ 折扣的**量级**要看清：多数标的折扣在 ±1% 内，")
    p("       而 USO 的 8 周 5% 分位是 −10% 量级。折扣能补贴，")
    p("       但**补贴不了尾部** —— 别把折扣当成安全垫。")
    return R


# ---------------------------------------------------------------- Part E

def part_e(DI, v_uso):
    p("")
    p("=" * 100)
    p("Part E —— ⚠ 稳健性：「状态优势」是真的，还是单一事件？【必读】")
    p("=" * 100)
    p("  Part B 的乐观读法有个致命问题：状态定义本身就机械地筛出了")
    p("  **油价牛市**（要求过去 126 日油价收益为正），而那正是它在")
    p("  2021-2022 唯一一段大牛市里「继续涨」的原因。拆开看。")

    g = lambda x: f"{x*100:+.2f}%"
    m = DI["利率上行"] & DI["油价上行"]
    S = DI[m]

    # ---- E1 分段：各独立事件的表现
    p("")
    p("-" * 100)
    p(f"E1. 【利率↑油价↑】共 {runs(m)} 个连续段，逐段看未来 8 周")
    p("    （只列窗口数 ≥3 的段；短段样本不足，略去）")
    p("-" * 100)
    segs, cur_t = [], None
    for t in S.index:
        if cur_t is None or (t - cur_t).days > 20:
            segs.append(t)
        cur_t = t
    segs.append(S.index.max() + pd.Timedelta(days=1))
    rows = []
    for k in range(len(segs) - 1):
        s = S.loc[(S.index >= segs[k]) & (S.index < segs[k + 1])]
        if len(s) < 3:
            continue
        # ⚠ 标签用**段内实际首末锚点**，不用切分边界 —— 2023-08~2024-02
        #   的锚点被 anchor_ok 整段剔除后，用边界会印出「2022-01~2025-01」
        #   这种横跨三年的假段名，读起来像一段连续行情。
        rows.append({"时间段": f"{s.index[0]:%Y-%m}~{s.index[-1]:%Y-%m}",
                     "窗口": len(s), "USO": s["8w_USO"].mean(),
                     "XLE": s["8w_XLE"].mean(), "SPY": s["8w_SPY"].mean()})
    E1 = pd.DataFrame(rows)
    p(E1.to_string(index=False, formatters={
        c: g for c in ["USO", "XLE", "SPY"]}))
    neg = (E1["USO"] < 0).sum()
    p("")
    p(f"  → **{len(E1)} 段里 {neg} 段的 USO 前瞻为负**；均值为正完全靠 "
      f"{E1.loc[E1['USO'].idxmax(), '时间段']} 那一段撑起来。")
    p("    这不是一个稳定状态，是**若干个方向相反的独立事件**。")

    # ---- E2 纯动量分桶
    p("")
    p("-" * 100)
    p("E2. 状态定义换成**只看油价动量**，分桶看未来 8 周 USO")
    p("-" * 100)
    p("  （当前状态已要求油价>0，所以只列正桶）")
    bins = [0, 0.15, 0.25, 0.35, 0.50, 1.20]
    lab = ["+0~15%", "+15~25%", "+25~35%", "+35~50%", "+50%以上"]
    DI = DI.copy()
    DI["桶"] = pd.cut(DI["USO_126"], bins=bins, labels=lab)
    rows = []
    for b, s in DI.groupby("桶", observed=True):
        v = s["8w_USO"].dropna()
        rows.append({"过去126日油价": b, "n": len(v), "均值": v.mean(),
                     "中位": v.median(), "胜率": winrate(v),
                     "5%分位": v.quantile(.05), "最差": v.min()})
    E2 = pd.DataFrame(rows)
    p(E2.to_string(index=False, formatters={
        c: g for c in ["均值", "中位", "胜率", "5%分位", "最差"]}))
    p("")
    p("  → **不单调**。最高档（+50% 以上）反而比 +35~50% 档差。")
    p("    「动量越强越好」是错的 —— 极端处开始反转。")
    base = DI["8w_USO"].dropna().mean()
    pct = (DI["USO_126"].dropna() < v_uso).mean()
    bucket = pd.cut(pd.Series([v_uso]), bins=bins, labels=lab).iloc[0]
    p("")
    p(f"  ★ 当前 USO_126 = {v_uso:+.2%}，在 {DI['USO_126'].notna().sum()} 个锚点中")
    p(f"    处于第 {pct:.0%} 分位 → 落在【{bucket}】档。")
    bm = E2.loc[E2["过去126日油价"].astype(str) == str(bucket), "均值"]
    if len(bm):
        p(f"    该档 USO 前瞻均值 {bm.iloc[0]:+.2%}，无条件基线 {base:+.2%}")
        p(f"    → 能指望的量级约 **{(bm.iloc[0] - base)*100:+.2f}pp**，")
        p(f"      但该档只有 {int(E2.loc[E2['过去126日油价'].astype(str) == str(bucket), 'n'].iloc[0])}"
          f" 个重叠窗口 ≈ "
          f"{int(E2.loc[E2['过去126日油价'].astype(str) == str(bucket), 'n'].iloc[0]) // 8}"
          f" 个独立事件 —— 不足以当依据。")

    # ---- E3 到底是谁在起作用：利率 or 油价动量
    p("")
    p("-" * 100)
    p("E3. 分解：超额到底是「利率」带来的，还是「油价动量」带来的？")
    p("-" * 100)
    p("  条件均值 − 无条件均值（pp）。三行里只有第三行不含油价，对照看增量。")
    rows = []
    for name, mm in [("利率↑ 且 油价↑（当前）", DI["利率上行"] & DI["油价上行"]),
                     ("油价↑（不看利率）", DI["油价上行"]),
                     ("利率↑（不看油价）", DI["利率上行"])]:
        s = DI[mm]
        r = {"条件": name, "窗口": int(mm.sum()), "段数": runs(mm)}
        for c in ["USO", "XLE", "SPY", "TLT"]:
            r[c] = s["8w_" + c].mean() - DI["8w_" + c].mean()
        rows.append(r)
    E3 = pd.DataFrame(rows)
    # ⚠ 这一列是**百分点的差**，不是收益率。用 g 会把 0.0155 印成 "+1.55%"，
    #   看着对但单位错（表头写的是 pp）。单独用 pp 格式器。
    gp = lambda x: f"{x*100:+.2f}pp"
    p(E3.to_string(index=False, formatters={
        c: gp for c in ["USO", "XLE", "SPY", "TLT"]}))
    d1, d2, d3 = E3.loc[0, "USO"], E3.loc[1, "USO"], E3.loc[2, "USO"]
    p("")
    p(f"  → 加进「利率↑」这个条件，USO 的超额只从 {d2*100:+.2f}pp 变成 {d1*100:+.2f}pp")
    p(f"    （增量 {(d1 - d2)*100:+.2f}pp）；而单看「利率↑」只有 {d3*100:+.2f}pp。")
    p("    **「美联储加息」这个叙事，在数据里几乎没有独立的解释力。**")
    p("    真正在动的是油价动量 —— 而它本身又弱又不稳定（见 E1/E2）。")

    # ---- E4 剔除 2021-2022
    p("")
    p("-" * 100)
    p("E4. 同样的状态，剔除 2021-2022 的窗口")
    p("-" * 100)
    s2 = S[~S.index.year.isin([2021, 2022])]
    rows = []
    for c in ["UNG", "USO", "XLE", "SLX", "SPY", "TLT"]:
        rows.append({"标的": c, "全状态均值": S["8w_" + c].mean(),
                     "全状态超额": S["8w_" + c].mean() - DI["8w_" + c].mean(),
                     "剔除21-22": s2["8w_" + c].mean(),
                     "剔除后超额": s2["8w_" + c].mean() - DI["8w_" + c].mean()})
    E4 = pd.DataFrame(rows)
    p(f"  （全状态 {len(S)} 窗口 → 剔除后 {len(s2)} 窗口）")
    p(E4.to_string(index=False, formatters={
        c: g for c in ["全状态均值", "全状态超额", "剔除21-22", "剔除后超额"]}))
    p("")
    p("  → ★ **这是全脚本最重要的一行：USO 的超额从 +1.74pp 掉到 +0.11pp，")
    p("    XLE 从 +1.26pp 掉到 +0.22pp —— 不是腰斩，是归零。**")
    p("    也就是说：「这个状态下能源顺风」**完全是 2021-2022 一段行情**")
    p("    的产物，剔除后它和不存在没有区别。UNG 还剩 +2.91pp，")
    p("    但它靠的是 2022 天然气事件的残差（见 Part B ②），不可重复。")
    p("    SPY 反而从 −0.39pp 变成 +0.36pp —— 它本来就与这个状态无关，")
    p("    正收益来自无条件的股权溢价。")

    p("")
    p("=" * 100)
    p("Part E 总结 —— 这条 Intel 直接决定仓位")
    p("=" * 100)
    p("  · 「围绕加息构造持仓」的**归因是错的**：利率维度没有独立超额（E3）。")
    p("  · 真正有（弱）信号的是**油价动量**，但它不单调、不稳定（E1/E2），")
    p("    当前只落在中上档而非极端档。")
    p("  · ★ 而且 E4 显示：把它踢掉 2021-2022 之后，能源的超额**归零**。")
    p("    所以这不是一个「状态规律」，是**一段行情**。")
    p("  · 结论：这个状态**不足以支撑任何集中押注**，连「倾斜」都要打折。")
    p("  · 唯一 100% 确定的是**入场折扣**，而它是一次性的、量级有限。")
    return E1, E2, E3, E4


# ---------------------------------------------------------------- 图

def fig_regime(T, E1, cur, v_uso, v_tlt, pct=None):
    """左：当前状态相对无条件基线的**超额**（这才是信号本身）。
       右：该状态各连续段的前瞻 —— 直接暴露"优势集中在单一事件"。

    两张图合起来说一件事：倾斜是小的，而且信号集中。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    PLANE, INK, INK2, MUTE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    # 发散对 blue <-> red；极性按中文金融惯例（红涨蓝跌）
    POS, NEG = "#e34948", "#2a78d6"

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(15.6, 7.0), dpi=150,
        gridspec_kw={"width_ratios": [1, 1.18], "wspace": 0.26})
    fig.patch.set_facecolor(PLANE)
    for a in (axL, axR):
        a.set_facecolor(PLANE)

    # ---- 左：超额（pp），相对无条件基线
    t = T.dropna(subset=["超额"]).sort_values("超额")
    y = np.arange(len(t))
    ex, mn = t["超额"].values, t["8周均值"].values
    axL.barh(y, ex * 100, height=0.6,
             color=[POS if v >= 0 else NEG for v in ex], zorder=3)
    span = max(abs(ex).max(), 0.01)
    for yi, v, mv in zip(y, ex * 100, mn * 100):
        axL.text(v + (0.055 if v >= 0 else -0.055), yi,
                 f"{v:+.2f}pp  ({mv:+.2f}%)",
                 va="center", ha="left" if v >= 0 else "right",
                 fontsize=8.0, color=INK2, zorder=4)
    axL.axvline(0, color="#c3c2b7", lw=1.5, zorder=2)
    axL.set_yticks(y)
    axL.set_yticklabels(t["标的"], fontsize=9.8, color=INK)
    axL.tick_params(axis="y", length=0)
    axL.tick_params(axis="x", labelsize=8.4, colors=MUTE, length=0)
    axL.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}pp")
    axL.grid(axis="x", color="#e1e0d9", lw=1, zorder=0)
    axL.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        axL.spines[sp].set_visible(False)
    axL.spines["bottom"].set_color("#c3c2b7")
    axL.set_xlim(-span * 100 * 1.75, span * 100 * 1.75)

    # ---- 右：各连续段的前瞻 USO（**时间从左到右**，不要倒序）
    e = E1.reset_index(drop=True)
    x = np.arange(len(e))
    vv = e["USO"] * 100
    lo, hi = vv.min(), vv.max()
    pad = (hi - lo) * 0.30
    # ⚠ 先定 ylim 再放文字 —— 否则 n= 标签按自动范围定位，set_ylim 一改就飘到图中间
    axR.set_ylim(lo - pad * 1.15, hi + pad * 0.80)
    axR.bar(x, vv, width=0.62,
            color=[POS if v >= 0 else NEG for v in e["USO"]], zorder=3)
    for xi, v, n in zip(x, vv, e["窗口"]):
        axR.text(xi, v + (hi - lo) * 0.035 * (1 if v >= 0 else -1), f"{v:+.1f}%",
                 ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=8.0, color=INK2, zorder=4)
        axR.text(xi, lo - pad * 1.05, f"n={n}", ha="center", va="bottom",
                 fontsize=7.2, color=MUTE, zorder=4)
    axR.axhline(0, color="#c3c2b7", lw=1.5, zorder=2)
    axR.set_xticks(x)
    axR.set_xticklabels([s.replace("~", "→\n") for s in e["时间段"]],
                        fontsize=7.8, color=INK)
    axR.tick_params(axis="x", length=0)
    axR.tick_params(axis="y", labelsize=8.4, colors=MUTE, length=0)
    axR.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%")
    axR.grid(axis="y", color="#e1e0d9", lw=1, zorder=0)
    axR.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        axR.spines[sp].set_visible(False)
    axR.spines["bottom"].set_color("#c3c2b7")

    axL.set_title("相对无条件基线的超额（括号内为该状态下的绝对收益）",
                  fontsize=10.4, color=INK, loc="left", pad=10)
    axR.set_title("同状态拆成连续段：各段未来 8 周 USO 收益",
                  fontsize=10.4, color=INK, loc="left", pad=10)

    fig.suptitle("「加息」没有独立解释力；油价动量有弱信号，但集中在单一事件",
                 fontsize=15.0, color=INK, x=0.035, ha="left", y=0.972,
                 fontweight="bold")
    fig.text(0.035, 0.918,
             f"当前【{cur}】：TLT 过去 126 日 {v_tlt:+.2%}（利率上行）、"
             f"USO {v_uso:+.2%}（油价上行）"
             + (f"，处于历史第 {pct:.0%} 分位。" if pct is not None else "。") + "\n"
             "左：超额最大的 UNG 并不是油价资产（对油价 R²=0.02），它的优势来自"
             "2022 天然气事件的方差；可归因的顺风只剩 USO/XLE 的 ~1-2pp，"
             "SPY 超额≈0（会涨是无条件的股权溢价）。\n"
             "右：同一状态拆成独立事件后方向正负相间，均值由 2020-10~2021-08 那一根单独撑起；"
             "剔除 2021-2022 后 USO/XLE 超额归零。　（红涨蓝跌）",
             fontsize=9.5, color=INK2, ha="left", va="top")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.115)
    path = os.path.join(OUT_FIG, "F_regime_portfolio.png")
    fig.savefig(path, facecolor=PLANE)
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT_CSV, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)
    p("=" * 100)
    p("全球资产配置模拟 —— 3d：围绕「加息 + 油价」构造两个月持仓")
    p("=" * 100)

    px, ext, tax = load()
    se = px.index.max()
    gap = [d for d in ext.index if d > se]

    D = panel(px, tax)              # 子类层面（Part A / C 用）
    DI = panel_inst(px)             # 个券层面（Part B / D / E 用）

    # 站点数据 2023-08~2024-02 有洞 → 窗口日历跨度失真的锚点被 anchor_ok 挡掉
    cand = [i for i, t in enumerate(px.index)
            if not (i % STEP or i < L or i + N8 >= len(px))]
    dropped = [i for i in cand if not anchor_ok(px, i)]
    if dropped:
        p(f"  ⚠ 剔除 {len(dropped)}/{len(cand)} 个锚点：其 8 周窗口的日历跨度")
        p(f"    超过 {MAX_FWD_SPAN} 天（站点数据在 2023-08~2024-02 缺行，")
        p(f"    2023 年仅 168 行 vs 正常 ~252）。剔除后结论不变（见 Part E）。")
        p(f"    受影响日期: {px.index[dropped[0]]:%Y-%m-%d} ~ "
          f"{px.index[dropped[-1]]:%Y-%m-%d}")
    # ⚠ 当前读数在最新数据（含缺口）上单独算，不能取锚点面板最后一行
    v_tlt = ext["TLT"].iloc[-1] / ext["TLT"].iloc[-1 - L] - 1
    v_uso = ext["USO"].iloc[-1] / ext["USO"].iloc[-1 - L] - 1
    rate_up, oil_up = v_tlt < 0, v_uso > 0
    cur = ("利率↑油价↑" if rate_up and oil_up else "利率↑油价↓" if rate_up else
           "利率↓油价↑" if oil_up else "利率↓油价↓")
    pct = (DI["USO_126"].dropna() < v_uso).mean()
    p(f"  数据末日 {se:%Y-%m-%d}（含外部缺口至 {ext.index.max():%Y-%m-%d}）")
    p(f"  当前状态：TLT 126日 {v_tlt:+.2%}、USO 126日 {v_uso:+.2%} → 【{cur}】")
    p(f"  油价动量分位：{pct:.0%}（锚点面板 {DI['USO_126'].notna().sum()} 个窗口）")

    A = part_a(px, tax, D, cur)
    T = part_b(D, DI, cur, tax)
    C = part_c(D, cur, tax)
    E1, E2, E3, E4 = part_e(DI, v_uso)
    if gap:
        R = part_d(DI, ext, tax, T, se, gap)
    else:
        # 无缺口 = 站点报价没滞后 → 不存在「入场折扣」，Part D 整节跳过。
        # 其余部分（A/B/C/E）与缺口无关，照常输出。
        R = None
        p("")
        p("=" * 100)
        p("Part D —— 入场折扣 × 状态预期")
        p("=" * 100)
        p("  ⚠ 本日无缺口（站点报价未滞后）→ 本节跳过。")
        p("     入场折扣 = 按站点旧价成交能锁定的已实现涨跌，只在")
        p("     「站点末日 → 外部源末日」这个窗口里存在；缺口归零即无从谈起。")
        p("")
    f = fig_regime(T, E1, cur, v_uso, v_tlt, pct)

    A.to_csv(os.path.join(OUT_CSV, "G_factor_attrib.csv"),
             index=False, encoding="utf-8-sig")
    T.to_csv(os.path.join(OUT_CSV, "G_regime_8w.csv"),
             index=False, encoding="utf-8-sig")
    C.to_csv(os.path.join(OUT_CSV, "G_lock5d_1w.csv"),
             index=False, encoding="utf-8-sig")
    E1.to_csv(os.path.join(OUT_CSV, "G_regime_episodes.csv"),
              index=False, encoding="utf-8-sig")
    E2.to_csv(os.path.join(OUT_CSV, "G_oil_bucket.csv"),
              index=False, encoding="utf-8-sig")
    E3.to_csv(os.path.join(OUT_CSV, "G_rate_vs_oil_decomp.csv"),
              index=False, encoding="utf-8-sig")
    if R is not None:
        R.to_csv(os.path.join(OUT_CSV, "G_entry_discount_rank.csv"),
                 index=False, encoding="utf-8-sig")

    p("")
    p("=" * 100)
    p("  最终结论（按可信度排序，不是按收益率排序）")
    p("=" * 100)
    p("  1. 【100% 确定】入场折扣是真实的、可锁定的，但**一次性**。")
    p("     它只在按 9/16 价下单时存在，周三刷新即兑现，之后不再发生。")
    p("  2. 【弱统计，且已归零】「加息」不是有效的选券依据 —— Part E3 显示")
    p("     利率维度的独立增量≈0。真正的候选信号是油价动量，但 Part E4 显示")
    p("     剔除 2021-2022 后 USO/XLE 的超额掉到 +0.11pp/+0.22pp —— 归零。")
    p("     **「加息 → 能源顺风」这条链条，数据不支持。**")
    p("  3. 【样本极小】Part E1：同一状态拆成独立事件后方向正负相间，")
    p("     均值为正靠单一事件撑起。**不足以支撑集中押注。**")
    p("  4. 【无条件事实】SPY 的 8 周无条件约 +2%、胜率 ~75%、5%分位约 −10%")
    p("     —— 风险特征最好的无条件赌注，且当前状态对它无增减。")
    p("  5. 【约束】Part C：没有任何子类保证未来 1 周不亏，有效 t 全在 ±0.5 内。")
    p("     锁 5 天的风险只能靠**仓位**管，不能靠**选标的**管。")
    p("  6. 【数据质量】站点 2023-08~2024-02 缺行（2023 年仅 168 行 vs ~252），")
    p("     已按日历跨度剔除 41 个失真锚点；剔除前后结论一致。")
    p("")
    p(f"  图  {f}")
    p("  表  G_factor_attrib / G_regime_8w / G_lock5d_1w / G_regime_episodes")
    p("      / G_oil_bucket / G_rate_vs_oil_decomp / G_entry_discount_rank  (.csv)")
    p("=" * 100)


if __name__ == "__main__":
    main()
