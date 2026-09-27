#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全球资产配置模拟 —— 4a：保底仓位（核心）与折扣仓位（卫星）的划分

回答用户 2026-09-20 的两个问题：
  1. 分析折扣最大的 8 只（IBIT BLOK SLV SIL XLK GDX GLD QQQ）
  2. 「SPY/QQQ 一带」是什么意思？→ 指的是**美股宽基**这个子类（8 只）

核心发现（Part B）：**折扣最大的标的，恰恰是最不平稳的。**
  corr(折扣, 8周标准差) = +0.47
  corr(折扣, 5%分位)   = −0.37
原因很直接：两日跳 5-7% 本身就是高波动的表现。所以「折扣最大」和「收益平稳」
**在结构上是对立的** —— 不能指望同一只标的既给最大折扣又当保底。

方法学注意：
  * 全部用**无条件**窗口（8 周 + 1 周），不叠加状态条件 ——
    3d 已证伪「利率↑油价↑」状态的预测力，这里不再引入。
  * 窗口按日历跨度过滤（站点 2023-08~2024-02 缺行，见 3d）。
  * 「保底」是**选取结果分布形状**的选择，不是最大化期望收益的选择。
    本次评分只看绝对收益 → 保底仓位是用**期望收益**换**亏损概率**。
    这一点必须说清楚，别把它当成免费的改进。

用法：
  python 4a_核心卫星划分.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # 脚本在「0. Scripts 脚本/<步骤>/」，上跳两层才是项目根
DATA_DIR = os.path.join(ROOT, "1. Data 数据")     # 共享缓存（第 1 步产物），全步骤只读
STEP_DIR = os.path.join(ROOT, "4. 组合构建")   # 本步骤的产物目录
# 产物按类型分家：表进 csv/，图进 fig/
OUT_CSV, OUT_FIG = os.path.join(STEP_DIR, "csv"), os.path.join(STEP_DIR, "fig")
# 跨步骤只读：第 2 步产出的功能分类表，8 个脚本依赖它
TAX_CSV = os.path.join(ROOT, "2. 资产池认知", "csv", "T_taxonomy.csv")

N8, N1, STEP, MAXF = 40, 5, 5, 70
FOCUS = ["IBIT", "BLOK", "SLV", "SIL", "XLK", "GDX", "GLD", "QQQ"]
CORE_SUB = "美股宽基"

p = print
g = lambda x: f"{x*100:+.2f}%"
g1 = lambda x: f"{x:+.2f}"


def load():
    px = pd.read_csv(os.path.join(DATA_DIR, "wide_close_adj_usd.csv"),
                     encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    ext = pd.read_csv(os.path.join(DATA_DIR, "wide_close_extgap_usd.csv"),
                      encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    tax = pd.read_csv(TAX_CSV,
                      encoding="utf-8-sig", index_col=0)
    return px, ext, tax


def anchors(px):
    """8 周窗口锚点。按日历跨度剔除站点缺行导致的失真窗口。"""
    return [(t, i) for i, t in enumerate(px.index)
            if i % STEP == 0 and i + N8 < len(px)
            and (px.index[i + N8] - t).days <= MAXF]


def profile(px, ext, tax):
    """每只标的的 8 周 / 1 周分布 + 当前入场折扣。"""
    idx = anchors(px)
    se = px.index.max()
    gap = [d for d in ext.index if d > se]
    disc = (ext.loc[gap[-1]] / ext.loc[se] - 1).dropna()
    rows = []
    for c in tax.index:
        if c not in px.columns:
            continue
        v = np.array([px[c].iloc[i + N8] / px[c].iloc[i] - 1 for _, i in idx])
        v = v[~np.isnan(v)]
        w = np.array([px[c].iloc[i + N1] / px[c].iloc[i] - 1 for _, i in idx])
        w = w[~np.isnan(w)]
        if len(v) < 100:
            continue
        rows.append({
            "标的": c, "中文名": tax.at[c, "中文名"],
            "子类": tax.at[c, "功能子类"],
            "8周均值": v.mean(), "8周中位": np.median(v),
            "胜率": (v > 0).mean(), "标准差": v.std(ddof=1),
            "5%分位": np.percentile(v, 5), "最差": v.min(),
            "1周最差": w.min(), "1周胜率": (w > 0).mean(),
            "折扣": disc.get(c, np.nan), "n": len(v),
        })
    return pd.DataFrame(rows).set_index("标的")


# ---------------------------------------------------------------- Part A

def part_a(D, tax):
    p("")
    p("=" * 100)
    p(f"Part A —— 「SPY/QQQ 一带」= 子类【{CORE_SUB}】（{sum(tax['功能子类'] == CORE_SUB)} 只）")
    p("=" * 100)
    p("  上一轮我说「核心仓位放在 SPY/QQQ 一带」，这里的「一带」指的就是")
    p(f"  功能分类里的【{CORE_SUB}】子类 —— 美股大盘指数 ETF，不是某两只特定标的。")
    p("")
    b = [c for c in tax.index if tax.at[c, "功能子类"] == CORE_SUB and c in D.index]
    T = D.loc[b].sort_values("胜率", ascending=False)
    p(T[["中文名", "折扣", "8周均值", "胜率", "标准差", "5%分位", "最差", "1周最差"]]
      .to_string(formatters={c: g for c in ["折扣", "8周均值", "胜率", "标准差",
                                            "5%分位", "最差", "1周最差"]}))
    p("")
    p("  这 8 只覆盖：标普 500(SPY) / 纳斯达克 100(QQQ) / 纳斯达克综合(ONEQ)")
    p("  / 道指 30(DIA) / 罗素 2000 小盘(IWM,IWN) / 标普 400(DSI) / 高股息(VYM)。")
    p("  同子类内部差异很大：SPY 胜率 73.5%、IWN 只有 62.4% ——")
    p("  「美股宽基」不等于「都一样稳」，小盘那一半明显更颠。")
    return T


# ---------------------------------------------------------------- Part B

def part_b(D):
    p("")
    p("=" * 100)
    p("Part B —— ⚠ 折扣与平稳是对立的（回答「为什么不能都要」）")
    p("=" * 100)
    p(f"  corr(折扣, 8周标准差) = {D['折扣'].corr(D['标准差']):+.3f}"
      f"    corr(折扣, 5%分位) = {D['折扣'].corr(D['5%分位']):+.3f}")
    p("  原因很直接：**两天跳 5-7% 这件事本身就是高波动的表现**。")
    p("  折扣大 = 它在 9/16→9/18 动得多 = 它平时也动得多。")
    p("")
    q = pd.qcut(D["折扣"], 4, labels=["折扣最低25%", "次低", "次高", "折扣最高25%"])
    p(f"  {'折扣分组':<12}{'平均折扣':>9}{'平均标准差':>11}"
      f"{'平均5%分位':>11}{'平均胜率':>9}")
    p("  " + "-" * 54)
    for lab, s in D.groupby(q, observed=True):
        p(f"  {lab:<12}{s['折扣'].mean()*100:>+8.2f}%{s['标准差'].mean()*100:>10.2f}%"
          f"{s['5%分位'].mean()*100:>10.2f}%{s['胜率'].mean()*100:>8.0f}%")
    p("")
    p("  → 折扣最高那 25% 的标的，平均标准差 10.9%（最低组 7.3%），")
    p("    平均 5% 分位 −14.7%（最低组 −11.0%）。**这不是巧合，是同一个东西。**")
    return q


# ---------------------------------------------------------------- Part C

def part_c(D):
    p("")
    p("=" * 100)
    p("Part C —— 折扣最大的 8 只，逐个体检")
    p("=" * 100)
    T = D.loc[FOCUS].sort_values("折扣", ascending=False)
    p(T[["中文名", "子类", "折扣", "8周均值", "胜率", "标准差",
         "5%分位", "最差", "1周最差", "1周胜率"]]
      .to_string(formatters={c: g for c in ["折扣", "8周均值", "胜率", "标准差",
                                            "5%分位", "最差", "1周最差", "1周胜率"]}))
    p("")
    p("  ★ 这 8 只**恰好分成两组**，中间没有过渡：")
    p("")
    p("  【勉强能当保底】只要 2 只 —— QQQ、XLK")
    core = T[T["胜率"] >= 0.65]
    for c, r in core.iterrows():
        p(f"    {c:<5}{r['中文名']:<9} 胜率 {r['胜率']:.1%}  标准差 {r['标准差']:.2%}  "
          f"5%分位 {r['5%分位']:.2%}")
    p("      特征：胜率 ≈70%、标准差 ≈7%、5% 分位 ≈−10%。")
    p("      它们和美股宽基几乎同档，属于「本来就会买，折扣只是送的」。")
    p("")
    p("  【只能当卫星】其余 6 只 —— 折扣诱人，但尾部是另一个量级")
    sat = T[T["胜率"] < 0.65]
    for c, r in sat.iterrows():
        p(f"    {c:<5}{r['中文名']:<9} 胜率 {r['胜率']:.1%}  标准差 {r['标准差']:>6.2%}  "
          f"5%分位 {r['5%分位']:>7.2%}  一周最差 {r['1周最差']:>7.2%}")
    p("")
    p("  ⚠ 特别点出 3 只：")
    p("    · **BLOK** 折扣第 2 大(+6.80%)，但 8 周最差 **−43.3%**、胜率仅 55%。")
    p("      它跳得高只是因为它是区块链主题，波动是它的**属性**，不是机会。")
    p("    · **SLV / SIL / GDX** 三个贵金属系：胜率只有 46-50%，")
    p("      标准差 13-17%，**一周最差 −24% 到 −25%**。")
    p("      而你恰好被锁死 5 个交易日 —— 这个数就是为你现在的处境量的。")
    p("    · **GLD** 是这一组里的例外：标准差 6.34% 接近 SPY(5.42%)。")
    p("      但它胜率只有 52.4%、8 周均值 +1.03%，是**低波低收益**，")
    p("      作用是分散不是复利。别把它和 SPY 当同类。")
    p("")
    p("  ⚠ 一条通用的量级判断：**折扣大约只能覆盖 5% 尾部损失的 1/4。**")
    cov = (T["折扣"] / T["5%分位"].abs())
    for c, r in T.iterrows():
        p(f"    {c:<5} 折扣 {r['折扣']:>+6.2%}  vs  5%分位 {r['5%分位']:>+7.2%}"
          f"   → 覆盖 {cov[c]:.0%}")
    p("    所以「用折扣当安全垫」是错的 —— 它补贴不了尾部，只补贴了起点。")
    return T, core, sat


# ---------------------------------------------------------------- Part D

def part_d(D, tax):
    p("")
    p("=" * 100)
    p("Part D —— 全部 79 只里，真正稳的是哪些")
    p("=" * 100)
    p("  排序口径：胜率与「5% 分位」各占一半（保底看的是**别亏**，不是**多赚**）。")
    D = D.copy()
    D["稳"] = D["胜率"] * 2 + D["5%分位"]
    T = D.nlargest(15, "稳")
    p(T[["中文名", "子类", "折扣", "8周均值", "8周中位", "胜率",
         "标准差", "5%分位", "最差"]]
      .to_string(formatters={c: g for c in ["折扣", "8周均值", "8周中位", "胜率",
                                            "标准差", "5%分位", "最差"]}))
    p("")
    n_broad = sum(1 for c in T.index if tax.at[c, "功能子类"] == CORE_SUB)
    p(f"  → 前 15 名里 **{n_broad} 只来自【{CORE_SUB}】**，剩下 8 只是美股行业")
    p("    （XLK 科技 / XLV 医疗 / XLY 可选消费 / XLF 金融 / XLI 工业 /")
    p("     XLP 必需消费 / XLC 通信）+ IGF。这 15 只**全部是美国股权类**。")
    p("")
    p("  ⚠ 但先别下结论 —— 换一个口径，答案会翻转。子类层面看：")
    p("")
    sub = D.copy()
    sub["子类"] = [tax.at[c, "功能子类"] for c in sub.index]
    S = (sub.groupby("子类")
         .agg(只数=("胜率", "size"), 胜率=("胜率", "mean"),
              标准差=("标准差", "mean"), 五分位=("5%分位", "mean"),
              最差=("最差", "mean"), 均值=("8周均值", "mean"))
         .sort_values("五分位", ascending=False))
    S = S[S["只数"] >= 2]
    p(S.to_string(formatters={c: g for c in ["胜率", "标准差", "五分位", "最差", "均值"]}))
    p("")
    p("  ★ 尾部最浅的三类**根本不是股票**：美元(标准差 2.69%)、信用债(2.74%)、")
    p("    其他货币(3.19%)，5% 分位只有 −4.4% ~ −5.6% —— 比 SPY 的 −8.0% 浅一半。")
    p("    它们没进上面那张榜，**只是因为它们的胜率≈50%、均值≈0**，")
    p("    被我那个「胜率×2 + 5%分位」的口径扣掉了分。")
    p("")
    p("  → 所以「保底」有两个互不相同的定义，先想清楚要哪个：")
    p("")
    p("    口径甲【不亏】：把亏损概率压到最低 → 美元 / 信用债 ETF。")
    p("      标准差 2.7%、5% 分位 −4.4%，代价是 8 周期望收益 **−0.03% ~ −0.28%**。")
    p("      这是**现金等价物**，几乎保证不亏，也几乎保证不赚。")
    p("    口径乙【稳中有涨】：胜率高 + 尾部浅 → 美股宽基 / 美股行业。")
    p("      标准差 5-8%、5% 分位 −7% ~ −10%，8 周期望收益 **+1.5% ~ +3.1%**。")
    p("")
    p("    **本次只看绝对收益 → 口径甲在评分上是纯拖累**（+0% 就是没得分），")
    p("    所以下面一律按口径乙给建议。但你要知道甲存在、且是免费的。")
    p("")
    p("  另一头，垫底的是天然气/原油/白银/铀/钢铁（能源 −27.3%）。")
    p("  在 8 周这个尺度上这个池子的「平稳」基本是**资产大类排序**，")
    p("  不是选券技巧 —— 因为对比组是原油、天然气、贵金属、加密。")
    p("  （单只成类的没进上表：IBIT 加密标准差 18.7%，比能源还颠。）")
    p("")
    p("  两类保底，取舍不同：")
    p("    · **要期望收益**：SPY（胜率 73.5%、8 周 +2.02%）/ QQQ（+2.94%、71.2%）")
    p("      / ONEQ（+2.67%、72.9%）—— 代价是 5% 分位 −8% ~ −10.5%。")
    p("    · **要尾部浅**：XLP 必需消费（标准差 4.32%、5% 分位 **−5.29%**、")
    p("      最差 **−17.31%**，全池最浅）/ VYM 高股息（−6.39%）/ XLV 医疗（−7.05%）。")
    p("      代价是均值降到 +1.2% ~ +1.8%、胜率降到 62-68%。")
    p("")
    p("  ⚠ 三条必须说清的限制：")
    p("    ① 胜率 70%+ 是 **2012-2026 美股长牛**的产物，不是自然规律。")
    p("       SPY 的 8 周最差是 **−27.4%** —— 尾部一直在那里。")
    p("    ② **「保底」在「只看绝对收益」的口径下是要付费的。**")
    p("       它降低亏损概率，同时压低期望收益（SPY +2.02% vs XLK +3.13%）。")
    p("       如果评分真的纯看绝对收益、且你只关心期望值，那保底仓位是负贡献；")
    p("       只有当你在意「两个月后是亏的」这件事本身时，它才值得。")
    p("    ③ 同子类内部也要挑：IWM/IWN（罗素 2000）挂着「美股宽基」的名，")
    p("       胜率只有 62-67%、最差 −38%~−40%，**不是保底料**。")
    return T


# ---------------------------------------------------------------- 图

def fig_core(D, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    PLANE, INK, INK2, MUTE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    # 统一编码：蓝 = 美股宽基成员，橙 = 非宽基。两个面板同一个意思。
    S1, S2 = "#2a78d6", "#eb6834"

    # 标签偏移：Q 群与贵金属群太挤，逐个手放
    OFF = {"IBIT": (0, 10), "BLOK": (-34, -7), "SLV": (0, 10), "SIL": (0, 10),
           "GDX": (0, 10), "XLK": (0, 11), "GLD": (-36, -4), "QQQ": (-34, 4)}

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(15.4, 7.0), dpi=150,
        gridspec_kw={"width_ratios": [1.0, 1], "wspace": 0.10})
    fig.patch.set_facecolor(PLANE)
    for a in (axL, axR):
        a.set_facecolor(PLANE)

    # ---- 左：折扣 vs 标准差，突出结构矛盾
    other = D.drop(index=FOCUS, errors="ignore")
    axL.scatter(other["折扣"] * 100, other["标准差"] * 100, s=26,
                color="#c9c8bf", edgecolor=PLANE, linewidth=0.8, zorder=3,
                label="其余 71 只")
    for c in FOCUS:
        r = D.loc[c]
        col_c = S1 if D.at[c, "子类"] == CORE_SUB else S2
        axL.scatter(r["折扣"] * 100, r["标准差"] * 100, s=115,
                    color=col_c, edgecolor=PLANE, linewidth=1.5, zorder=5)
    hL = [plt.Line2D([], [], marker="o", ls="", ms=6, color="#c9c8bf",
                     label="其余 71 只"),
          plt.Line2D([], [], marker="o", ls="", ms=8, color=S1,
                     label="折扣组·美股宽基"),
          plt.Line2D([], [], marker="o", ls="", ms=8, color=S2,
                     label="折扣组·其他")]
    for c in FOCUS:
        r = D.loc[c]
        axL.annotate(c, (r["折扣"] * 100, r["标准差"] * 100),
                     textcoords="offset points", xytext=OFF[c],
                     ha="center", fontsize=8.4, color=INK, zorder=6)
    m = np.polyfit(D["折扣"] * 100, D["标准差"] * 100, 1)
    xs = np.linspace(D["折扣"].min() * 100, D["折扣"].max() * 100, 20)
    axL.plot(xs, np.polyval(m, xs), color=MUTE, lw=1.6, ls="--", zorder=2)
    axL.axvline(0, color="#c3c2b7", lw=1.2, zorder=1)
    axL.set_xlabel("9/16 → 9/18 入场折扣（已实现，一次性）",
                   fontsize=9.6, color=INK2)
    axL.set_ylabel("8 周收益标准差", fontsize=9.6, color=INK2)
    axL.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%")
    axL.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    axL.xaxis.set_major_locator(MultipleLocator(2))
    axL.xaxis.set_minor_locator(MultipleLocator(1))
    axL.yaxis.set_major_locator(MultipleLocator(5))
    axL.yaxis.set_minor_locator(MultipleLocator(1))
    axL.tick_params(which="both", labelsize=8.4, colors=MUTE, length=0)
    axL.grid(which="major", color="#e1e0d9", lw=1, zorder=0)
    axL.grid(which="minor", color="#f0efe9", lw=0.7, zorder=0)
    axL.set_axisbelow(True)
    for sp in ("top", "right"):
        axL.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axL.spines[sp].set_color("#c3c2b7")
    axL.legend(handles=hL, loc="upper left", frameon=False, fontsize=9.0,
               labelcolor=INK2)
    axL.set_title(f"折扣越大越颠：corr = {D['折扣'].corr(D['标准差']):+.2f}",
                  fontsize=10.6, color=INK, loc="left", pad=10)

    # ---- 右：5% 分位（尾部），按稳排序
    T2 = T.sort_values("5%分位")
    y = np.arange(len(T2))
    cols = [S1 if D.at[c, "子类"] == CORE_SUB else S2 for c in T2.index]
    axR.barh(y, T2["5%分位"] * 100, height=0.62, color=cols, zorder=3)
    for yi, c, v in zip(y, T2.index, T2["5%分位"] * 100):
        axR.text(v - 0.3, yi, f"{v:.1f}", va="center", ha="right",
                 fontsize=8.3, color=INK2, zorder=4)
        axR.text(0.35, yi, f"{c}   胜率 {D.at[c,'胜率']:.0%}",
                 va="center", ha="left", fontsize=8.3, color=INK2, zorder=4)
    axR.axvline(0, color="#c3c2b7", lw=1.4, zorder=2)
    axR.set_yticks([])
    axR.set_ylim(-1.15, len(T2) - 0.35)
    lo = T2["5%分位"].min() * 100
    axR.set_xlim(lo - 4.2, 10.6)
    axR.xaxis.set_major_locator(MultipleLocator(5))
    axR.xaxis.set_minor_locator(MultipleLocator(1))
    axR.tick_params(which="both", axis="x", labelsize=8.4, colors=MUTE, length=0)
    axR.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    axR.grid(which="major", axis="x", color="#e1e0d9", lw=1, zorder=0)
    axR.grid(which="minor", axis="x", color="#f0efe9", lw=0.7, zorder=0)
    axR.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        axR.spines[sp].set_visible(False)
    axR.spines["bottom"].set_color("#c3c2b7")
    h = [plt.Line2D([], [], marker="s", ls="", ms=8, color=S1, label="美股宽基"),
         plt.Line2D([], [], marker="s", ls="", ms=8, color=S2, label="非宽基")]
    axR.legend(handles=h, loc="lower right", frameon=False, fontsize=9.0,
               labelcolor=INK2, ncols=2)
    axR.set_title("越靠右越好：8 周收益的 5% 分位（最差的 1/20 情形）",
                  fontsize=10.6, color=INK, loc="left", pad=10)

    fig.suptitle("保底仓位：折扣最大的都不稳，稳的一批集中在美股宽基与美股行业",
                 fontsize=15.2, color=INK, x=0.030, ha="left", y=0.975,
                 fontweight="bold")
    fig.text(0.030, 0.925,
             "8 周窗口（n≈692，已剔除站点缺行导致的失真窗口）；"
             "折扣 = 9/16→9/18 已实现涨跌，按 9/16 旧价成交即可锁定，一次性。\n"
             "左：两日跳 5-7% 本身就是高波动的表现，所以「折扣最大」与「收益平稳」"
             "结构上对立。右：保底看的是尾部（5% 分位），不是收益均值。\n"
             "右图口径 = 胜率×2 + 5%分位，前排 7 只美股宽基 + 8 只美股行业。"
             "注意：美元/信用债 ETF 的尾部其实更浅（标准差 2.7%、5% 分位 −4.4%），"
             "只是 8 周期望收益 ≈0，被这个口径扣掉了。",
             fontsize=9.5, color=INK2, ha="left", va="top")
    fig.subplots_adjust(left=0.058, right=0.988, top=0.808, bottom=0.075)
    path = os.path.join(OUT_FIG, "F_core_satellite.png")
    fig.savefig(path, facecolor=PLANE)
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT_CSV, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)
    p("=" * 100)
    p("全球资产配置模拟 —— 4a：保底仓位与折扣仓位")
    p("=" * 100)

    px, ext, tax = load()
    D = profile(px, ext, tax)
    p(f"  标的 {len(D)} 只，8 周窗口锚点 {len(anchors(px))} 个"
      f"（已按日历跨度剔除站点缺行导致的失真窗口）")

    part_a(D, tax)
    part_b(D)
    T, core, sat = part_c(D)
    C = part_d(D, tax)
    f = fig_core(D, C)

    D.to_csv(os.path.join(OUT_CSV, "H_profile_all.csv"),
             encoding="utf-8-sig")
    T.to_csv(os.path.join(OUT_CSV, "H_focus8_check.csv"),
             encoding="utf-8-sig")
    C.to_csv(os.path.join(OUT_CSV, "H_core_candidates.csv"),
             encoding="utf-8-sig")

    p("")
    p("=" * 100)
    p("  一句话结论")
    p("=" * 100)
    n_b = sum(1 for c in C.index if tax.at[c, "功能子类"] == CORE_SUB)
    p(f"  · 「SPY/QQQ 一带」= 【美股宽基】子类，8 只；"
      f"前 15 名稳的里有 {n_b} 只来自它。")
    p("  · 折扣最大的 8 只里，**只有 QQQ / XLK 够格进保底**，其余 6 只只能当卫星。")
    p("  · 折扣与平稳**结构对立**（corr +0.47）—— 别指望一只标的两样都占。")
    p("  · 折扣只能覆盖 5% 尾部损失的约 1/4，**当不了安全垫**。")
    p("  · 「保底」在只看绝对收益的口径下是**付费选项**：降亏损概率、降期望收益。")
    p("")
    p(f"  图  {f}")
    p("  表  H_profile_all.csv / H_focus8_check.csv / H_core_candidates.csv")
    p("=" * 100)


if __name__ == "__main__":
    main()
