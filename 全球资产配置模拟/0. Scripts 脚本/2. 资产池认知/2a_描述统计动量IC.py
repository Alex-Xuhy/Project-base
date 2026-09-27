#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全球资产配置模拟 —— 2a：数据探索

回答三组问题：
  A. 资产池的波动结构长什么样？谁是"离群值"？
  B. 去掉高波动离群值后，剩下的等权持仓，过去两年月度收益如何？超额几何？
  C. 单资产统计、截面动量 IC、自相关、相关性结构（为 3a 策略设计铺路）

数据来源一律用 wide_close_adj_usd.csv（**已复权**）—— 未复权数据会因拆股事件
产生 -50% 的假单日收益（详见 1b 与 README）。

⚠ 两个方法学要点，看结论前必读：
  1) 前视偏差：Part B 的"静态分组"用**整段两年**的波动率挑股票，再去算**同一段**的收益，
     属于描述性刻画，**不能当作可交易策略的业绩**。故同时给出"动态分组"版本
     （每次调仓只用当时可得的历史算波动率）作为对照。
  2) 停更 vs 晚上市：判定停更必须看**最后有效日**，不能看缺失比例 ——
     否则会把 BLOK/IBIT 这类新上市 ETF 一并误杀。

用法：
  python 2a_描述统计动量IC.py
  python 2a_描述统计动量IC.py --years 2 --no-charts
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from math import sqrt

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 全局

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # 脚本在「0. Scripts 脚本/<步骤>/」，上跳两层才是项目根
DATA_DIR = os.path.join(ROOT, "1. Data 数据")     # 共享缓存（第 1 步产物），全步骤只读
STEP_DIR = os.path.join(ROOT, "2. 资产池认知")   # 本步骤的产物目录
# 产物按类型分家：表进 csv/，图进 fig/
OUT_CSV, OUT_FIG = os.path.join(STEP_DIR, "csv"), os.path.join(STEP_DIR, "fig")

TRADING_DAYS = 252
VOL_WINDOW = 126          # 动态分组的波动率回看窗口（约半年）

# 用户既定配色（已过 CVD 验证器：4 项 PASS，1 项 contrast WARN → 用直接标注补偿）
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"
PLANE = "#fcfcfb"

CLASS_COLOR = {"Equity": S1, "FixedIncome": S3, "Commodity": S2, "Currency": S4}
CLASS_ORDER = ["Equity", "FixedIncome", "Commodity", "Currency"]

# 低波前 90/80/70/60/50% 是**有序**子集（按截断比例排序），不是并列的分类实体，
# 所以用单色渐变的 5 级序列色，而不是循环取分类色（循环会让两组撞成同一色）。
# 越"极端"的筛选（前50%）用越深的色。
SEQ5 = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]

# 图表文字一律避开 SimHei 缺字形（σ、U+2212、下标）；见仓库约定
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": PLANE,
    "axes.facecolor": PLANE,
    "savefig.facecolor": PLANE,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "font.size": 10,
})


def p(*a):
    print(*a, flush=True)


def style(ax, title="", xlabel="", ylabel="", grid_axis="y"):
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


# 各列的显示格式。⚠ 之前用 float_format 一刀切，把「夏普」也渲染成了 115.58%、
# 把「月数」渲染成 2400.00% —— 必须逐列指定。
PCT_COLS = {"累计收益", "年化收益", "年化波动", "最大回撤", "月胜率", "年化超额"}


def fmt_table(df):
    out = df.copy()
    for c in out.columns:
        if c == "月数":
            out[c] = out[c].astype(int).astype(str)
        elif c == "夏普":
            out[c] = out[c].map(lambda x: f"{x:>8.2f}" if pd.notna(x) else "     n/a")
        elif c in PCT_COLS:
            out[c] = out[c].map(lambda x: f"{x:>8.2%}" if pd.notna(x) else "     n/a")
    return out


# ---------------------------------------------------------------- 载入

def load():
    px = pd.read_csv(os.path.join(DATA_DIR, "wide_close_adj_usd.csv"),
                     encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    stocks = pd.read_csv(os.path.join(DATA_DIR, "stocks.csv"), encoding="utf-8-sig")
    meta = stocks.set_index("symbol")

    pool_last = px.index.max()
    # 停更判定看**最后有效日**：看缺失比例会把晚上市的新 ETF 一起误杀
    last_valid = px.apply(lambda s: s.last_valid_index())
    stale = [c for c in px.columns
             if pd.isna(last_valid[c]) or last_valid[c] < pool_last - pd.Timedelta(days=10)]
    if stale:
        p(f"⚠ 剔除停更标的 {stale}")
        for c in stale:
            p(f"    {c:<7} 最后有效日 {last_valid[c]:%Y-%m-%d}，落后池内最新 "
              f"{(pool_last - last_valid[c]).days} 天")

    keep = [c for c in px.columns if c not in stale]
    late = {c: px[c].first_valid_index() for c in keep}
    early = pool_last - pd.Timedelta(days=365 * 3)
    newer = {c: d for c, d in late.items() if d > early}
    if newer:
        p(f"ℹ 晚上市（历史 <3 年，不剔除，仅按可用区间参与）{len(newer)} 只: "
          + ", ".join(f"{c}({d:%Y-%m})" for c, d in sorted(newer.items(),
                                                          key=lambda kv: kv[1])))

    return px[keep], meta.loc[keep]


def month_ends(px):
    """每月最后一个交易日，作为调仓时点。"""
    s = px.groupby([px.index.year, px.index.month]).apply(lambda x: x.index[-1])
    # groupby.apply 在某些 pandas 版本下会把结果降级成字符串，显式转回 Timestamp
    return [pd.Timestamp(t) for t in s.tolist()]


# ---------------------------------------------------------------- Part A

def window_vol(px, start):
    """窗口内各资产年化波动率（按各自可用数据算，不因个别 NaN 丢整段）。"""
    sub = px.loc[px.index >= start]
    return (sub.pct_change().std() * np.sqrt(TRADING_DAYS)).dropna()


def part_a(px, meta, years):
    p("")
    p("=" * 78)
    p("Part A —— 波动结构")
    p("=" * 78)

    end = px.index.max()
    start = end - pd.DateOffset(years=years)
    vol = window_vol(px, start).sort_values()

    p(f"全样本 {px.index.min():%Y-%m-%d} ~ {end:%Y-%m-%d}  {len(px)} 个交易日")
    p(f"波动窗口 {start:%Y-%m-%d} ~ {end:%Y-%m-%d}  可用标的 {len(vol)} 只")
    p(f"年化波动率: 最小 {vol.min():.1%} ({vol.index[0]})  "
      f"中位 {vol.median():.1%}  最大 {vol.max():.1%} ({vol.index[-1]})")

    q = vol.quantile([.1, .25, .5, .75, .9, .95])
    p("分位: " + "  ".join(f"P{int(k*100)}={v:.1%}" for k, v in q.items()))

    # MAD 稳健离群检测：波动率分布右偏严重，中位数/MAD 比均值/标准差稳
    med, mad = vol.median(), (vol - vol.median()).abs().median()
    scale = 1.4826 * mad
    p("")
    p(f"MAD 稳健离群检测（中位 {med:.1%}, MAD {mad:.1%}, 尺度 {scale:.1%}）:")
    rows = []
    for k in (2.0, 2.5, 3.0, 3.5):
        thr = med + k * scale
        out = vol[vol > thr]
        p(f"  k={k}: 阈值 {thr:>6.1%} → 剔除 {len(out):>2} 只  {list(out.index)}")
        rows.append({"k": k, "threshold": thr, "n_out": len(out),
                     "outliers": ",".join(out.index)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_CSV, "A_outliers_mad.csv"),
                              index=False, encoding="utf-8-sig")

    p("")
    p("波动最高的 15 只及其大类:")
    for s in vol.tail(15).index[::-1]:
        m = meta.loc[s]
        p(f"    {s:<7}{vol[s]:>7.1%}   {m['assetClass']:<12}{m['assetClassSub']}")

    p("")
    p("按大类的波动率（说明池子的异质性有多强）:")
    for c in CLASS_ORDER:
        sub = vol[[s for s in vol.index if meta.loc[s, "assetClass"] == c]]
        if len(sub):
            p(f"    {c:<12} n={len(sub):>2}  中位 {sub.median():>6.1%}  "
              f"范围 {sub.min():.1%} ~ {sub.max():.1%}")

    pd.DataFrame({"ann_vol": vol,
                  "assetClass": [meta.loc[s, "assetClass"] for s in vol.index],
                  "assetClassSub": [meta.loc[s, "assetClassSub"] for s in vol.index]}
                 ).to_csv(os.path.join(OUT_CSV, "A_vol_by_asset.csv"),
                          encoding="utf-8-sig")
    return vol, med + 2.5 * scale


def chart_vol(vol, meta, k25):
    fig, ax = plt.subplots(figsize=(13, 17))
    y = np.arange(len(vol))
    colors = [CLASS_COLOR.get(meta.loc[s, "assetClass"], MUTED) for s in vol.index]
    ax.barh(y, vol.values * 100, color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(vol.index, fontsize=7.5)
    for i, s in enumerate(vol.index):          # 标注离群区，补偿浅色对比度不足
        if vol[s] > k25:
            ax.text(vol[s] * 100 + 0.7, i, f"{vol[s]*100:.0f}%", va="center",
                    fontsize=7.5, color=INK)
    style(ax, "资产池年化波动率（横轴 0 起，不截断）", "年化波动率 (%)", "",
          grid_axis="x")
    ax.set_xlim(0, max(vol) * 100 * 1.12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[c]) for c in CLASS_ORDER]
    ax.legend(handles, CLASS_ORDER, loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "A_vol_barh.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- Part B

def portfolio_monthly(px, sel_fn, dates):
    """月末调仓等权组合。sel_fn(t) 返回该时点选中的标的；NaN 价格自动跳过。"""
    rows = []
    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        sel = sel_fn(t)
        if not sel:
            continue
        sub = px.loc[[t, t1], [c for c in sel if c in px.columns]]
        valid = sub.columns[sub.notna().all()]
        if len(valid) == 0:
            continue
        r = (sub.loc[t1, valid] / sub.loc[t, valid] - 1).mean()
        rows.append({"date": t1, "ret": r, "n": len(valid)})
    d = pd.DataFrame(rows).set_index("date")
    return d["ret"], d["n"]


def part_b(px, meta, vol, years):
    p("")
    p("=" * 78)
    p(f"Part B —— 等权组合模拟（过去 {years} 年，月度调仓）")
    p("=" * 78)

    end = px.index.max()
    start = end - pd.DateOffset(years=years)
    dates = month_ends(px.loc[px.index >= start])
    p(f"调仓时点 {dates[0]:%Y-%m-%d} ~ {dates[-1]:%Y-%m-%d}，共 {len(dates)-1} 个月")

    allsym = list(px.columns)
    bench, bench_n = portfolio_monthly(px, lambda t: allsym, dates)

    # ---- 静态分组（有前视）----
    # 保持最低的 keep 比例：阈值取 vol.quantile(keep)
    static = {}
    for keep in (0.9, 0.8, 0.7, 0.6, 0.5):
        thr = vol.quantile(keep)
        ks = list(vol[vol <= thr].index)
        static[f"低波前{int(keep*100)}%"] = ks

    for k in (2.0, 2.5, 3.0):
        thr = vol.median() + k * 1.4826 * (vol - vol.median()).abs().median()
        static[f"剔除MAD>k{k}"] = list(vol[vol <= thr].index)

    res_static, n_static = {}, {}
    res_static["全池等权 (基准)"] = bench
    n_static["全池等权 (基准)"] = bench_n
    for name, ks in static.items():
        res_static[name], n_static[name] = portfolio_monthly(px, lambda t, k=ks: k, dates)

    # ---- 动态分组（无前视）：每次调仓只用当时可得的历史 ----
    def dynamic(keep_frac):
        def sel(t):
            hist = px.loc[px.index <= t]
            if len(hist) < VOL_WINDOW + 5:
                return None
            v = (hist.tail(VOL_WINDOW).pct_change().std() * np.sqrt(TRADING_DAYS)).dropna()
            if v.empty:
                return None
            return list(v[v <= v.quantile(keep_frac)].index)
        return sel

    res_dyn, n_dyn = {}, {}
    for keep in (0.9, 0.8, 0.7, 0.6, 0.5):
        res_dyn[f"动态低波前{int(keep*100)}%"], n_dyn[f"动态低波前{int(keep*100)}%"] = \
            portfolio_monthly(px, dynamic(keep), dates)

    # ---- 汇总 ----
    def stats(r):
        n = len(r)
        cum = (1 + r).cumprod()
        ann_ret = cum.iloc[-1] ** (12 / n) - 1
        ann_vol = r.std() * np.sqrt(12)
        return {"月数": n, "累计收益": cum.iloc[-1] - 1, "年化收益": ann_ret,
                "年化波动": ann_vol, "夏普": ann_ret / ann_vol if ann_vol else np.nan,
                "最大回撤": (cum / cum.cummax() - 1).min(),
                "月胜率": (r > 0).mean()}

    base_ann = stats(bench)["年化收益"]

    def build(res):
        tab = pd.DataFrame({k: stats(v) for k, v in res.items()}).T
        tab["年化超额"] = tab["年化收益"] - base_ann
        return tab[["月数", "累计收益", "年化收益", "年化波动", "夏普",
                    "最大回撤", "月胜率", "年化超额"]]

    tab_s = build(res_static)
    tab_d = build(res_dyn)

    p("")
    p("--- 静态分组（⚠ 有前视偏差，仅供描述，不可当作可实现业绩）---")
    p(fmt_table(tab_s).to_string())
    p("")
    p("--- 动态分组（无前视，每次调仓用当时可得的历史算波动）---")
    p(fmt_table(tab_d).to_string())
    p("")
    p(f"（基准：全池 {len(allsym)} 只等权，年化 {base_ann:.2%}）")

    tab_s.to_csv(os.path.join(OUT_CSV, "B_static_stats.csv"), encoding="utf-8-sig")
    tab_d.to_csv(os.path.join(OUT_CSV, "B_dynamic_stats.csv"), encoding="utf-8-sig")

    # ---- 月度收益明细（按模式取键，避免硬编码只数）----
    detail = {"全池等权(基准)": bench}
    for k in res_static:
        if k.startswith("低波前80%") or k.startswith("低波前70%") or \
           k.startswith("低波前60%") or k.startswith("低波前50%"):
            detail[k] = res_static[k]
    det = pd.DataFrame(detail)
    p("")
    p("--- 月度收益明细（静态分组，%）---")
    p((det * 100).round(2).to_string())
    (det * 100).round(4).to_csv(os.path.join(OUT_CSV, "B_monthly_returns.csv"),
                                encoding="utf-8-sig")

    # ---- 超额显著性检验 ----
    p("")
    p("--- 超额收益检验（组合月收益 - 基准月收益）---")
    p(f"    {'组合':<22}{'月均超额':>10}{'t值':>8}{'胜率':>8}{'n':>5}")
    p("    " + "-" * 51)
    for src in (res_static, res_dyn):
        for name, r in src.items():
            if name.startswith("全池"):
                continue
            d = (r - bench).dropna()
            if len(d) < 6:
                continue
            t = d.mean() / (d.std(ddof=1) / sqrt(len(d)))
            p(f"    {name:<22}{d.mean():>+10.3%}{t:>+8.2f}"
              f"{(d>0).mean():>8.1%}{len(d):>5}")

    # ---- 归因：低波子集跑输，是"低波本身差"还是"低波 = 排掉了股票"？----
    p("")
    p("--- 归因 1：各分组的资产大类构成 ---")
    p(f"    {'组合':<14}{'Equity':>9}{'FixedIncome':>13}{'Commodity':>11}{'Currency':>10}")
    p("    " + "-" * 57)
    for name, ks in static.items():
        cls = [meta.loc[s, "assetClass"] for s in ks if s in meta.index]
        cnt = {c: cls.count(c) / len(cls) for c in CLASS_ORDER}
        p(f"    {name:<14}" + "".join(f"{cnt[c]:>12.1%}" + (" " if c == "Equity" else "")
                                      for c in CLASS_ORDER))
    all_cls = [meta.loc[s, "assetClass"] for s in allsym]
    bench_mix = {c: all_cls.count(c) / len(all_cls) for c in CLASS_ORDER}
    p(f"    {'全池等权':<14}" + "".join(f"{bench_mix[c]:>12.1%}" + (" " if c == "Equity" else "")
                                       for c in CLASS_ORDER))

    p("")
    p("--- 归因 2：单个大类等权（区分「低波本身差」与「股票本来就强」）---")
    for c in CLASS_ORDER:
        ks = [s for s in allsym if meta.loc[s, "assetClass"] == c]
        if len(ks) < 2:
            continue
        r, _ = portfolio_monthly(px, lambda t, k=ks: k, dates)
        st = stats(r)
        p(f"    {c:<14} n={len(ks):>2}  年化 {st['年化收益']:>7.2%}  "
          f"波动 {st['年化波动']:>6.2%}  夏普 {st['夏普']:>5.2f}  "
          f"超额 {st['年化收益'] - base_ann:>+7.2%}")

    # 组内低波：在 Equity 内部再做低波筛选，剥离"板块配置"的影响
    p("")
    p("--- 归因 3：股票内部再做低波筛选（剥离大类配置）---")
    eq = [s for s in allsym if meta.loc[s, "assetClass"] == "Equity"]
    veq = vol[eq]
    for keep in (0.5, 0.3):
        ks = list(veq[veq <= veq.quantile(keep)].index)
        r, _ = portfolio_monthly(px, lambda t, k=ks: k, dates)
        eqr, _ = portfolio_monthly(px, lambda t, k=eq: k, dates)
        p(f"    Equity 内低波前{int(keep*100)}% ({len(ks):>2}只)  年化 "
          f"{stats(r)['年化收益']:>7.2%}   vs 全体 Equity {stats(eqr)['年化收益']:>7.2%}"
          f"   组内超额 {stats(r)['年化收益'] - stats(eqr)['年化收益']:>+7.2%}")

    return bench, res_static, res_dyn, dates


def chart_cumulative(bench, res_static, res_dyn):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for ax, res, title in ((axes[0], res_static, "静态分组（有前视，仅描述）"),
                           (axes[1], res_dyn, "动态分组（无前视，可实现）")):
        b = (1 + bench).cumprod()
        ax.plot(b.index, b.values, color=INK, linewidth=2.4, label="全池等权 (基准)")
        ax.annotate(f"{b.iloc[-1]:.2f}", (b.index[-1], b.iloc[-1]),
                    textcoords="offset points", xytext=(5, 0), fontsize=8.5,
                    color=INK, va="center", fontweight="bold")
        # 只画"低波前 NN%"这组有序子集，按截断比例从浅到深配单色渐变
        keeps = sorted([k for k in res if k.startswith("低波前") or k.startswith("动态低波前")],
                       key=lambda k: -int(k.split("前")[1].rstrip("%")))
        for name, c in zip(keeps, SEQ5):
            cum = (1 + res[name]).cumprod()
            ax.plot(cum.index, cum.values, color=c, linewidth=1.8, label=name)
            ax.annotate(f"{cum.iloc[-1]:.2f}", (cum.index[-1], cum.iloc[-1]),
                        textcoords="offset points", xytext=(5, 0),
                        fontsize=8.5, color=c, va="center")
        ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.6)
        style(ax, title, "", "累计净值（起点 = 1）")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        ax.margins(x=0.16)
    fig.suptitle("低波动筛选 vs 全池等权 —— 累计净值（月度调仓，已复权）",
                 color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "B_cumulative.png"), dpi=150)
    plt.close(fig)


def chart_monthly(bench, res_static):
    keys = [k for k in res_static if k.startswith("低波前60%")]
    if not keys:
        return
    name = keys[0]
    ex = (res_static[name] - bench).dropna()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.4), sharex=True)
    ax = axes[0]
    ax.bar(np.arange(len(bench)), bench.values * 100, color=S1, width=0.66)
    ax.axhline(0, color=INK, linewidth=1.0)
    style(ax, "全池等权（基准）月度收益", "", "月收益 (%)")

    ax = axes[1]
    colors = [S3 if v >= 0 else S2 for v in ex.values]
    ax.bar(np.arange(len(ex)), ex.values * 100, color=colors, width=0.66)
    ax.axhline(0, color=INK, linewidth=1.0)
    style(ax, f"{name} 相对基准的超额收益", "调仓月", "超额 (%)")
    ax.set_xticks(np.arange(len(ex)))
    ax.set_xticklabels([d.strftime("%y-%m") for d in ex.index], rotation=60,
                       fontsize=7.5, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "B_monthly_bars.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- Part C

def part_c(px, meta):
    p("")
    p("=" * 78)
    p("Part C —— 描述性统计（为 3a 动量设计铺路）")
    p("=" * 78)

    ret = px.pct_change()
    n = ret.notna().sum()
    ann = (1 + ret).prod() ** (TRADING_DAYS / n) - 1
    v = ret.std() * np.sqrt(TRADING_DAYS)
    cum = (1 + ret).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    per = pd.DataFrame({"年化收益": ann, "年化波动": v,
                        "夏普": ann / v.replace(0, np.nan), "最大回撤": mdd})
    p("")
    p("--- 单资产统计（全样本 2012 起）---")
    p(f"    年化收益 中位 {ann.median():.1%}  范围 {ann.min():.1%} ~ {ann.max():.1%}")
    p(f"    夏普     中位 {per['夏普'].median():.2f}  "
      f"范围 {per['夏普'].min():.2f} ~ {per['夏普'].max():.2f}")
    p(f"    最大回撤 中位 {mdd.median():.1%}  最差 {mdd.min():.1%} ({mdd.idxmin()})")
    per.to_csv(os.path.join(OUT_CSV, "C_asset_stats.csv"), encoding="utf-8-sig")

    # ---- 截面动量 IC ----
    p("")
    p("--- 截面动量 IC（Spearman，月频）---")
    p("    信号 = 过去 L 个交易日的累计收益；预测 = 未来 21 个交易日收益")
    p("    每个调仓点做一次截面秩相关，再看 IC 序列的均值与显著性")
    p("")
    p(f"    {'L(日)':>6}{'L(月)':>7}{'IC均值':>10}{'IC标准差':>10}{'ICIR':>9}"
      f"{'t值':>8}{'胜率':>8}{'n':>6}")
    p("    " + "-" * 64)

    pxw = px.loc[px.index >= "2015-01-01"]
    dates = month_ends(pxw)
    ic_rows = []
    for L in (21, 63, 126, 189, 252, 378):
        ics = []
        for t in dates:
            i = pxw.index.searchsorted(t)
            if i < L or i + 21 >= len(pxw):
                continue
            past = (pxw.iloc[i] / pxw.iloc[i - L] - 1).dropna()
            fut = (pxw.iloc[i + 21] / pxw.iloc[i] - 1).dropna()
            common = past.index.intersection(fut.index)
            if len(common) < 20:
                continue
            ics.append(past[common].corr(fut[common], method="spearman"))
        if len(ics) < 12:
            continue
        ic = pd.Series(ics)
        icir = ic.mean() / ic.std(ddof=1)
        t = icir * sqrt(len(ic))
        p(f"    {L:>6}{L/21:>7.1f}{ic.mean():>10.4f}{ic.std(ddof=1):>10.4f}"
          f"{icir:>9.3f}{t:>8.2f}{(ic>0).mean():>8.1%}{len(ic):>6}")
        ic_rows.append({"L_days": L, "L_months": L / 21, "IC_mean": ic.mean(),
                        "IC_std": ic.std(ddof=1), "ICIR": icir, "t": t,
                        "hit": (ic > 0).mean(), "n": len(ic)})
    ic_df = pd.DataFrame(ic_rows)
    ic_df.to_csv(os.path.join(OUT_CSV, "C_momentum_ic.csv"), index=False,
                 encoding="utf-8-sig")

    # ---- 自相关 ----
    p("")
    p("--- 月度收益自相关（判定动量 / 反转）---")
    mret = px.resample("ME").last().pct_change()
    ac1 = pd.Series({c: mret[c].autocorr(1) for c in mret.columns}).dropna()
    ac1 = ac1[ac1.notna()]
    p(f"    一阶自相关中位 {ac1.median():+.3f}   正自相关 {(ac1>0).mean():.1%} 只")
    p(f"    范围 {ac1.min():+.3f} ~ {ac1.max():+.3f}")
    p(f"    → 中位数为 {'正' if ac1.median()>0 else '负'}，"
      f"{'动量占优' if ac1.median()>0 else '反转占优'}（仅就自相关而言）")
    ac1.to_frame("ac1").to_csv(os.path.join(OUT_CSV, "C_autocorr.csv"),
                               encoding="utf-8-sig")

    # ---- 相关性结构 ----
    p("")
    p("--- 相关性结构 ---")
    corr = ret.corr()
    iu = np.triu_indices_from(corr, k=1)
    p(f"    两两相关 中位 {np.median(corr.values[iu]):.3f}  "
      f"范围 {corr.values[iu].min():.3f} ~ {corr.values[iu].max():.3f}")
    ev = np.linalg.eigvalsh(corr.fillna(0).values)[::-1]
    p(f"    第一主成分解释 {ev[0]/ev.sum():.1%} 的方差  "
      f"（前 3 个累计 {ev[:3].sum()/ev.sum():.1%}）")
    mc = corr.mean().sort_values(ascending=False)
    p(f"    平均相关最高 5 只: " + ", ".join(f"{s}({mc[s]:.2f})" for s in mc.head(5).index))
    p(f"    平均相关最低 5 只: " + ", ".join(f"{s}({mc[s]:.2f})" for s in mc.tail(5).index))
    corr.to_csv(os.path.join(OUT_CSV, "C_corr.csv"), encoding="utf-8-sig")

    return ic_df


def chart_ic(ic_df):
    if ic_df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(ic_df))
    colors = [S3 if v > 0 else S2 for v in ic_df["IC_mean"]]
    ax.bar(x, ic_df["IC_mean"], color=colors, width=0.6)
    ax.axhline(0, color=INK, linewidth=1.0)
    for i, r in ic_df.iterrows():
        up = r["IC_mean"] >= 0
        ax.text(i, r["IC_mean"] + (0.004 if up else -0.004), f"t={r['t']:.1f}",
                ha="center", va="bottom" if up else "top", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(m)}月" for m in ic_df["L_months"]])
    # t 值标注画在柱顶上方，不放大留白会把最高那根的标签裁掉
    lo, hi = ic_df["IC_mean"].min(), ic_df["IC_mean"].max()
    ax.set_ylim(min(0, lo * 1.3), hi * 1.25)
    style(ax, "截面动量 IC —— 不同回看窗口（信号 vs 未来 1 月收益）",
          "回看窗口 L", "IC 均值（Spearman）")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "C_momentum_ic.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="2a：数据探索")
    ap.add_argument("--years", type=int, default=2, help="组合模拟回看年数")
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_CSV, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)
    p("=" * 78)
    p("全球资产配置模拟 —— 2a 数据探索")
    p(f"时间 {datetime.now():%Y-%m-%d %H:%M:%S}")
    p(f"输出 {OUT_CSV} / {OUT_FIG}")
    p("=" * 78)

    px, meta = load()
    p(f"参与分析的标的 {px.shape[1]} 只")

    vol, k25 = part_a(px, meta, args.years)
    bench, res_static, res_dyn, dates = part_b(px, meta, vol, args.years)
    ic_df = part_c(px, meta)

    if not args.no_charts:
        p("")
        p("生成图表 ...")
        chart_vol(vol, meta, k25)
        chart_cumulative(bench, res_static, res_dyn)
        chart_monthly(bench, res_static)
        chart_ic(ic_df)
        for f in sorted(os.listdir(OUT_FIG)):
            if f.endswith(".png"):
                p(f"    {f}")

    p("")
    p("=" * 78)
    p(f"完成。产物在「{STEP_DIR}/」的 csv/ 与 fig/ 两个子目录")
    p("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        p("\n中断。")
        sys.exit(130)
