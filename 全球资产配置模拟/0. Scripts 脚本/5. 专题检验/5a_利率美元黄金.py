#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全球资产配置模拟 —— 5a：加息 → 美元？加息 → 黄金？

回答用户 2026-09-21 的两个问题：
  1.「美联储加息，美元一般来说应该是走多吧」
  2.「为何这次加息，黄金和贵金属的表现相反」

★ 本脚本的核心是把两件事**分开**：

  A. **同期关系**（"理论对不对"）—— 同一段窗口内，利率动 → 美元/黄金怎么动。
     这是用户问的问题，是一个关于世界如何运转的陈述。

  B. **预测关系**（"能不能拿它交易"）—— 事前知道利率动过之后，
     **接下来** 8 周美元/黄金怎么走。这是赚钱需要的。

  3d 已经证明这两个答案**可以完全相反**（油价状态对能源：同期很强、
  预测归零）。所以这里必须分开测，不能拿 A 的结论去做 B 的决策。

债券符号约定：
  SHY 跌 = 前端利率升；IEF 跌 = 中期利率升；TLT 跌 = 长端利率升。
  所以「加息 → 美元涨」在数据上表现为 corr(UUP, SHY) < 0。

数据陷阱（沿用 3d）：
  * 窗口按 iloc 行号取，站点 2023-08~2024-02 缺行 → 用日历跨度过滤。
  * 站点价格未复权 → 用 wide_close_adj_usd.csv。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # 脚本在「0. Scripts 脚本/<步骤>/」，上跳两层才是项目根
DATA_DIR = os.path.join(ROOT, "1. Data 数据")     # 共享缓存（第 1 步产物），全步骤只读
STEP_DIR = os.path.join(ROOT, "5. 专题检验")   # 本步骤的产物目录
# 产物按类型分家：表进 csv/，图进 fig/
OUT_CSV, OUT_FIG = os.path.join(STEP_DIR, "csv"), os.path.join(STEP_DIR, "fig")
# 跨步骤只读：第 2 步产出的功能分类表，8 个脚本依赖它
TAX_CSV = os.path.join(ROOT, "2. 资产池认知", "csv", "T_taxonomy.csv")

N8, N1, STEP, MAXF, MAXL = 40, 5, 5, 70, 210
L = 126

# 利率曲线的三个点 + 美元 + 贵金属 + 对照
RATES = {"SHY": "1-3年美债(前端/政策)", "IEF": "7-10年美债(中期)",
         "TLT": "20年+美债(长端)"}
USD = ["UUP", "UDN"]
METAL = ["GLD", "SLV"]
OTHER = ["GDX", "SIL"]
BENCH = ["SPY", "USO", "DBC"]

p = print
g = lambda x: f"{x*100:+.2f}%"


def load():
    px = pd.read_csv(os.path.join(DATA_DIR, "wide_close_adj_usd.csv"),
                     encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    ext = pd.read_csv(os.path.join(DATA_DIR, "wide_close_extgap_usd.csv"),
                      encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    return px, ext


def anchors(px):
    return [i for i, t in enumerate(px.index)
            if i % STEP == 0 and i + N8 < len(px)
            and (px.index[i + N8] - t).days <= MAXF
            and (t - px.index[i - L]).days <= MAXL]


def r2(x, y):
    """两列同期收益的 R² 与斜率。"""
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 20 or x.std() == 0:
        return np.nan, np.nan, 0
    b, a = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return r ** 2, b, len(x)


# ---------------------------------------------------------------- Part A

def part_a(px, idx):
    p("")
    p("=" * 104)
    p("Part A —— 同期关系：利率动的时候，美元/黄金**同时**在做什么")
    p("=" * 104)
    p("  做法：取每个 8 周窗口，比较各资产**同一段窗口内**的收益。")
    p("  这不是预测，是「这段行情里它们是否一起动」—— 回答的是理论对不对。")
    p("")
    p("  符号提醒：债券**跌** = 利率**升**。所以「加息→美元涨」= 负相关。")
    p("")

    assets = USD + METAL + OTHER + BENCH
    hdr = f"  {'资产':<6}{'中文名':<16}" + "".join(
        f"{k:>22}" for k in RATES)
    p(hdr)
    p("  " + "-" * 98)
    # 中文名
    tax = pd.read_csv(TAX_CSV,
                      encoding="utf-8-sig", index_col=0)
    rows = []
    for a in assets:
        line = f"  {a:<6}{tax.at[a,'中文名']:<16}"
        rec = {"标的": a, "中文名": tax.at[a, "中文名"]}
        for k in RATES:
            x = np.array([px[k].iloc[i + N8] / px[k].iloc[i] - 1 for i in idx])
            y = np.array([px[a].iloc[i + N8] / px[a].iloc[i] - 1 for i in idx])
            rr, b, n = r2(x, y)
            rec[f"R²_{k}"] = rr
            rec[f"β_{k}"] = b
            line += f"{rr:>10.2f} (β{b:>+6.2f})"
        p(line)
        rows.append(rec)
    p("")
    p("  解读：β > 0 表示「债券涨(利率降)时它涨」= **加息时它跌**；")
    p("        β < 0 表示「债券跌(利率升)时它涨」= **加息时它涨**。")
    p("")

    D = pd.DataFrame(rows).set_index("标的")
    for k in RATES:
        p(f"  【以 {k} 为利率代理】按 R² 排序：")
        s = D[f"R²_{k}"].sort_values(ascending=False)
        top = "  ".join(f"{c} {v:.2f}(β{D.at[c,f'β_{k}']:+.2f})"
                        for c, v in s.head(6).items())
        p(f"    {top}")
    p("")
    return D


# ---------------------------------------------------------------- Part B

def state(px, idx, k, N=L):
    """事前可知的「利率↑」状态：k 在过去 N 个交易日的收益 < 0。"""
    return np.array([1.0 if px[k].iloc[i] / px[k].iloc[i - N] - 1 < 0 else 0.0
                     for i in idx], dtype=bool)


def part_b(px, idx, tax):
    p("")
    p("=" * 104)
    p("Part B —— 预测关系：事前知道「利率升」之后，**接下来** 8 周怎么走")
    p("=" * 104)
    p("  状态定义（无未来函数）：利率↑ = 该债券过去 126 日收益 < 0。")
    p("  关键列是**超额** = 条件均值 − 无条件均值。这才是这个状态的贡献。")
    p("")
    assets = RATES and (USD + METAL + OTHER + BENCH)
    for k in RATES:
        st = state(px, idx, k)
        p(f"  【利率代理 = {k}】当前状态：" +
          ("利率↑" if st[-1] else "利率↓") +
          f"   （{k} 过去126日 "
          f"{px[k].iloc[idx[-1]]/px[k].iloc[idx[-1]-L]-1:+.2%}，"
          f"命中 {st.sum()}/{len(st)} 个窗口）")
        p(f"    {'资产':<6}{'中文名':<16}{'条件均值':>10}{'无条件':>10}"
          f"{'超额':>10}{'条件胜率':>10}{'无条件胜率':>11}{'有效t':>8}")
        p("    " + "-" * 88)
        for a in assets:
            v = np.array([px[a].iloc[i + N8] / px[a].iloc[i] - 1 for i in idx])
            m = ~np.isnan(v)
            v, s = v[m], st[m]
            if s.sum() < 20 or (~s).sum() < 20:
                continue
            cm, um = v[s].mean(), v.mean()
            # 有效 t：重叠窗口按非重叠段数折算（3d 做法）
            runs = np.sum(np.diff(np.concatenate([[0], s.astype(int), [0]])) == 1)
            t_eff = ((cm - um) / v[s].std(ddof=1) * np.sqrt(max(runs, 1))
                     if v[s].std(ddof=1) > 0 else np.nan)
            flag = " ★" if abs(cm - um) > 0.01 and abs(t_eff) > 1.5 else ""
            p(f"    {a:<6}{tax.at[a,'中文名']:<16}{cm:>+10.2%}{um:>+10.2%}"
              f"{cm-um:>+10.2%}{(v[s]>0).mean():>10.1%}{(v>0).mean():>11.1%}"
              f"{t_eff:>8.2f}{flag}")
        p("")


# ---------------------------------------------------------------- Part C

def part_c(px, idx):
    p("")
    p("=" * 104)
    p("Part C —— 「加息周期」不是单一事件：把连续区间拆出来看")
    p("=" * 104)
    p("  做法（3d 方法论）：把「利率↑」状态拆成**连续段**，逐段看结果。")
    p("  窗口数 ≠ 独立事件数；重叠 8 周窗口会重复计同一段行情。")
    p("")
    st = state(px, idx, "IEF")
    # 连续段
    d = np.diff(np.concatenate([[0], st.astype(int), [0]]))
    segs = list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))
    p(f"  IEF 利率↑ 状态共 {len(segs)} 个连续段"
      f"（下表只列 ≥3 个窗口的段，并连续重新编号）。逐段看 UUP / GLD：")
    p(f"    {'段':<4}{'起':<12}{'止':<12}{'窗口':>5}"
      f"{'UUP均值':>10}{'GLD均值':>10}{'SPY均值':>10}")
    p("    " + "-" * 66)
    n = 0
    nup = ndn = 0
    kept = []
    for a, b in segs:
        ii = idx[a:b + 1]
        if len(ii) < 3:
            continue
        n += 1
        vals = {}
        for x in ["UUP", "GLD", "SPY"]:
            v = np.array([px[x].iloc[i + N8] / px[x].iloc[i] - 1 for i in ii])
            vals[x] = np.nanmean(v)
        kept.append((n, px.index[ii[0]].date(), len(ii), vals))
        if vals["UUP"] > 0:
            nup += 1
        else:
            ndn += 1
        p(f"    {n:<4}{px.index[ii[0]].date().isoformat():<12}"
          f"{px.index[ii[-1]].date().isoformat():<12}{len(ii):>5}"
          f"{vals['UUP']:>+10.2%}{vals['GLD']:>+10.2%}{vals['SPY']:>+10.2%}")
    p("")
    p(f"  ★ **{n} 段里 UUP 为正的只有 {nup} 段，为负的有 {ndn} 段。**")
    p("    方向都不一致，就不叫规律 —— 这叫「同一个标签下装了一堆不同的行情」。")
    p(f"    GLD 在利率↑时为正的有 {sum(1 for _,_,_,v in kept if v['GLD']>0)} 段、"
      f"为负的有 {sum(1 for _,_,_,v in kept if v['GLD']<=0)} 段 —— 同样是五五开。")
    p("")
    p("  ⚠ 看这一段的意义：如果各段方向**正负相间**，说明「利率↑状态」不是")
    p("    一个稳定的规律，而是一堆互不相同的行情被同一个标签归了类。")
    p("")


# ---------------------------------------------------------------- Part D

def part_d(px, idx, tax):
    p("")
    p("=" * 104)
    p("Part D —— 那黄金到底跟什么走？给 GLD 做一次因子筛选")
    p("=" * 104)
    p("  对每个候选因子，算它与 GLD **同期** 8 周收益的 R²（不是预测）。")
    p("")
    cands = ["SHY", "IEF", "TLT", "UUP", "UDN", "SPY", "USO", "DBC",
             "FXE", "FXY", "IBIT", "USCI"]
    y = np.array([px["GLD"].iloc[i + N8] / px["GLD"].iloc[i] - 1 for i in idx])
    rows = []
    for c in cands:
        if c not in px.columns:
            continue
        x = np.array([px[c].iloc[i + N8] / px[c].iloc[i] - 1 for i in idx])
        rr, b, n = r2(x, y)
        rows.append({"因子": c, "中文名": tax.at[c, "中文名"], "R²": rr, "β": b})
    T = pd.DataFrame(rows).set_index("因子").sort_values("R²", ascending=False)
    p(T.to_string(formatters={"R²": lambda x: f"{x:.3f}",
                              "β": lambda x: f"{x:+.2f}"}))
    p("")
    best = T.index[0]
    p(f"  → 解释力最强的因子是 **{best}（{T.at[best,'中文名']}）**，"
      f"R² = {T.at[best,'R²']:.3f}。")
    p(f"    利率系三个代理的 R²：" +
      "  ".join(f"{k} {T.at[k,'R²']:.3f}" for k in ["SHY", "IEF", "TLT"]))
    p("")
    p("  ⚠ R² 低**不等于**黄金随机 —— 只说明它不被**这些**因子解释。")
    p("    黄金的定价里有一大块是各国央行购金、地缘、ETF 流向，")
    p("    这些是这个 80 只 ETF 的池子里根本没有变量的东西。")
    p("")
    return T


# ---------------------------------------------------------------- Part E

def part_e(px, ext):
    p("")
    p("=" * 104)
    p("Part E —— 回到 2026-09-16 这次加息：当天和之后两天各发生了什么")
    p("=" * 104)
    se = px.index.max()
    gap = [d for d in ext.index if d > se]
    p(f"  站点停在 {se.date()}（含 FOMC 当天），外部数据到 {gap[-1].date()}。")
    p("")
    p(f"  {'资产':<6}{'中文名':<16}{'9/15→9/16':>12}{'9/16→9/18':>12}{'合计':>10}")
    p("  " + "-" * 58)
    tax = pd.read_csv(TAX_CSV,
                      encoding="utf-8-sig", index_col=0)
    prev = px.index[-2]
    for a in ["UUP", "UDN", "GLD", "SLV", "GDX", "SIL", "SPY", "TLT", "SHY", "USO"]:
        d1 = px[a].loc[se] / px[a].loc[prev] - 1
        d2 = ext.loc[gap[-1], a] / ext.loc[se, a] - 1
        p(f"  {a:<6}{tax.at[a,'中文名']:<16}{d1:>+12.2%}{d2:>+12.2%}"
          f"{(1+d1)*(1+d2)-1:>+10.2%}")
    p("")
    p("  ★ 关键：**看 UUP 那一行**。如果加息真的推升美元，这里应该是正的。")
    p("")


def main():
    os.makedirs(OUT_CSV, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)
    p("=" * 104)
    p("全球资产配置模拟 —— 5a：加息 → 美元？加息 → 黄金？")
    p("=" * 104)
    px, ext = load()
    idx = anchors(px)
    tax = pd.read_csv(TAX_CSV,
                      encoding="utf-8-sig", index_col=0)
    p(f"  有效锚点 {len(idx)} 个（8 周窗口，已按日历跨度剔除站点缺行导致的失真）")

    D = part_a(px, idx)
    part_b(px, idx, tax)
    part_c(px, idx)
    T = part_d(px, idx, tax)
    part_e(px, ext)

    D.to_csv(os.path.join(OUT_CSV, "K_rate_contemporaneous.csv"),
             encoding="utf-8-sig")
    T.to_csv(os.path.join(OUT_CSV, "K_gld_factors.csv"), encoding="utf-8-sig")
    p("  表  K_rate_contemporaneous.csv / K_gld_factors.csv")
    p("=" * 104)


if __name__ == "__main__":
    main()
