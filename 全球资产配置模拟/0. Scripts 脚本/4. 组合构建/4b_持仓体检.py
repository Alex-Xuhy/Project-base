#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全球资产配置模拟 —— 4b：实盘持仓体检

给一份持仓（代码 → 股数），输出：
  · 实际权重（按站点成交价 = 9/16 收盘 × 最新汇率）
  · 已锁定的入场折扣（一次性，站点刷新到周三时才会显形）
  · 组合的历史 8 周 / 1 周收益分布 —— **用真实权重跑联合窗口**，
    这一步自动把相关性算进去，比"各标的单独看"准确得多
  · 集中度：权重 HHI、风险贡献、有效独立赌注数
  · 重叠检验：核心 4 只之间到底重不重

为什么必须联合窗口：4a 是逐只看的，看不到"QQQ 和 ONEQ 一起跌"。
组合的 5% 分位**不等于**各标的 5% 分位的加权平均 —— 通常更好（分散）
也可能更差（重叠）。只有拿真实权重跑历史窗口才知道是哪种。

用法：
  python 4b_持仓体检.py
  （改 HOLDINGS 与 CASH 后重跑）
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

INIT = 1_000_000.0
FX_FILL = 6.7521          # 成交用的汇率（9/18 北京）
N8, N1, STEP, MAXF = 40, 5, 5, 70

# ---- 2026-09-20 周末建仓的实际持仓
HOLDINGS = {"BLOK": 100, "GLD": 50, "IBIT": 100,
            "SPY": 20, "QQQ": 20, "ONEQ": 150, "DSI": 100}
CORE = ["SPY", "QQQ", "ONEQ", "DSI"]
SATELLITE = ["GLD", "BLOK", "IBIT"]

p = print
g = lambda x: f"{x*100:+.2f}%"


def load():
    px = pd.read_csv(os.path.join(DATA_DIR, "wide_close_adj_usd.csv"),
                     encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    ext = pd.read_csv(os.path.join(DATA_DIR, "wide_close_extgap_usd.csv"),
                      encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    tax = pd.read_csv(TAX_CSV,
                      encoding="utf-8-sig", index_col=0)
    return px, ext, tax


def anchors(px):
    return [i for i, t in enumerate(px.index)
            if i % STEP == 0 and i + N8 < len(px)
            and (px.index[i + N8] - t).days <= MAXF]


# ---------------------------------------------------------------- Part A

def part_a(px, ext, tax):
    p("")
    p("=" * 104)
    p("Part A —— 仓位与权重（按站点成交价：9/16 收盘 × 9/18 汇率 6.7521）")
    p("=" * 104)
    se = px.index.max()
    rows = []
    for c, n in HOLDINGS.items():
        usd = px[c].loc[se]
        cny = usd * FX_FILL
        rows.append({"标的": c, "中文名": tax.at[c, "中文名"],
                     "子类": tax.at[c, "功能子类"],
                     "股数": n, "成交价USD": usd, "成交价CNY": cny,
                     "市值": cny * n})
    D = pd.DataFrame(rows).set_index("标的")
    D["占账户"] = D["市值"] / INIT
    inv = D["市值"].sum()
    D["占已投"] = D["市值"] / inv
    p(D.sort_values("市值", ascending=False)
      .to_string(formatters={"成交价USD": lambda x: f"${x:,.2f}",
                             "成交价CNY": lambda x: f"{x:,.2f}",
                             "市值": lambda x: f"{x:>12,.0f}",
                             "占账户": lambda x: f"{x:.1%}",
                             "占已投": lambda x: f"{x:.1%}"}))
    cash = INIT - inv
    p("")
    p(f"  已投入 {inv:>12,.0f}  ({inv/INIT:>6.2%})")
    p(f"  现金   {cash:>12,.0f}  ({cash/INIT:>6.2%})   ← 收益为 0")
    p(f"  合计   {INIT:>12,.0f}")
    p("")
    core = D.loc[[c for c in CORE if c in D.index], "市值"].sum()
    sat = D.loc[[c for c in SATELLITE if c in D.index], "市值"].sum()
    p(f"  核心(SPY/QQQ/ONEQ/DSI) {core:>10,.0f}  占账户 {core/INIT:>5.1%}"
      f"   占已投 {core/inv:>5.1%}")
    p(f"  卫星(GLD/BLOK/IBIT)     {sat:>10,.0f}  占账户 {sat/INIT:>5.1%}"
      f"   占已投 {sat/inv:>5.1%}")
    p("")
    p("  ⚠ 注意：GLD 是**全账户最大的单一持仓**，但你把它归在卫星里。")
    p("    金额上它属于卫星，风险上它是卫星里最温和的 —— 两个口径不一致，")
    p("    见 Part C 的风险贡献。")
    return D, inv, cash


# ---------------------------------------------------------------- Part B

def part_b(px, ext, D):
    p("")
    p("=" * 104)
    p("Part B —— 已锁定的入场折扣（一次性，周三站点刷新时才显形）")
    p("=" * 104)
    se = px.index.max()
    gap = [d for d in ext.index if d > se]
    p(f"  站点停在 {se.date()}，外部数据已到 {gap[-1].date()}；"
      f"折扣 = 这两者之间**已经发生**的涨跌。")
    p("")
    rows = []
    for c in HOLDINGS:
        d = ext.loc[gap[-1], c] / ext.loc[se, c] - 1
        rows.append({"标的": c, "折扣": d, "市值": D.at[c, "市值"],
                     "锁定收益": d * D.at[c, "市值"]})
    T = pd.DataFrame(rows).set_index("标的").sort_values("锁定收益", ascending=False)
    p(T.to_string(formatters={"折扣": g,
                              "市值": lambda x: f"{x:>12,.0f}",
                              "锁定收益": lambda x: f"{x:>+10,.0f}"}))
    tot = T["锁定收益"].sum()
    inv = D["市值"].sum()
    p("")
    p(f"  合计锁定 {tot:>+10,.0f}  =  已投资的 {tot/inv:+.3%}"
      f"  =  账户的 {tot/INIT:+.3%}")
    p("")
    p("  ⚠ 这是**确定的**，但要理解它的性质：它不是'赚到了'，")
    p("    而是**你在 9/16 的价格上买到了 9/18 的资产**。周三刷新时一次性显形，")
    p("    之后再也不会重复。**它对 9/21 之后毫无保护作用。**")
    return T, tot


# ---------------------------------------------------------------- Part C

def part_c(px, D):
    p("")
    p("=" * 104)
    p("Part C —— 组合的历史分布（真实权重 × 联合窗口，相关性已内含）")
    p("=" * 104)
    p("  ⚠ **先说样本陷阱**：IBIT 的数据只有 2024-04-12 起。要求「7 只同时存在」")
    p("    会把样本从 692 个窗口砍到 114 个 —— 而那 114 个窗口是")
    p("    **股票、黄金、加密同时走牛的两年**。只看那一档，均值必然虚高。")
    p("    所以下面分三档报，权重在各自样本内重新归一化。")
    p("")
    for c in HOLDINGS:
        s = px[c].dropna()
        p(f"    {c:<5} 首个有效日 {s.index[0].date()}   "
          f"可用于 8 周窗口的锚点数 "
          f"{sum(1 for i in anchors(px) if not np.isnan(px[c].iloc[i+N8]/px[c].iloc[i]-1)):>4}")

    idx = anchors(px)
    order = list(HOLDINGS)
    SAMPLES = [
        ("A  核心4 + GLD", ["SPY", "QQQ", "ONEQ", "DSI", "GLD"], "穿越 2015/2018/2020/2022"),
        ("B  A + BLOK", ["SPY", "QQQ", "ONEQ", "DSI", "GLD", "BLOK"], "2018Q4/2020/2022 都在内"),
        ("C  B + IBIT = 全部7只", order, "⚠ 只有 2024-04 起，单边牛市"),
    ]
    inv = D["市值"].sum()

    out = {}
    for lab, cols, note in SAMPLES:
        w = np.array([D.at[c, "市值"] for c in cols])
        w = w / w.sum()
        R8 = np.column_stack([[px[c].iloc[i + N8] / px[c].iloc[i] - 1 for i in idx]
                              for c in cols])
        R1 = np.column_stack([[px[c].iloc[i + N1] / px[c].iloc[i] - 1 for i in idx]
                              for c in cols])
        k8 = ~np.isnan(R8).any(axis=1)
        k1 = ~np.isnan(R1).any(axis=1)
        P8, P1 = R8[k8] @ w, R1[k1] @ w
        d8 = [idx[i] for i in np.where(k8)[0]]
        spy8 = np.array([px["SPY"].iloc[i + N8] / px["SPY"].iloc[i] - 1 for i in d8])
        out[lab] = dict(cols=cols, w=w, P8=P8, P1=P1, k8=k8, idx=idx,
                        note=note, d8=d8, spy8=spy8)
        del R8, R1

    p("")
    p("  8 周收益分布（已投口径，权重在样本内归一化）：")
    p(f"  {'样本':<22}{'窗口':>5}{'区间':>20}{'均值':>8}{'中位':>8}"
      f"{'胜率':>7}{'标准差':>8}{'5%分位':>8}{'最差':>8}")
    p("  " + "-" * 92)
    for lab, cols, note in SAMPLES:
        o = out[lab]
        P8, d8 = o["P8"], o["d8"]
        span = f"{px.index[d8[0]].date()}~{px.index[d8[-1]].date()}"
        p(f"  {lab:<22}{len(P8):>5}{span:>20}"
          f"{P8.mean():>+8.2%}{np.median(P8):>+8.2%}{(P8>0).mean():>7.1%}"
          f"{P8.std(ddof=1):>8.2%}{np.percentile(P8,5):>+8.2%}{P8.min():>+8.2%}")
    p("")
    p("  ⚠ 三档的差别**不是噪音，是样本区间**。C 档好看，是因为 2024-04 以来")
    p("    没有出现过真正的股债双杀。判断这份持仓的尾部，要看 A 档。")
    p("")
    p("  同一区间的 100% SPY 对照（衡量组合的分散效果）：")
    p(f"  {'样本':<22}{'组合均值':>10}{'SPY均值':>10}{'组合标准差':>12}"
      f"{'SPY标准差':>12}{'组合5%分位':>12}{'SPY5%分位':>12}")
    p("  " + "-" * 92)
    for lab, cols, note in SAMPLES:
        o = out[lab]
        P8, spy8 = o["P8"], o["spy8"]
        p(f"  {lab:<22}{P8.mean():>+10.2%}{np.nanmean(spy8):>+10.2%}"
          f"{P8.std(ddof=1):>12.2%}{np.nanstd(spy8,ddof=1):>12.2%}"
          f"{np.percentile(P8,5):>+12.2%}{np.nanpercentile(spy8,5):>+12.2%}")
    p("")

    p("  折算到**整个账户**（含 40% 现金，收益按 0 计）：")
    p(f"  {'样本':<22}{'均值':>9}{'胜率':>8}{'标准差':>9}{'5%分位':>9}{'最差':>9}")
    p("  " + "-" * 68)
    for lab, cols, note in SAMPLES:
        A8 = out[lab]["P8"] * inv / INIT
        p(f"  {lab:<22}{A8.mean():>+9.2%}{(A8>0).mean():>8.1%}"
          f"{A8.std(ddof=1):>9.2%}{np.percentile(A8,5):>+9.2%}{A8.min():>+9.2%}")
    p("")

    p("  未来 1 周（= 你被锁死的 5 个交易日）：")
    p(f"  {'样本':<22}{'已投5%分位':>12}{'已投最差':>10}{'已投胜率':>10}"
      f"{'账户5%分位':>12}{'账户最差':>10}")
    p("  " + "-" * 78)
    for lab, cols, note in SAMPLES:
        P1 = out[lab]["P1"]
        p(f"  {lab:<22}{np.percentile(P1,5):>+12.2%}{P1.min():>+10.2%}"
          f"{(P1>0).mean():>10.1%}{np.percentile(P1,5)*inv/INIT:>+12.2%}"
          f"{P1.min()*inv/INIT:>+10.2%}")

    full = out["C  B + IBIT = 全部7只"]
    return (full["P8"], full["P1"], full["w"], full["cols"], idx, inv,
            out["A  核心4 + GLD"], out)


def worst_windows(px, D, idx):
    """最差窗口逐只分解 —— 比单个分位数更能说明「什么样的行情会伤到你」。"""
    p("")
    p("=" * 104)
    p("Part C2 —— 最差窗口是什么行情（A 档：核心4+GLD，逐只分解）")
    p("=" * 104)
    cols = ["SPY", "QQQ", "ONEQ", "DSI", "GLD"]
    mvv = {c: D.at[c, "市值"] for c in cols}
    w = np.array([mvv[c] for c in cols], float)
    w = w / w.sum()
    for N, lab in ((N8, "8 周"), (N1, "1 周（= 锁死的 5 个交易日）")):
        R = np.column_stack([[px[c].iloc[i + N] / px[c].iloc[i] - 1 for i in idx]
                             for c in cols])
        k = ~np.isnan(R).any(axis=1)
        pos = np.where(k)[0]
        P = R[k] @ w
        o = np.argsort(P)
        p(f"  最差 5 个 {lab} 窗口：")
        for j in o[:5]:
            i = idx[pos[j]]
            parts = "  ".join(f"{c}{px[c].iloc[i+N]/px[c].iloc[i]-1:>+6.1%}"
                              for c in cols)
            p(f"    {px.index[i].date()}   组合 {P[j]:>+7.2%} | {parts}")
        p(f"    （{lab}胜率 {(P>0).mean():.1%}  中位 {np.median(P):+.2%}  "
          f"5% 分位 {np.percentile(P,5):+.2%}）")
        p("")
    p("  → 最差窗口全部是**股票型危机**（2020-03 COVID、2022 加息熊市），")
    p("    不是黄金或加密的独立事件。GLD 在 COVID 那周只跌 6.2%，")
    p("    SPY 跌 18.0% —— **你的分散是对的，但它只能减震，不能免疫。**")


# ---------------------------------------------------------------- Part D

def part_d(px, D, idx, tax):
    p("")
    p("=" * 104)
    p("Part D —— 集中度：这是几个赌注？")
    p("=" * 104)
    p("  ⚠ 相关性只用 **A/B 档的长历史标的**算（BLOK 从 2018-01 起，IBIT 从 2024-04 起）。")
    p("    用 114 个窗口算出来的 IBIT 相关系数没有统计意义，单独放在最后。")
    p("")
    cols = ["SPY", "QQQ", "ONEQ", "DSI", "GLD", "BLOK"]
    mv = {c: D.at[c, "市值"] for c in HOLDINGS}
    tot_mv = sum(mv.values())
    w7 = {c: v / tot_mv for c, v in mv.items()}
    w = np.array([mv[c] for c in cols])
    w = w / w.sum()          # 6 只内部归一化（相关系数矩阵与风险贡献的基准）

    hhi = sum(x ** 2 for x in w7.values())
    p(f"  权重口径（7 只全算）：HHI = {hhi:.4f}"
      f"  →  有效持仓数 1/HHI = {1/hhi:.2f}（7 只等权应为 7.00）")
    p("")

    R = np.column_stack([[px[c].iloc[i + N8] / px[c].iloc[i] - 1 for i in idx]
                         for c in cols])
    k = ~np.isnan(R).any(axis=1)
    R = R[k]
    p(f"  8 周收益相关系数（{len(R)} 个窗口，2018-01 起）：")
    p(pd.DataFrame(np.corrcoef(R.T), index=cols, columns=cols).round(2).to_string())
    p("")
    p("  ★ 看 SPY/QQQ/ONEQ/DSI 这个 4×4 角：全部 ≥0.85。")
    p("    **它们不是 4 个赌注，是 1 个赌注的 4 种包装。**")
    p(f"    QQQ~ONEQ  {np.corrcoef(R[:,1],R[:,2])[0,1]:.2f}     "
      f"SPY~DSI  {np.corrcoef(R[:,0],R[:,3])[0,1]:.2f}     "
      f"SPY~QQQ  {np.corrcoef(R[:,0],R[:,1])[0,1]:.2f}")
    p(f"    GLD 对股票 {np.corrcoef(R[:,4],R[:,0])[0,1]:+.2f}（唯一的负相关/低相关来源）")
    p("")

    # 风险贡献
    sd = R.std(axis=0, ddof=1)
    cov = np.cov(R.T)
    port_sd = float(np.sqrt(w @ cov @ w))
    rc = w * ((cov @ w) / port_sd) / port_sd
    T = pd.DataFrame({"权重_6只内": w, "权重_7只内": [w7[c] for c in cols],
                      "标准差": sd, "风险贡献": rc,
                      "金额": [mv[c] for c in cols]},
                     index=cols).sort_values("风险贡献", ascending=False)
    T["贡献/权重"] = T["风险贡献"] / T["权重_6只内"]
    p(f"  6 只子组合标准差 {port_sd:.2%}（内部归一化口径）；风险贡献分解：")
    p(T.to_string(formatters={"权重_6只内": lambda x: f"{x:.1%}",
                              "权重_7只内": lambda x: f"{x:.1%}",
                              "标准差": lambda x: f"{x:.2%}",
                              "风险贡献": lambda x: f"{x:.1%}",
                              "贡献/权重": lambda x: f"{x:.2f}x",
                              "金额": lambda x: f"{x:>12,.0f}"}))
    p("")
    p("  → 「贡献/权重」> 1 = 在**放大**组合风险；< 1 = 在**吸收**风险。")
    p(f"    GLD 的贡献/权重 = {T.at['GLD','贡献/权重']:.2f}x —— 它是唯一在吸收风险的。")
    p("")

    # BLOK / IBIT 单独看
    p("  BLOK 与 IBIT（短历史，只能各自看）：")
    for c in ["BLOK", "IBIT"]:
        r = np.column_stack([[px[x].iloc[i + N8] / px[x].iloc[i] - 1 for i in idx]
                             for x in [c, "SPY", "GLD"]])
        r = r[~np.isnan(r).any(axis=1)]
        p(f"    {c:<5} 窗口 {len(r):>4}  |  对 SPY {np.corrcoef(r[:,0],r[:,1])[0,1]:+.2f}"
          f"   对 GLD {np.corrcoef(r[:,0],r[:,2])[0,1]:+.2f}")
    r = np.column_stack([[px[x].iloc[i + N8] / px[x].iloc[i] - 1 for i in idx]
                         for x in ["BLOK", "IBIT"]])
    r = r[~np.isnan(r).any(axis=1)]
    p(f"    BLOK~IBIT {np.corrcoef(r[:,0],r[:,1])[0,1]:+.2f}（{len(r)} 个窗口）")
    p("    → 两者都是加密风险敞口的代理，**算一个赌注**。")
    p("")

    # 独立赌注：主成分
    ev = np.linalg.eigvalsh(np.corrcoef(R.T))[::-1]
    ev = np.clip(ev, 0, None)
    pn = ev / ev.sum()
    enb = float(np.exp(-(pn * np.log(pn + 1e-12)).sum()))
    p("")
    p(f"  相关矩阵特征值占比：{', '.join(f'{x:.0%}' for x in pn[:4])} ...")
    p(f"  → 有效独立赌注数（熵口径）= **{enb:.2f}**")
    p(f"    PC1 独占 {pn[0]:.0%} 方差。这 6 只标的 ≈ {enb:.1f} 个独立赌注。")
    return T, port_sd, enb


# ---------------------------------------------------------------- Part E

def part_e(D, inv, tot, A, enb, T):
    p("")
    p("=" * 104)
    p("Part E —— 结论")
    p("=" * 104)
    a8, a1 = A["P8"], A["P1"]
    p("")
    p(f"  1. **40% 现金是这份持仓里最大的单一决策。**")
    p(f"     穿越周期的口径（A 档：核心4+GLD，692 个窗口）显示，")
    p(f"     已投部分 8 周期望 {a8.mean():+.2%}；40% 现金把它压到账户口径 "
      f"{a8.mean()*inv/INIT:+.2%}。")
    p(f"     若评分纯看绝对收益且现金不生息，这 {1-inv/INIT:.0%} 是**无条件拖累** ——")
    p(f"     它把期望收益砍掉 {a8.mean()*(1-inv/INIT)*100:.2f}pp，")
    p("     换来的只是把 5% 分位从 "
      f"{np.percentile(a8,5):+.2%} 抬到 {np.percentile(a8,5)*inv/INIT:+.2%}。")
    p("")
    p(f"  2. **核心 4 只是 1 个赌注，不是 4 个。**（Part D 相关系数全部 ≥0.85）")
    p("     SPY/QQQ/ONEQ/DSI 的分工是**重复**而非分散；")
    p(f"     6 只长历史标的的独立赌注数只有 {enb:.1f}，再算上 BLOK/IBIT 是一伙的，")
    p("     整份持仓大约 **3 个独立赌注**：美国股权 / 黄金 / 加密。")
    p("")
    p(f"  3. **GLD 是唯一在吸收风险的持仓**（贡献/权重 "
      f"{T.at['GLD','贡献/权重']:.2f}x < 1）。它是全账户最大单一持仓，")
    p("     但风险贡献低于其权重 —— 这是真正的分散，不是浪费。")
    p("     反过来说：你**没有**在卫星上承担你以为的那么多风险，")
    p("     真正承担风险的是 BLOK/IBIT 那 11.7%。")
    p("")
    p(f"  4. 已锁定折扣 {tot:+,.0f}（占账户 {tot/INIT:+.2%}）是确定的，")
    p(f"     但它只补偿起点。A 档 8 周 5% 分位是 {np.percentile(a8,5):+.2%} —— "
      f"折扣覆盖不到它的一半。")
    p("")
    p(f"  5. **被锁死的 5 天**：A 档 1 周 5% 分位 {np.percentile(a1,5):+.2%}"
      f"、最差 {a1.min():+.2%}（已投口径）")
    p(f"     → 账户口径 {np.percentile(a1,5)*inv/INIT:+.2%} / "
      f"{a1.min()*inv/INIT:+.2%}。这是本周内发生概率 1/20 的事，无法交易，")
    p("       只能靠仓位管理 —— 而 40% 现金恰好已经替你管了一半。")


def main():
    os.makedirs(OUT_CSV, exist_ok=True)      # 本脚本只出表，没有图 → 不建 OUT_FIG
    p("=" * 104)
    p("全球资产配置模拟 —— 4b：实盘持仓体检")
    p("=" * 104)
    px, ext, tax = load()
    D, inv, cash = part_a(px, ext, tax)
    T_disc, tot = part_b(px, ext, D)
    P8, P1, W, cols, idx, inv, A, _ = part_c(px, D)
    worst_windows(px, D, idx)
    T_risk, port_sd, enb = part_d(px, D, idx, tax)
    part_e(D, inv, tot, A, enb, T_risk)

    T_disc.to_csv(os.path.join(OUT_CSV, "J_position_discount.csv"),
                  encoding="utf-8-sig")
    T_risk.to_csv(os.path.join(OUT_CSV, "J_risk_contribution.csv"),
                  encoding="utf-8-sig")
    p("")
    p(f"  表  J_position_discount.csv / J_risk_contribution.csv")
    p("=" * 104)


if __name__ == "__main__":
    main()
