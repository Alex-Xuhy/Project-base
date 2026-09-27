# -*- coding: utf-8 -*-
"""全球资产配置模拟 —— 5b：卫星加 TLT / UUP 值不值？

用户 2026-09-21：「我准备再增加一部分 TLT 和 UUP 的卫星仓位」。

拆成四个必须**分开**回答的问题：

  1. **它们是不是「保底」？**
     4a 已经证明「保底」有两个互不兼容的定义：
       口径甲 = 平时波动小（美元/信用债）
       口径乙 = 股票崩的时候它不崩（美股宽基）
     所以不能只看无条件标准差，必须看**股票最差窗口里的条件表现**。

  2. **它们是第 4、第 5 个赌注，还是伪装成新赌注的老赌注？**
     4b 已证明现有 7 只其实只有 3 个赌注（美股/黄金/加密）。
     TLT 和 UUP 是不是真的加法，看相关矩阵。

  3. **入场折扣** —— 本游戏唯一的确定性收益来源。
     4b 锁定 +14,936 = 账户 +1.494%（一次性，站内刷新到周三时显形）。
     没有折扣的标的，等于「用现金换了一次纯暴露」，一分钱不多拿。

  4. **账户口径的取舍** —— 现金 40% 每 8 周拖 0.83pp（4b）。
     加 TLT/UUP 就是拿这个拖累换成暴露。换得值不值？

数据口径（沿用 4b/5a）：
  * 站点股价停在 2026-09-16（周三），成交按该价 × 汇率 6.7521。
  * 折扣 = 外部数据里 9/16 → 9/18 的已实现涨跌（一次性）。
  * 窗口按 iloc 行号取 → 必须用日历跨度过滤站点缺行（2023-08~2024-02）。
  * 样本必须分档报（4b 教训：IBIT 把联合样本从 692 砍到 114）。
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

N8, N1, STEP, MAXF, MAXF1, MAXL = 40, 5, 5, 70, 12, 210
FX = 6.7521
INIT = 1_000_000.0

# 用户实际持仓（股数，2026-09-20 按站点 9/16 价成交）
HOLD = {"SPY": 20, "QQQ": 20, "ONEQ": 150, "DSI": 100,
        "GLD": 50, "BLOK": 100, "IBIT": 100}
CORE4 = ["SPY", "QQQ", "ONEQ", "DSI"]

# 新候选 + 参照
NEW = ["TLT", "UUP"]
REF = ["SHY", "IEF"]

p = print


def load():
    px = pd.read_csv(os.path.join(DATA_DIR, "wide_close_adj_usd.csv"),
                     encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    ext = pd.read_csv(os.path.join(DATA_DIR, "wide_close_extgap_usd.csv"),
                      encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    return px, ext


def anchors(px, n, maxf, look=0):
    """按 iloc 行号取窗口，但用日历跨度剔除站点缺行造成的失真。

    look = 回看行数；0 表示本脚本不需要事前状态。注意 look 是**行数**，
    换算成日历天要乘 ~1.4 —— 5a 用 126 行配 210 天阈值正好，若照抄
    「210 行 + 210 天」会把 691 个锚点悄悄砍到 42 个（见 4b 截断教训）。
    """
    out = []
    for i, t in enumerate(px.index):
        if i % STEP or i + n >= len(px):
            continue
        if (px.index[i + n] - t).days > maxf:
            continue
        if look and (t - px.index[i - look]).days > MAXL:
            continue
        out.append(i)
    return out


def ret(px, c, idx, n):
    return np.array([px[c].iloc[i + n] / px[c].iloc[i] - 1 for i in idx])


def span(px, idx, m, n=N8):
    """联合样本掩码：m 只标的在窗口两端都有数据。"""
    k = np.ones(len(idx), bool)
    for c in m:
        k &= ~np.isnan(ret(px, c, idx, n))
    return k


def stats(v):
    return (np.mean(v), np.std(v, ddof=1), np.quantile(v, 0.05),
            (v > 0).mean(), len(v))


def risk_contrib(cov, w):
    """各资产对组合方差的贡献占比（和为 1）。"""
    pv = w @ cov @ w
    return w * (cov @ w) / pv


def eff_bets(C):
    """有效独立赌注数（相关矩阵特征值的熵口径）。"""
    e = np.linalg.eigvalsh(C)
    e = e[e > 0]
    q = e / e.sum()
    return float(np.exp(-(q * np.log(q)).sum()))


# ---------------------------------------------------------------- Part A

def part_a(px, idx, idx1, tax):
    p("")
    p("=" * 108)
    p("Part A —— 逐只画像：它们到底是「保底」还是「赌注」")
    p("=" * 108)
    p("  列说明：「股跌时胜率」= 在 SPY 8 周为负的那些窗口里，它为正的比例。")
    p("       这才是「保底」的检验 —— 不是问它平不平，是问它**在股票崩的时候**平不平。")
    p("")
    cols = CORE4 + ["GLD", "BLOK", "TLT", "UUP", "SHY", "IEF"]
    p(f"  {'标的':<6}{'中文名':<15}{'首日':<12}{'窗口':>5}"
      f"{'8周均值':>9}{'标准差':>8}{'5%分位':>9}{'胜率':>7}"
      f"{'股跌时胜率':>11}{'最差8周':>9}{'发生':>12}{'1周5%分位':>10}")
    p("  " + "-" * 106)
    rows = []
    for c in cols:
        sp = px[c].dropna()
        j = ~np.isnan(ret(px, c, idx, N8))
        v = ret(px, c, idx, N8)[j]
        ii = [x for x, keep in zip(idx, j) if keep]
        sv = ret(px, "SPY", ii, N8)
        down = sv < 0
        cond = np.nan if c == "SPY" else (v[down] > 0).mean() if down.sum() >= 10 else np.nan
        w = ret(px, c, idx1, N1)
        w = w[~np.isnan(w)]
        if len(v) < 20 or len(w) < 20:
            p(f"  {c:<6}{tax.at[c,'中文名']:<15}{sp.index[0].date().isoformat():<12}"
              f"{len(v):>5}   —— 样本不足，跳过")
            continue
        mean, sd, q5, wr, n = stats(v)
        wpos = int(np.argmin(v))
        rows.append({"标的": c, "中文名": tax.at[c, "中文名"],
                     "首日": sp.index[0].date().isoformat(), "窗口": n,
                     "8周均值": mean, "标准差": sd, "5%分位": q5, "胜率": wr,
                     "股跌时胜率": cond, "最差8周": v[wpos],
                     "最差8周日期": px.index[ii[wpos]].date().isoformat(),
                     "1周5%分位": np.quantile(w, 0.05)})
        r = rows[-1]
        cond_s = f"{cond:>10.1%}" if cond == cond else f"{'—':>10}"
        p(f"  {c:<6}{r['中文名']:<15}{r['首日']:<12}{n:>5}"
          f"{mean:>+9.2%}{sd:>8.2%}{q5:>+9.2%}{wr:>7.1%}"
          f"{cond_s}{v[wpos]:>+9.2%}{r['最差8周日期']:>12}"
          f"{r['1周5%分位']:>+10.2%}")
    p("")
    A = pd.DataFrame(rows).set_index("标的")
    p("  → 读表三问：")
    d = A.loc[["TLT", "UUP", "GLD", "SHY"]]
    for c in ["TLT", "UUP"]:
        t = A.at[c, "股跌时胜率"]
        p(f"    {c}：标准差 {A.at[c,'标准差']:.2%}、股跌时胜率 {t:.1%}、"
          f"最差 8 周 {A.at[c,'最差8周']:+.2%}（{A.at[c,'最差8周日期']}）")
    p("")
    return A


# ---------------------------------------------------------------- Part B

def part_b(px, idx, tax):
    p("")
    p("=" * 108)
    p("Part B —— 它们是新赌注，还是老赌注换了个包装？")
    p("=" * 108)
    p("  做法（4b 方法论）：算 8 周收益的相关矩阵 + 有效独立赌注数。")
    p("")
    m = CORE4 + ["GLD", "BLOK", "TLT", "UUP"]
    j = span(px, idx, m)
    ii = [x for x, keep in zip(idx, j) if keep]
    R = pd.DataFrame({c: ret(px, c, ii, N8) for c in m})
    C = R.corr()
    p(f"  联合样本 {len(ii)} 个 8 周窗口"
      f"（{px.index[ii[0]].date()} ~ {px.index[ii[-1]].date()}）")
    p("")
    p("        " + "".join(f"{c:>8}" for c in m))
    for a in m:
        line = f"  {a:<6}"
        for b in m:
            v = C.at[a, b]
            line += f"{v:>8.2f}" if a != b else f"{'1':>8}"
        p(line)
    p("")
    p("  ★ 只看最后两行 —— TLT / UUP 与已有持仓的最大相关：")
    for c in NEW:
        s = C.loc[c, CORE4 + ["GLD", "BLOK"]].abs().sort_values(ascending=False)
        partner = s.index[0]
        p(f"    {c}：最大 |corr| = {s.iloc[0]:.2f}（对 {partner}）"
          f"，对美股四项 {'/'.join(f'{C.at[c,x]:+.2f}' for x in CORE4)}")
    p("")
    p(f"  TLT ~ UUP 之间：{C.at['TLT','UUP']:+.2f}")
    p("")
    for name, mm in [("现有 6 只（不含加密）", CORE4 + ["GLD", "BLOK"]),
                     ("现有 6 只 + TLT", CORE4 + ["GLD", "BLOK", "TLT"]),
                     ("现有 6 只 + UUP", CORE4 + ["GLD", "BLOK", "UUP"]),
                     ("现有 6 只 + TLT + UUP", m)]:
        p(f"    有效独立赌注数  {name:<26}= {eff_bets(C.loc[mm, mm].values):.2f}")
    p("")
    return C


# ---------------------------------------------------------------- Part C

def build(px, d, extra, cash):
    """按目标权重构建账户：extra = {标的: 占总账户比例}，cash = 现金比例。
    已投部分按现有持仓的市值比例分摊。"""
    base = {c: n * px[c].loc[d] * FX for c, n in HOLD.items()}
    tot = sum(base.values())
    w = {c: v / tot * (1 - cash - sum(extra.values())) for c, v in base.items()}
    w.update(extra)
    w["现金"] = cash
    return w


def acct_stats(px, ii, w):
    """账户口径的 8 周 / 1 周分布。权重键含「现金」（零收益）。"""
    risky = [c for c in w if c != "现金"]
    R8 = np.column_stack([ret(px, c, ii, N8) for c in risky])
    k = ~np.isnan(R8).any(axis=1)
    R8 = R8[k]
    wv = np.array([w[c] for c in risky])
    pr = R8 @ wv
    R1 = np.column_stack([ret(px, c, ii, N1) for c in risky])[k]
    p1 = R1 @ wv
    return pr, p1, risky, wv, k


def part_c(px, idx, tax):
    p("")
    p("=" * 108)
    p("Part C —— 账户口径：把现金换成 TLT / UUP，期望和尾部各换到了什么")
    p("=" * 108)
    p("  基准 = 用户现在的账户：7 只持仓 59.99% + 现金 40.01%。")
    p("  加仓 = 从现金里划出一部分买入 TLT / UUP（不卖出任何现有持仓）。")
    p("")
    se = px.index.max()
    m = CORE4 + ["GLD", "BLOK", "TLT", "UUP"]
    j = span(px, idx, m)
    ii = [x for x, keep in zip(idx, j) if keep]
    p(f"  样本 {len(ii)} 个窗口（{px.index[ii[0]].date()} ~ "
      f"{px.index[ii[-1]].date()}），已剔除站点缺行。")
    p("")

    plans = [("基准：现状（40% 现金）", {}),
             ("全现金（极端对照）", None),
             ("+ TLT  5%", {"TLT": 0.05}),
             ("+ TLT 10%", {"TLT": 0.10}),
             ("+ TLT 15%", {"TLT": 0.15}),
             ("+ UUP  5%", {"UUP": 0.05}),
             ("+ UUP 10%", {"UUP": 0.10}),
             ("+ UUP 15%", {"UUP": 0.15}),
             ("+ TLT5  UUP5", {"TLT": 0.05, "UUP": 0.05}),
             ("+ TLT8  UUP8", {"TLT": 0.08, "UUP": 0.08}),
             ("+ TLT10 UUP10", {"TLT": 0.10, "UUP": 0.10})]
    p(f"  {'方案':<22}{'8周期望':>9}{'8周标准差':>10}{'8周5%分位':>11}"
      f"{'1周5%分位':>11}{'最差8周':>10}{'相对现金':>10}")
    p("  " + "-" * 84)
    rows = []
    for name, extra in plans:
        if extra is None:
            w = {"现金": 1.0}
            R8 = np.zeros((len(ii), 1))
            pr = R8[:, 0]
            p1 = pr
        else:
            w = build(px, se, extra, 0.4001 - sum(extra.values()))
            pr, p1, _, _, _ = acct_stats(px, ii, w)
        mu, sd, q5, _, _ = stats(pr)
        q1 = np.quantile(p1, 0.05)
        wst = pr.min()
        rows.append({"方案": name, "8周期望": mu, "8周标准差": sd,
                     "8周5%分位": q5, "1周5%分位": q1, "最差8周": wst})
        rel = f"{mu:>+9.2%}"
        p(f"  {name:<22}{mu:>+9.2%}{sd:>10.2%}{q5:>+11.2%}{q1:>+11.2%}"
          f"{wst:>+10.2%}{rel:>10}")
    p("")
    T = pd.DataFrame(rows).set_index("方案")
    p("  ★ 「相对现金」= 该方案 8 周期望相对全现金账户的提升（全现金 = 0.0000%）。")
    p("    这就是「把 1 块钱从现金挪进这只标的」买到的期望。")
    p("")
    p("  现金在本站不计息，8 周期望 = 0。所以基准与全现金之差就是「持有这 7 只」")
    p("  赚到的。反推每只新候选在**本样本**上的自身期望（与 Part A 的全样本会有差）：")
    b = T.at["基准：现状（40% 现金）", "8周期望"]
    for lab, k in [("TLT", "+ TLT 10%"), ("UUP", "+ UUP 10%")]:
        p(f"    {lab}：该标的自身 8 周期望 ≈ {-(b-T.at[k,'8周期望'])/0.10:+.2%}"
          f"（现金 0.00%）")
    p("")
    return T, ii


# ---------------------------------------------------------------- Part D

def part_d(px, ii, tax):
    p("")
    p("=" * 108)
    p("Part D —— 风险贡献：加进来之后，谁在放大风险、谁在吸收")
    p("=" * 108)
    m = CORE4 + ["GLD", "BLOK", "TLT", "UUP"]
    j = span(px, ii, m)
    ii = [x for x, keep in zip(ii, j) if keep]
    R = np.column_stack([ret(px, c, ii, N8) for c in m])
    cov = np.cov(R, rowvar=False)
    se = px.index.max()
    for label, extra in [("现状（不含 TLT/UUP）", {}),
                         ("+ TLT 8% + UUP 8%", {"TLT": 0.08, "UUP": 0.08})]:
        w = build(px, se, extra, 0.4001 - sum(extra.values()))
        wv = np.array([w.get(c, 0.0) for c in m])
        rc = risk_contrib(cov, wv)
        p(f"  【{label}】")
        p(f"    {'标的':<6}{'中文名':<15}{'权重':>9}{'风险贡献':>10}{'贡献/权重':>11}")
        p("    " + "-" * 51)
        for i, c in enumerate(m):
            p(f"    {c:<6}{tax.at[c,'中文名']:<15}{wv[i]:>9.2%}{rc[i]:>10.2%}"
              f"{rc[i]/wv[i] if wv[i] > 0 else np.nan:>11.2f}")
        p(f"    {'现金':<6}{'现金':<15}{w['现金']:>9.2%}{0:>10.2%}{'—':>11}")
        p("")
    p("  贡献/权重 > 1 = 它在放大组合风险；< 1 = 它在吸收。")
    p("")


# ---------------------------------------------------------------- Part E

def part_e(px, idx, tax):
    p("")
    p("=" * 108)
    p("Part E —— 保底检验：股票最惨的 6 个 8 周窗口里，它们各做了什么")
    p("=" * 108)
    p("  做法（3d/7 方法论）：先按 SPY 排序挑出最差的窗口，")
    p("  且两个被选窗口间隔 > 8 周，避免同一段行情被重复计入。")
    p("")
    m = CORE4 + ["GLD", "BLOK", "TLT", "UUP", "SHY"]
    j = span(px, idx, m + ["SPY"])
    ii = [x for x, keep in zip(idx, j) if keep]
    sv = ret(px, "SPY", ii, N8)
    order = np.argsort(sv)
    picked, used = [], []
    for o in order:
        if all(abs(o - u) > N8 for u in used):
            picked.append(o)
            used.append(o)
        if len(picked) == 6:
            break
    picked = sorted(picked)
    names = ["SPY", "TLT", "UUP", "GLD", "SHY"]
    p(f"  {'窗口':<13}{'SPY':>10}{'TLT':>10}{'UUP':>10}{'GLD':>10}{'SHY':>10}")
    p("  " + "-" * 63)
    rets = {c: ret(px, c, ii, N8) for c in names}
    for o in picked:
        line = f"  {px.index[ii[o]].date().isoformat():<13}"
        for c in names:
            line += f"{rets[c][o]:>+10.2%}"
        p(line)
    p("")
    p("  六窗口平均：")
    line = f"  {'':<13}"
    for c in names:
        line += f"{np.mean([rets[c][o] for o in picked]):>+10.2%}"
    p(line)
    p("")
    p("  ⚠ 判断「保底」的全部依据就在这张表里 —— 不要去看向上时的涨幅。")
    p("")

    # 供画图使用：(窗口日期, {标的: 收益})
    crisis = [(px.index[ii[o]].date().isoformat(), {c: rets[c][o] for c in names})
              for o in picked]
    return crisis


# ---------------------------------------------------------------- Part F

def part_f(px, ext, tax):
    p("")
    p("=" * 108)
    p("Part F —— 入场折扣：本游戏唯一的确定性收益")
    p("=" * 108)
    p("  站点停在周三 2026-09-16；外部数据到周五 2026-09-18。")
    p("  按 9/16 价成交 → 站内刷新到周三时，这笔已实现涨跌**一次性**变成你的钱。")
    p("")
    se = px.index.max()
    gap = [d for d in ext.index if d > se]
    if not gap:
        # 无缺口 = 站点报价没滞后 → 「按旧价成交锁定涨跌」不再可得，折扣不存在。
        # 返回带表头的空表，保住 L_entry_discount.csv 的列结构（下游/阅读者都不会误读）。
        p("  ⚠ 本日无缺口（站点报价未滞后）→ 本节跳过。")
        p("     折扣只在「站点末日 → 外部源末日」这个窗口里存在；缺口归零即无从谈起。")
        p("")
        return pd.DataFrame(columns=["中文名", "9/16", "9/18", "折扣"]).rename_axis("标的")
    p(f"  {'标的':<6}{'中文名':<15}{'9/16 价':>10}{'9/18 价':>10}"
      f"{'折扣':>9}{'每 10 万 CNY':>13}")
    p("  " + "-" * 63)
    rows = []
    for c in ["TLT", "UUP", "SHY", "IEF", "GLD", "SPY"]:
        if c not in ext.columns:
            continue
        a, b = px[c].loc[se], ext.loc[gap[-1], c]
        disc = b / a - 1
        rows.append({"标的": c, "中文名": tax.at[c, "中文名"],
                     "9/16": a, "9/18": b, "折扣": disc})
        p(f"  {c:<6}{tax.at[c,'中文名']:<15}{a:>10.2f}{b:>10.2f}"
          f"{disc:>+9.2%}{disc*100000:>+13,.0f}")
    p("")
    p("  ⚠ 折扣是**一次性**的，不进入 8 周期望；但它是**当天就到手**的钱。")
    p("    「有折扣 + 后面的期望为正」才是值得买；「没折扣」就等于用现金换暴露。")
    p("")
    return pd.DataFrame(rows).set_index("标的")


# ---------------------------------------------------------------- 图

def fig(A, crisis, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    SURF, INK, INK2, MUT = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    GRID, BASE = "#e1e0d9", "#c3c2b7"
    C_OLD, C_NEW = "#2a78d6", "#eb6834"          # 槽位 1 / 2
    S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # 槽位 1-4（已过 validator）
    HOLD_SET = CORE4 + ["GLD", "BLOK"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.8, 6.6), facecolor=SURF,
        gridspec_kw={"width_ratios": [1.0, 1.15]})
    # 中文字体回退：Windows 上若 Microsoft YaHei 缺失不至于满屏豆腐块
    for ax in (ax1, ax2):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(BASE)
        ax.tick_params(colors=MUT, labelsize=9.5, length=0)

    # ---- 左：风险—收益平面 ----
    # 美股四项（SPY/QQQ/ONEQ/DSI）在这个平面上基本重合 —— 那正是 4b 的结论，
    # 所以不逐个标注（会糊成一团），改用一个浅色包络 + 一句说明。
    ax1.axhline(0, color=BASE, lw=1.0, zorder=1)
    ax1.grid(axis="both", color=GRID, lw=0.7, zorder=0)
    ax1.set_axisbelow(True)
    eq_x = [A.at[c, "标准差"] for c in CORE4]
    eq_y = [A.at[c, "8周均值"] for c in CORE4]
    ax1.add_patch(plt.matplotlib.patches.Ellipse(
        (np.mean(eq_x), np.mean(eq_y)),
        width=max(eq_x) - min(eq_x) + 0.016,
        height=max(eq_y) - min(eq_y) + 0.010,
        facecolor=C_OLD, alpha=0.10, edgecolor=C_OLD, lw=1.1, zorder=2))
    ax1.annotate("美股四项\n= 1 个赌注", (max(eq_x) + 0.010, np.mean(eq_y)),
                 fontsize=9.5, color=C_OLD, va="center", ha="left", zorder=4)

    OFF = {"GLD": (9, -4), "BLOK": (-6, 8), "TLT": (8, 1), "UUP": (8, 1)}
    for c in A.index:
        if np.isnan(A.at[c, "8周均值"]):
            continue
        x, y = A.at[c, "标准差"], A.at[c, "8周均值"]
        col = C_OLD if c in HOLD_SET else C_NEW
        ax1.scatter(x, y, s=110, color=col, edgecolor=SURF, lw=1.6, zorder=3)
        if c in OFF:
            ha = "right" if c == "BLOK" else "left"
            ax1.annotate(c, (x, y), textcoords="offset points", xytext=OFF[c],
                         fontsize=10.5, color=col, fontweight="bold",
                         ha=ha, zorder=4)
    ax1.scatter(0, 0, s=110, marker="s", color=MUT, edgecolor=SURF, lw=1.6, zorder=3)
    ax1.set_xlabel("8 周标准差（风险）", fontsize=10.5, color=INK2)
    ax1.set_ylabel("8 周平均收益", fontsize=10.5, color=INK2)
    ax1.set_title("风险—收益平面：TLT 与股票同风险、零收益", fontsize=13,
                  color=INK, pad=12, loc="left")
    ax1.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax1.set_xlim(-0.008, 0.198)
    ax1.set_ylim(-0.013, 0.042)
    ax1.set_xticks(np.arange(0, 0.20, 0.05))
    ax1.set_yticks(np.arange(-0.01, 0.045, 0.01))
    h1 = ax1.scatter([], [], s=110, color=C_OLD, label="现有持仓")
    h2 = ax1.scatter([], [], s=110, color=C_NEW, label="新候选 TLT / UUP")
    h3 = ax1.scatter([], [], s=110, marker="s", color=MUT, label="现金")
    ax1.legend(handles=[h1, h2, h3], loc="upper left", frameon=False,
               fontsize=10, labelcolor=INK2)

    # ---- 右：股票最惨 6 个窗口 ----
    names = ["SPY", "TLT", "UUP", "GLD"]
    ax2.axvline(0, color=BASE, lw=1.0, zorder=1)
    ax2.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax2.set_axisbelow(True)
    for j, (d, r) in enumerate(crisis):
        y = len(crisis) - 1 - j
        ax2.plot([min(r[c] for c in names), max(r[c] for c in names)], [y, y],
                 color=GRID, lw=1.4, zorder=1)
        for i, c in enumerate(names):
            ax2.scatter(r[c], y, s=105, color=S[i], edgecolor=SURF, lw=1.5, zorder=3)
    ax2.set_yticks(range(len(crisis)))
    ax2.set_yticklabels([d for d, _ in reversed(crisis)], fontsize=10)
    ax2.set_ylim(-0.7, len(crisis) - 0.3)
    ax2.set_xlabel("该窗口内 8 周收益", fontsize=10.5, color=INK2)
    ax2.set_title("股票最惨的 6 个 8 周窗口：谁真的保住了", fontsize=13,
                  color=INK, pad=12, loc="left")
    ax2.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax2.legend(handles=[ax2.scatter([], [], s=105, color=S[i], label=c)
                        for i, c in enumerate(names)],
               loc="lower left", frameon=False, fontsize=10, labelcolor=INK2,
               ncol=4, columnspacing=1.1, handletextpad=0.3)

    fig.suptitle("TLT / UUP 加仓体检 —— 样本 387 个 8 周窗口（2018-01 ~ 2026-07）",
                 fontsize=14, color=INK, x=0.007, ha="left", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)
    p(f"  图  {os.path.basename(out)}")


# ---------------------------------------------------------------- main

def main():
    os.makedirs(OUT_CSV, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)
    p("=" * 108)
    p("全球资产配置模拟 —— 5b：卫星加 TLT / UUP 值不值？")
    p("=" * 108)
    px, ext = load()
    tax = pd.read_csv(TAX_CSV,
                      encoding="utf-8-sig", index_col=0)
    idx = anchors(px, N8, MAXF)
    idx1 = anchors(px, N1, MAXF1)
    p(f"  有效锚点：8 周 {len(idx)} 个 / 1 周 {len(idx1)} 个（已按日历跨度剔除缺行失真）")
    p("  数据可用性（4b 教训：先打印再取联合样本）：")
    for c in CORE4 + ["GLD", "BLOK", "IBIT", "TLT", "UUP"]:
        sp = px[c].dropna()
        p(f"    {c:<6}{tax.at[c,'中文名']:<15}首日 {sp.index[0].date()}  "
          f"（有效 8 周窗口 {int((~np.isnan(ret(px,c,idx,N8))).sum())}/{len(idx)}）")

    A = part_a(px, idx, idx1, tax)
    C = part_b(px, idx, tax)
    T, ii = part_c(px, idx, tax)
    part_d(px, ii, tax)
    crisis = part_e(px, idx, tax)
    F = part_f(px, ext, tax)
    fig(A, crisis, os.path.join(OUT_FIG, "F_tlt_uup.png"))

    A.to_csv(os.path.join(OUT_CSV, "L_tlt_uup_profile.csv"), encoding="utf-8-sig")
    C.to_csv(os.path.join(OUT_CSV, "L_tlt_uup_corr.csv"), encoding="utf-8-sig")
    T.to_csv(os.path.join(OUT_CSV, "L_account_mix.csv"), encoding="utf-8-sig")
    F.to_csv(os.path.join(OUT_CSV, "L_entry_discount.csv"), encoding="utf-8-sig")
    p("  表  L_tlt_uup_profile.csv / L_tlt_uup_corr.csv / "
      "L_account_mix.csv / L_entry_discount.csv")
    p("=" * 108)


if __name__ == "__main__":
    main()
