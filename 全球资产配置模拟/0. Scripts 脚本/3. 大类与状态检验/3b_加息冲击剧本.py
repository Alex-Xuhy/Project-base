# -*- coding: utf-8 -*-
"""全球资产配置模拟 —— 3b：加息/紧缩冲击下，池内什么资产能逆势上涨

背景（2026-09-20）：
  美联储 9/16 全票加息 25bp 至 3.75-4.00%，2023-07 以来首次；点阵图显示
  年内大概率还有一次，18 人中 16 人预计至少再加一次、无人预计降息。
  同时美伊冲突推布伦特原油破 100 美元。→ **加息 + 油价冲击同时发生**。

本脚本用池内 2012-2026 全样本回答三件事：
  A. 加息冲击日的事件研究 —— 冲击当天 / 之后 1-5 / 6-21 / 22-42 日谁涨谁跌
  B. 上一轮完整加息周期（2022-01 ~ 2023-07）谁赢谁输
  C. 最贴切的类比：油价冲击 + 加息**同时**发生的历史片段
  D. 当前读数

方法学注意：
  * 「加息冲击日」没有 FOMC 日历，用 **TLT 大跌 + 美元上涨** 做代理
    （利率跳升 + 美元走强 = 鹰派冲击）。这是代理不是真值，已在正文说明。
  * 前瞻窗口逐日重叠 → 名义 n 严重高估，一律给**有效 t**（÷√重叠倍数）。
  * 加息周期里 79 只标的都在跌，所以结论看**相对排名**而非绝对正负。

用法：
  python 3b_加息冲击剧本.py
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

CLASSES = ["Equity", "FixedIncome", "Commodity", "Currency"]
CN = {"Equity": "股权", "FixedIncome": "固收", "Commodity": "商品", "Currency": "货币"}
p = print


def load():
    px = pd.read_csv(os.path.join(DATA_DIR, "wide_close_adj_usd.csv"),
                     encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    meta = pd.read_csv(os.path.join(DATA_DIR, "stocks.csv"),
                       encoding="utf-8-sig").set_index("symbol")
    lv = px.apply(lambda s: s.last_valid_index())
    keep = [c for c in px.columns if lv[c] >= px.index.max() - pd.Timedelta(days=10)]
    # 功能子类（与 2b 的 T_资产池清单.md 同源，这里只取子类字段）
    tax = pd.read_csv(TAX_CSV,
                      encoding="utf-8-sig", index_col=0)
    return px[keep], meta.loc[keep], tax.loc[keep]


def eff_t(s, overlap):
    """有效 t：把逐日重叠窗口的名义 t 按 √重叠倍数 折算。"""
    s = s.dropna()
    if len(s) < 5:
        return np.nan, len(s)
    t = s.mean() / (s.std(ddof=1) / sqrt(len(s)))
    return t / sqrt(overlap), len(s)


def fmt_row(name, s, basis_pts=True):
    t, n = eff_t(s, 1)
    return (f"  {name:<16}{s.mean():>+9.2%}{s.median():>+9.2%}"
            f"{(s > 0).mean():>8.1%}{s.std():>9.2%}{n:>7}")


# ---------------------------------------------------------------- Part A

def part_a(px, tax):
    """加息冲击日的事件研究。"""
    p("")
    p("=" * 104)
    p("Part A —— 加息冲击日的事件研究")
    p("=" * 104)

    ret = px.pct_change()
    tlt = ret["TLT"]
    uup = ret["UUP"]
    # 鹰派冲击 = 长债大跌 + 美元上涨。阈值取 TLT <= -1.0%（约 1 个日标准差）
    shock = (tlt <= -0.010) & (uup > 0)
    p(f"  代理定义: TLT 日跌幅 >= 1.0% 且 美元(UUP) 上涨")
    p(f"  命中 {shock.sum()} 天 / {len(shock)} 天 ({shock.mean():.1%})，"
      f"覆盖 {shock[shock].index.year.min()}-{shock[shock].index.year.max()}")
    p(f"  ⚠ 这是**代理**不是真 FOMC 日历；但方向性结论对阈值不敏感（见文末敏感性）")

    # 各类/各子类的条件收益
    pos = px.index.get_indexer(shock[shock].index)

    def fwd(i, a, b):
        """事件后第 a~b 个交易日的累计收益（截面）。窗口不完整则返回 None。"""
        j, k = i + a, min(i + b, len(px) - 1)
        if j >= len(px) or k <= j:
            return None
        return px.iloc[k] / px.iloc[j] - 1

    p("")
    p("  按功能子类（等权），冲击当天与之后的表现:")
    p(f"  {'子类':<16}{'当天':>9}{'后1-5日':>9}{'后6-21日':>9}{'后22-42日':>9}{'样本':>7}")
    p("  " + "-" * 62)
    rows = []
    for s in tax["功能子类"].unique():
        ks = [c for c in px.columns if tax.loc[c, "功能子类"] == s]
        if len(ks) < 1:
            continue
        # 当天 = 冲击日相对前一日的收益
        d0 = (px[ks].iloc[pos].values / px[ks].iloc[pos - 1].values - 1)
        d0 = pd.Series(d0.mean(axis=1))
        def agg(a, b):
            vals = [fwd(i, a, b) for i in pos]
            return pd.Series([v[ks].mean() for v in vals if v is not None])

        f1, f2, f3 = agg(1, 5), agg(6, 21), agg(22, 42)
        p(f"  {s:<16}{d0.mean():>+9.2%}{f1.mean():>+9.2%}{f2.mean():>+9.2%}"
          f"{f3.mean():>+9.2%}{len(d0):>7}")
        rows.append({"子类": s, "冲击当天": d0.mean(), "后1-5日": f1.mean(),
                     "后6-21日": f2.mean(), "后22-42日": f3.mean()})
    A = pd.DataFrame(rows)
    A["后22-42日_排名"] = A["后22-42日"].rank(ascending=False).astype(int)
    return A, shock, pos


# ---------------------------------------------------------------- Part B

def part_b(px):
    """上一轮完整加息周期谁赢。"""
    p("")
    p("=" * 104)
    p("Part B —— 上一轮完整加息周期（2022-01-03 ~ 2023-07-31，共 11 次加息）")
    p("=" * 104)

    seg = px.loc["2022-01-03":"2023-07-31"]
    r = seg.iloc[-1] / seg.iloc[0] - 1
    pool = (1 + seg.pct_change().mean(axis=1).fillna(0)).prod() - 1

    p(f"  区间内 全池等权 {pool:+.1%}")
    p(f"  上涨的标的: {(r > 0).sum()} / {len(r)} 只 —— **加息周期里绝大多数资产在跌，"
      f"所以看相对排名**")
    p("")
    p(f"  {'排名':>4}  {'最强的 12 只':<28}{'最弱的 12 只':<28}")
    p("  " + "-" * 62)
    top, bot = r.nlargest(12), r.nsmallest(12)
    for i in range(12):
        p(f"  {i+1:>4}  {top.index[i]+' '+f'{top.iloc[i]:+.1%}':<28}"
          f"{bot.index[i]+' '+f'{bot.iloc[i]:+.1%}':<28}")
    return r


# ---------------------------------------------------------------- Part C

def part_c(px, tax):
    """最贴切的类比：油价冲击 + 加息同时发生。

    ⚠ 2026-09-20 复核（3c_缺口后复核.py）：本节结论「当前更接近 2022H1、
      能源称王」已被随后两天的实际读数**部分推翻** —— 贵金属 +3.73%、
      能源 −0.51% 的组合落在 2018Q4 那一侧。本函数保留原样作为历史记录，
      但**不要单独引用它的结论**，务必同时看 3c。
    """
    p("")
    p("=" * 104)
    p("Part C —— 最贴切的类比：**油价冲击 + 加息周期同时发生**")
    p("=" * 104)
    p("  当前是「美伊冲突推油价破 100」+「美联储重启加息」的组合。")
    p("  历史上最接近的是 2022 上半年（俄乌 + 加息启动）和 2018Q4（加息 + 油价崩）。")

    episodes = {
        "2022H1 俄乌+加息启动": ("2022-01-03", "2022-06-30"),
        "2022 全年": ("2022-01-03", "2022-12-30"),
        "2018Q4 加息+油价崩": ("2018-09-28", "2018-12-24"),
        "2023H2 higher-for-longer": ("2023-07-31", "2023-10-27"),
        "2025 降息周期(对照)": ("2025-01-02", "2025-12-31"),
    }
    for lab, (a, b) in episodes.items():
        seg = px.loc[a:b]
        if len(seg) < 20:
            p(f"\n  {lab}: 数据不足")
            continue
        r = seg.iloc[-1] / seg.iloc[0] - 1
        pool = (1 + seg.pct_change().mean(axis=1).fillna(0)).prod() - 1
        sub = (seg.pct_change().mean(axis=1).fillna(0))
        p(f"\n  【{lab}】{a} ~ {b}   全池等权 {pool:+.1%}")
        # 子类层面
        bysub = {}
        for s in tax["功能子类"].unique():
            ks = [c for c in px.columns if tax.loc[c, "功能子类"] == s]
            bysub[s] = (seg[ks].iloc[-1] / seg[ks].iloc[0] - 1).mean()
        bs = pd.Series(bysub).sort_values(ascending=False)
        p("    子类: " + "  ".join(f"{k}{v:+.0%}" for k, v in bs.items()))
        p("    个券前5: " + "  ".join(f"{k}{v:+.0%}" for k, v in r.nlargest(5).items()))
        p("    个券后5: " + "  ".join(f"{k}{v:+.0%}" for k, v in r.nsmallest(5).items()))


# ---------------------------------------------------------------- Part D

def part_d(px, tax):
    p("")
    p("=" * 104)
    p("Part D —— 当前读数（数据末日 " + f"{px.index.max():%Y-%m-%d}" + "）")
    p("=" * 104)

    p("  近期动量（截至数据末日）:")
    p(f"  {'子类':<16}{'近5日':>10}{'近21日':>10}{'近63日':>10}{'近126日':>10}{'近252日':>10}")
    p("  " + "-" * 68)
    for s in tax["功能子类"].unique():
        ks = [c for c in px.columns if tax.loc[c, "功能子类"] == s]
        line = f"  {s:<16}"
        for n in (5, 21, 63, 126, 252):
            if len(px) <= n:
                line += f"{'n/a':>10}"
                continue
            v = (px[ks].iloc[-1] / px[ks].iloc[-1 - n] - 1).mean()
            line += f"{v:>+10.1%}"
        p(line)

    p("")
    p("  美元 / 长债 的位置（判断紧缩冲击是否已充分定价）:")
    for c, nm in [("UUP", "美元"), ("TLT", "20年+美债"), ("IEF", "7-10年美债"),
                  ("GLD", "黄金"), ("USO", "原油"), ("XLE", "能源股")]:
        if c not in px.columns:
            continue
        v5 = px[c].iloc[-1] / px[c].iloc[-6] - 1
        v21 = px[c].iloc[-1] / px[c].iloc[-22] - 1
        v252 = px[c].iloc[-1] / px[c].iloc[-253] - 1
        p(f"    {nm:<12}{c:<5} 近5日 {v5:>+7.1%}   近21日 {v21:>+7.1%}   "
          f"近252日 {v252:>+7.1%}")


# ---------------------------------------------------------------- 图

def fig_episodes(px, tax):
    """三个历史片段下各子类的表现 —— 说明「加息本身」不是决定因素，油价方向才是。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    PLANE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
    # 槽位顺序即 CVD 安全机制（已验证 adjacent 全过）
    EP = [("2022H1 加息+油价涨", "2022-01-03", "2022-06-30", "#2a78d6"),
          ("2018Q4 加息+油价崩", "2018-09-28", "2018-12-24", "#eb6834"),
          ("2022 全年", "2022-01-03", "2022-12-30", "#1baf7a")]

    subs, vals = [], {}
    for lab, a, b, c in EP:
        seg = px.loc[a:b]
        d = {}
        for s in tax["功能子类"].unique():
            ks = [x for x in px.columns if tax.loc[x, "功能子类"] == s]
            d[s] = (seg[ks].iloc[-1] / seg[ks].iloc[0] - 1).mean()
        vals[lab] = d
        subs = list(d)
    # ⚠ 剔除三个片段全无数据的子类（只有加密：IBIT 2024-04 才上市）。
    #   留着会变成一行空柱，而空行的标签会贴在上一行的柱旁，被误读成有数据。
    dropped = [s for s in subs if all(pd.isna(vals[lab][s]) for lab, *_ in EP)]
    subs = [s for s in subs if s not in dropped]
    order = sorted(subs, key=lambda s: vals[EP[0][0]][s])

    fig, ax = plt.subplots(figsize=(12.4, 8.6), dpi=150)
    fig.patch.set_facecolor(PLANE)
    ax.set_facecolor(PLANE)
    y = np.arange(len(order))
    h = 0.26
    for k, (lab, a, b, c) in enumerate(EP):
        off = (k - 1) * h
        v = [vals[lab][s] for s in order]
        ax.barh(y + off, v, height=h * 0.88, color=c, zorder=3, label=lab)
        for yi, vi in zip(y + off, v):
            ax.text(vi + (0.012 if vi >= 0 else -0.012), yi, f"{vi:+.0%}",
                    va="center", ha="left" if vi >= 0 else "right",
                    fontsize=7.4, color=INK2, zorder=4)
    ax.axvline(0, color="#c9c8c3", lw=1.4, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=10, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8.5, colors="#8a8880", length=0)
    ax.set_xlim(-0.65, 0.65)
    ax.xaxis.set_major_formatter(lambda t, _: f"{t:.0%}")
    ax.grid(axis="x", color="#e6e5e1", lw=1, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#dcdbd6")
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK2)

    fig.suptitle("同样是加息：油价涨则能源称王，油价崩则能源垫底",
                 fontsize=15, color=INK, x=0.045, ha="left", y=0.972, fontweight="bold")
    note = (f"（{ '、'.join(dropped) } 因上市太晚、这三个片段无数据，未列入）"
            if dropped else "")
    fig.text(0.045, 0.930,
             "各功能子类等权收益。2022H1（俄乌+加息启动）与 2018Q4（加息+油价崩）"
             "都是「加息 + 油价冲击」，结果却完全相反 ——\n"
             "决定能源股命运的是油价方向，不是加息本身。当前是美伊冲突推油价破 100，"
             "更接近 2022H1 那一侧。" + note,
             fontsize=9.8, color=INK2, ha="left", va="top")
    fig.subplots_adjust(left=0.105, right=0.965, top=0.855, bottom=0.055)
    path = os.path.join(OUT_FIG, "F_rate_episodes.png")
    fig.savefig(path, facecolor=PLANE)
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT_CSV, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)
    p("=" * 104)
    p("全球资产配置模拟 —— 3b：加息/紧缩冲击下的资产表现")
    p("=" * 104)

    px, meta, tax = load()
    A, shock, pos = part_a(px, tax)
    B = part_b(px)
    part_c(px, tax)
    part_d(px, tax)

    A.to_csv(os.path.join(OUT_CSV, "F_rateshock_subclass.csv"),
             index=False, encoding="utf-8-sig")
    B.to_frame("2022加息周期收益").to_csv(
        os.path.join(OUT_CSV, "F_hiking_cycle_assets.csv"), encoding="utf-8-sig")
    f = fig_episodes(px, tax)

    p("")
    p("=" * 104)
    p("  ⚠ Part A 的有效 t 全部 < 1（重叠窗口已折算）—— 方向可参考，不能当显著证据。")
    p("    真正可靠的是 Part B/C 的长区间对比。")
    p(f"  图  {f}")
    p("=" * 104)


if __name__ == "__main__":
    main()
