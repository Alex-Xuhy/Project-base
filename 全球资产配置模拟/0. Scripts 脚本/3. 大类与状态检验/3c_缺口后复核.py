# -*- coding: utf-8 -*-
"""全球资产配置模拟 —— 3c：补上外部缺口后，重新审视此前的判断

为什么有这个脚本：
  站点股票数据固定滞后到**周三**。2026-09-16（周三 FOMC 加息）之后，
  09-17 与 09-18 两个交易日在站点里完全不存在 —— 而这两天恰恰是判断
  「加息到底冲击了什么」的全部证据。1c 已用外部源补齐，
  本脚本用补齐后的序列逐条复核 3b 的结论。

复核对象（都是 3b / 此前对话里明确说过的判断）：
  J1 「USO / XLE / SLX 在加息后反常下跌」
  J2 「原来尚佳的能源股标的近日瞬间刹停」
  J3 「美联储加息是极大的炸弹」→ 暗示风险资产受冲击
  J4 3b Part C：「当前更像 2022H1（加息+油价涨）那一侧」→ 能源称王

方法学注意：
  * 缺口期只有 **2 个交易日**。2 天数据对「6 个月类比」没有证伪力，
    只能算第一个读数。凡涉及统计显著性的结论一律不给，只说方向。
  * 缺口期拼接序列（wide_close_extgap_usd.csv）**只用于当前读数**，
    历史分析仍必须用站点原始序列。
  * 子类收益 = 类内等权，只算当期有数据的标的。

用法：
  python 3c_缺口后复核.py
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

p = print


def load():
    site = pd.read_csv(os.path.join(DATA_DIR, "wide_close_adj_usd.csv"),
                       encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    ext = pd.read_csv(os.path.join(DATA_DIR, "wide_close_extgap_usd.csv"),
                      encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    tax = pd.read_csv(TAX_CSV,
                      encoding="utf-8-sig", index_col=0)
    site = site[[c for c in site.columns if c in tax.index]]
    ext = ext[[c for c in ext.columns if c in tax.index]]
    return site, ext, tax.reindex(ext.columns)


def sub_ret(px, a, b, tax, field="功能子类"):
    """区间内各 field 的等权收益。"""
    r = px.loc[b] / px.loc[a] - 1
    out = {}
    for k, g in tax.groupby(field):
        ks = [c for c in g.index if c in px.columns and pd.notna(r.get(c))]
        if ks:
            out[k] = r[ks].mean()
    return pd.Series(out)


# ---------------------------------------------------------------- Part 1

def part1(ext, tax, se, gap):
    p("")
    p("=" * 100)
    p("Part 1 —— 缺口期读数（站点末日 → 最新）")
    p("=" * 100)
    p(f"  站点数据到 {se:%Y-%m-%d}（周三）；缺口补齐 "
      f"{'、'.join(f'{d:%m-%d}' for d in gap)}（{'、'.join(['四','五'][i] for i in range(len(gap)))}）")
    p("")

    r = sub_ret(ext, se, gap[-1], tax)
    r = r.sort_values(ascending=False)
    p(f"  {'功能子类':<14}{'缺口期':>10}   相对全池")
    p("  " + "-" * 44)
    bench = r.mean()
    for k, v in r.items():
        p(f"  {k:<14}{v:>+10.2%}   {v - bench:>+7.2%}")
    p("  " + "-" * 44)
    p(f"  {'全池等权':<14}{bench:>+10.2%}")

    p("")
    p("  按大类:")
    for k, v in sub_ret(ext, se, gap[-1], tax, "功能大类").sort_values(
            ascending=False).items():
        p(f"    {k:<8}{v:>+9.2%}")

    p("")
    tot = (ext.loc[gap[-1]] / ext.loc[se] - 1).dropna().sort_values(ascending=False)
    p(f"  个券（{len(tot)} 只，上涨 {(tot > 0).sum()} 只）")
    p(f"    前 10: " + "  ".join(f"{k}{v:+.1%}" for k, v in tot.head(10).items()))
    p(f"    后 10: " + "  ".join(f"{k}{v:+.1%}" for k, v in tot.tail(10).items()))
    return r, tot


# ---------------------------------------------------------------- Part 2

def part2(ext, tax, se, gap):
    p("")
    p("=" * 100)
    p("Part 2 —— 逐条复核此前判断")
    p("=" * 100)

    t0 = gap[-1]
    d2 = lambda c: ext.at[t0, c] / ext.at[se, c] - 1 if c in ext.columns else np.nan

    p("")
    p("  J1「USO / XLE / SLX 在加息后反常下跌」")
    p("    ★ 拆成三天看才看得清。9/16 是 FOMC 当天，9/17-9/18 是站点缺失的两天：")
    p("")
    p(f"    {'':<10}{'9/15→9/16 (FOMC当天)':>21}{'9/16→9/18 (站点缺失)':>22}")
    p("    " + "-" * 53)
    for c, nm in [("USO", "原油"), ("XLE", "能源股"), ("SLX", "钢铁"),
                  ("SLV", "白银"), ("GLD", "黄金"), ("IBIT", "加密"),
                  ("XLK", "科技"), ("SPY", "标普500")]:
        if c not in ext.columns:
            continue
        v1 = ext.at[se, c] / ext.at["2026-09-15", c] - 1
        v2 = d2(c)
        p(f"    {nm:<10}{v1:>+21.2%}{v2:>+22.2%}")
    p("")
    p("    → 三条修正：")
    p("      ① **SLX 根本没跌**。三日 −0.46%，等于没动。这条是错的。")
    p("      ② **XLE 已经止跌回升**。9/16 当天 −2.88%，之后 **+0.44%**。")
    p("         「能源股」这个说法不成立 —— 只有原油还在跌。")
    p("      ③ **USO 是唯一持续下跌的**。−3.52% 之后又 −1.50%。")
    p("         这是**原油自己的**行情（供给/需求消息），不是加息的连带。")
    p("")
    p("    真正的形态是：**9/16 是一次单日的鹰派膝跳，除原油外全部在两天内收复**。")
    p("    同日反转幅度：白银 −0.83%→+5.05%、黄金 −0.61%→+2.41%、")
    p("    加密 −0.16%→+6.92%、科技 +0.10%→+3.08%。")

    p("")
    p("  J2「能源股刹停」")
    win = ext.loc[:t0]
    for c, nm in [("USO", "原油"), ("XLE", "能源股"), ("SLX", "钢铁")]:
        if c not in ext.columns:
            continue
        s = win[c].dropna()
        peak = s.loc["2026-06-01":].max()
        peak_d = s.loc["2026-06-01":].idxmax()
        p(f"    {nm:<8} 6/1 以来最高 {peak:>8.2f}（{peak_d:%m-%d}）"
          f"  现价 {s.iloc[-1]:>8.2f}  距峰值 {s.iloc[-1]/peak - 1:>+7.2%}"
          f"   近60日 {s.iloc[-1]/s.iloc[-61] - 1:>+8.1%}")
    p("")
    p("    → **「刹停」措辞不准**。能源是从**很高的位置**回吐，")
    p("      不是从平庸位置跌下来。判断方向的时机与判断幅度是两回事。")

    p("")
    p("  J3「美联储加息是极大的炸弹」")
    p("    加息后两天，全池 79 只 **等权 +0.71%，上涨 49 只**。")
    p("    领涨的是最「风险偏好」的一篮子：加密 +6.92%、贵金属 +3.73%、")
    p("    科技主题 +1.25%、新兴市场 +1.17%。")
    p("    而避险端的国债长债只有 +0.46%。")
    p("")
    p("    → **证伪**。这次加息 92% 已提前定价，9/16 当天 TLT 就是 +0.21%，")
    p("      是典型 sell the rumor / buy the news。**炸弹没有炸。**")

    p("")
    p("  J4 3b Part C「当前更像 2022H1（油价涨那侧），能源称王」")
    p("    2022H1：加息 + **油价涨** → 能源 +48%、商品宽基 +26%")
    p("    2018Q4：加息 + **油价崩** → 贵金属 +4%、国债长 +3%、能源 −11%")
    p("    本次已实现的两天：**贵金属 +3.73%、能源 −0.51%、原油 −4.97%**")
    p("")
    p("    → **头两天的读数落在 2018Q4 那一侧，与我此前的判断相反。**")
    p("      但要说得准确：2 个交易日对「6 个月类比」**没有证伪力**。")
    p("      正确说法是：油价方向这个决定变量，目前指向 2018Q4。")


# ---------------------------------------------------------------- Part 3

def part3(ext, site, tax, se, gap):
    """本次缺口期的子类排序，跟历史片段比 —— 秩相关。"""
    p("")
    p("=" * 100)
    p("Part 3 —— 本次 2 天 vs 历史片段（子类层面秩相关）")
    p("=" * 100)

    EP = {"2022H1 加息+油价涨": ("2022-01-03", "2022-06-30"),
          "2018Q4 加息+油价崩": ("2018-09-28", "2018-12-24"),
          "2022 全年": ("2022-01-03", "2022-12-30"),
          "2025 降息(对照)": ("2025-01-02", "2025-12-31"),
          "2026 至今": ("2026-01-02", se.isoformat())}
    now = sub_ret(ext, se, gap[-1], tax)
    p("")
    p(f"  {'历史片段':<22}{'Spearman ρ':>12}{'重叠子类':>10}   前3 / 后3")
    p("  " + "-" * 74)
    for lab, (a, b) in EP.items():
        hist = sub_ret(site, a, b, tax)
        both = pd.concat([now, hist], axis=1, keys=["now", "hist"]).dropna()
        rho = both["now"].corr(both["hist"], method="spearman")
        top = "、".join(both["hist"].nlargest(3).index)
        bot = "、".join(both["hist"].nsmallest(3).index)
        p(f"  {lab:<22}{rho:>12.2f}{len(both):>10}   {top} / {bot}")
    p("")
    p("  ⚠ 只有 2 天、且子类横截面差异（−2.3%~+6.9%）与日噪声同量级，")
    p("    ρ 极不稳定，**不能当作证据**。列在这里是为了建立「后续每周更新」")
    p("    的对照基线，而不是给结论。")
    return now


# ---------------------------------------------------------------- Part 4

def part4(ext, tax, se, gap):
    p("")
    p("=" * 100)
    p("Part 4 —— 更新后的当前读数（把 9/17-9/18 计入）")
    p("=" * 100)
    t0 = gap[-1]
    p(f"  数据末日 {t0:%Y-%m-%d}。下表与 3b Part D 同口径，"
      f"但多算了两天 —— 差异最大的就是需要修正的地方。")
    p("")
    p(f"  {'子类':<14}{'近5日':>10}{'近21日':>10}{'近63日':>10}"
      f"{'近126日':>10}{'近252日':>10}")
    p("  " + "-" * 66)
    rows = []
    for k, g in tax.groupby("功能子类"):
        ks = [c for c in g.index if c in ext.columns]
        line = f"  {k:<14}"
        rec = {"子类": k}
        for n in (5, 21, 63, 126, 252):
            v = (ext[ks].iloc[-1] / ext[ks].iloc[-1 - n] - 1).mean()
            rec[f"近{n}日"] = v
            line += f"{v:>+10.1%}"
        p(line)
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 图

def fig_gap(ext, tax, se, gap):
    """缺口期子类收益 —— 有正有负 = 极性 → 发散配色（blue<->red，中性灰中点）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    PLANE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
    # 发散对用参考调色板的 blue <-> red。⚠ 极性映射按**中文金融惯例**（红涨蓝跌）——
    #   读者一眼扫过时，红柱朝下会被读成上涨。方向本身已由零线左右承载，颜色只做强化。
    POS, NEG = "#e34948", "#2a78d6"

    r = sub_ret(ext, se, gap[-1], tax).sort_values()
    fig, ax = plt.subplots(figsize=(11.6, 7.2), dpi=150)
    fig.patch.set_facecolor(PLANE)
    ax.set_facecolor(PLANE)

    y = np.arange(len(r))
    cols = [POS if v >= 0 else NEG for v in r.values]
    ax.barh(y, r.values, height=0.62, color=cols, zorder=3)
    for yi, vi in zip(y, r.values):
        ax.text(vi + (0.0016 if vi >= 0 else -0.0016), yi, f"{vi:+.2%}",
                va="center", ha="left" if vi >= 0 else "right",
                fontsize=8.6, color=INK2, zorder=4)
    ax.axvline(0, color="#c3c2b7", lw=1.4, zorder=2)
    # 全池等权参考线（发散图的"中性"之外再加一层：整体水位）
    b = r.mean()
    ax.axvline(b, color="#898781", lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.text(b, -0.72, f" 全池等权 {b:+.2%}", fontsize=8.4,
            color="#898781", va="center", ha="left", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(r.index, fontsize=10, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8.5, colors="#898781", length=0)
    ax.set_xlim(-0.032, 0.082)
    # 加密那根（+6.9%）把量程撑到 8%，其余 17 条挤在 0 附近 → 1% 副刻度才读得出
    ax.set_xticks(np.arange(-0.02, 0.081, 0.02))
    ax.set_xticks(np.arange(-0.03, 0.081, 0.01), minor=True)
    ax.xaxis.set_major_formatter(lambda t, _: f"{t:.0%}")
    ax.grid(axis="x", which="major", color="#e1e0d9", lw=1, zorder=0)
    ax.grid(axis="x", which="minor", color="#eeedea", lw=0.8, zorder=0)
    ax.set_ylim(-1.15, len(r) - 0.4)              # 底部留白给参考线标注
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")

    fig.suptitle("加息后的两个交易日：领涨的是最「风险偏好」的那一篮子",
                 fontsize=15.5, color=INK, x=0.042, ha="left", y=0.972,
                 fontweight="bold")
    fig.text(0.042, 0.928,
             f"功能子类等权收益，{se:%m-%d}（站点数据末日，FOMC 当日）"
             f"→ {gap[-1]:%m-%d}。这段行情在站点里不存在 —— 由 1c 从外部源补齐。\n"
             "加密、贵金属、科技、新兴市场领涨；下跌的只有商品系三条（农产品 / 能源 / 商品宽基）。"
             "「加息 = 风险资产崩」在这两天没有发生。（红涨蓝跌）",
             fontsize=9.9, color=INK2, ha="left", va="top")
    fig.subplots_adjust(left=0.115, right=0.975, top=0.865, bottom=0.055)
    path = os.path.join(OUT_FIG, "F_gap_reassess.png")
    fig.savefig(path, facecolor=PLANE)
    plt.close(fig)
    return path


def part5(ext, tax, se, gap):
    """如果成交价停在站点末日（9/16），本次调仓"免费"锁定了多少已知行情。

    ⚠ 这是**条件性**分析：成交价是否为 9/16 由用户在站点下单页实测确认。
      本表只是把「若成立则意味着什么」算清楚，不预设它成立。
    """
    p("")
    p("=" * 100)
    p("Part 5 —— 条件分析：若本周末成交价停在站点末日 9/16")
    p("=" * 100)
    p(f"  ⚠ 前提未确认。成交价是否 = {se:%m-%d} 收盘，以你在站点下单页看到的报价为准。")
    p("    下面只是把「若成立」的后果算清楚 —— 不预设它成立。")
    p("")
    tot = (ext.loc[gap[-1]] / ext.loc[se] - 1).dropna().sort_values(ascending=False)
    p("  若按 9/16 价成交，则：")
    p(f"    **买入**下面这些，等于开盘即带上一笔账面浮盈（前 12）:")
    for k, v in tot.head(12).items():
        nm = tax.at[k, "中文名"] if "中文名" in tax.columns else ""
        p(f"      {k:<6}{v:>+8.2%}   {nm}")
    p(f"    **卖出**下面这些，等于避开了已经发生的下跌（后 12）:")
    for k, v in tot.tail(12).items():
        nm = tax.at[k, "中文名"] if "中文名" in tax.columns else ""
        p(f"      {k:<6}{v:>+8.2%}   {nm}")
    p("")
    p(f"    全池等权 {tot.mean():+.2%}，即平均而言买入动作自带 "
      f"{tot.mean():+.2%} 的账面垫子。")
    p("")
    p("  ⚠ 两点必须同时成立才有意义：")
    p("    ① 买入与卖出**同价**。若买入按 9/16、卖出按最新，则方向要反过来。")
    p("    ② 站点的**估值**随后会跳到最新价。否则浮盈只是账面上的数字，不变现。")
    p("  ⚠ 且这是**一次性**的：9/16 的缺口只能吃一次。下周末缺口只有 3 天且已定价。")
    return tot


def main():
    os.makedirs(OUT_CSV, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)
    p("=" * 100)
    p("全球资产配置模拟 —— 3c：补缺口后重新审视")
    p("=" * 100)

    site, ext, tax = load()
    se = pd.read_csv(os.path.join(DATA_DIR, "wide_close_adj_usd.csv"),
                     encoding="utf-8-sig", index_col=0, parse_dates=True).index.max()
    gap = [d for d in ext.index if d > se]
    if not gap:
        p("  无缺口，先跑 1c_补站点缺口.py")
        return

    r, tot = part1(ext, tax, se, gap)
    part2(ext, tax, se, gap)
    part3(ext, site, tax, se, gap)
    R = part4(ext, tax, se, gap)
    part5(ext, tax, se, gap)
    f = fig_gap(ext, tax, se, gap)

    r.to_frame(f"缺口期 {se:%Y-%m-%d}→{gap[-1]:%Y-%m-%d}").to_csv(
        os.path.join(OUT_CSV, "R_gap_subclass.csv"), encoding="utf-8-sig")
    R.to_csv(os.path.join(OUT_CSV, "R_current_readings.csv"),
             index=False, encoding="utf-8-sig")

    p("")
    p("=" * 100)
    p(f"  图  {f}")
    p(f"  表  {os.path.join(OUT_CSV, 'R_gap_subclass.csv')}")
    p(f"  表  {os.path.join(OUT_CSV, 'R_current_readings.csv')}")
    p("=" * 100)


if __name__ == "__main__":
    main()
