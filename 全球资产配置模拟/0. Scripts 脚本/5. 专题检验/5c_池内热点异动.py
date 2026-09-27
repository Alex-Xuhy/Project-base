# -*- coding: utf-8 -*-
"""全球资产配置模拟 —— 5c：池内热点与成交额异动

回答用户 2026-09-21：「近期美国市场的投资热点/机构关注/资金流向（避险）
主要在哪些板块？」

★ 口径警告（必须先说清楚，否则整篇都是错的）：

  站点只给 **成交量（股数）**，没有 ETF 份额（shares outstanding）历史，
  因此 **算不出真正的「资金净流入」**（ETF 的净流入 = 份额变化 × 净值，
  需要 creation/redemption 数据）。

  能算的是 **成交额异动**：近期成交额 / 历史基准成交额。
  它是「关注度 / 换手 / 参与度」的代理，**方向必须配合价格变动读**：
    放量 + 上涨 → 增量买盘
    放量 + 下跌 → 增量卖盘
    缩量 + 上涨 → 无人接力的上涨
  下文一律用「成交额异动」，不写「净流入」。

  外部的机构流向数据（BofA 客户流、LSEG Lipper 周度基金流、
  Goldman 对冲基金持仓）只能从公开新闻取得，不在本脚本里，
  见 README 第十三节 13.4。

数据：
  * 价格：wide_close_extgap_usd.csv（站点历史 + 外部补齐，到 2026-09-18）
  * 成交量：prices_long.csv（站点，到 2026-09-16）
           + external_gap_volume.csv（外部，2025-09-18 ~ 2026-09-18）
  ⚠ 成交量是**未复权**的股数。若窗口内发生拆股，比值会失真 ——
    脚本会打印最近 90 天的拆股事件做守卫（目前最近的两次是
    2025-12-05 的 XLK/XLY/XLE/XLU/XLB，已在 60 日基准窗口之外）。
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

# 成交额异动的窗口（交易日）
V_RECENT, V_BASE = 5, 60
# 动量窗口（交易日）
W1, W4, W13 = 5, 20, 65

SAFE = ["国债短", "国债中长", "国债长", "债宽基", "信用债", "美元", "贵金属"]
RISK = ["美股宽基", "美股行业", "科技主题股", "发达市场", "新兴市场",
        "加密", "资源主题股", "能源", "商品宽基", "农产品"]
OTHER = ["其他货币"]

p = print


def load():
    px = pd.read_csv(os.path.join(DATA_DIR, "wide_close_extgap_usd.csv"),
                     encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    lg = pd.read_csv(os.path.join(DATA_DIR, "prices_long.csv"),
                     encoding="utf-8-sig", parse_dates=["dataDatetime"])
    vol = lg.pivot_table(index="dataDatetime", columns="symbol",
                         values="volume", aggfunc="last").sort_index()
    gv = pd.read_csv(os.path.join(DATA_DIR, "external_gap_volume.csv"),
                     encoding="utf-8-sig", parse_dates=True, index_col=0)
    if "symbol" in gv.columns:                     # 长表
        gv = gv.pivot_table(index=gv.index, columns="symbol",
                            values="volume", aggfunc="last").sort_index()
    new = [d for d in gv.index if d > vol.index.max()]
    if new:
        vol = pd.concat([vol, gv.loc[new]])
    vol = vol[~vol.index.duplicated(keep="last")].sort_index()
    return px, vol


def heat(px, vol, tax, syms):
    """逐只：动量 + 成交额异动。"""
    rows = []
    v = vol.dropna(how="all")
    dv = (v * px.reindex(v.index)).dropna(how="all")   # 美元成交额
    for s in syms:
        if s not in px.columns or s not in dv.columns:
            continue
        q = px[s].dropna()
        d = dv[s].dropna()
        if len(q) < W13 + 5 or len(d) < V_BASE + V_RECENT:
            continue
        rec = {"标的": s, "中文名": tax.at[s, "中文名"],
               "功能子类": tax.at[s, "功能子类"]}
        for lab, w in [("1周", W1), ("4周", W4), ("13周", W13)]:
            rec[lab] = q.iloc[-1] / q.iloc[-1 - w] - 1
        rec["距52周高"] = q.iloc[-1] / q.tail(252).max() - 1
        base = d.iloc[-(V_BASE + V_RECENT):-V_RECENT].mean()
        rec["成交额异动"] = d.iloc[-V_RECENT:].mean() / base if base > 0 else np.nan
        rec["日均成交额$M"] = base / 1e6
        rows.append(rec)
    return pd.DataFrame(rows).set_index("标的")


# ---------------------------------------------------------------- Part A

def part_a(D):
    p("")
    p("=" * 100)
    p("Part A —— 热点在哪：按功能子类看近期动量")
    p("=" * 100)
    g = D.groupby("功能子类").agg(
        只数=("1周", "size"), 周1=("1周", "median"), 周4=("4周", "median"),
        周13=("13周", "median"), 成交额异动=("成交额异动", "median"))
    g = g[g["只数"] >= 1].sort_values("周4", ascending=False)
    p(f"  {'功能子类':<10}{'只数':>5}{'近1周':>9}{'近4周':>9}{'近13周':>9}"
      f"{'距52周高':>10}{'成交额异动':>11}")
    p("  " + "-" * 63)
    for k, r in g.iterrows():
        dd = D[D["功能子类"] == k]["距52周高"].median()
        p(f"  {k:<10}{int(r['只数']):>5}{r['周1']:>+9.2%}{r['周4']:>+9.2%}"
          f"{r['周13']:>+9.2%}{dd:>+10.2%}{r['成交额异动']:>10.2f}x")
    p("")
    return g


# ---------------------------------------------------------------- Part B

def part_b(D):
    p("")
    p("=" * 100)
    p("Part B —— 成交额异动榜（近 5 日均额 ÷ 前 60 日均额）")
    p("=" * 100)
    p("  ⚠ 这是**关注度**，不是净流入。方向要看同列的动量：")
    p("    放量+涨 = 增量买盘；放量+跌 = 增量卖盘；缩量+涨 = 无人接力。")
    p("")
    X = D.sort_values("成交额异动", ascending=False)
    p(f"  {'标的':<6}{'中文名':<16}{'功能子类':<10}{'成交额异动':>11}"
      f"{'近1周':>9}{'近4周':>9}{'日均额$M':>10}   解读")
    p("  " + "-" * 92)
    for lab, sub in [("▲ 放量前列", X.head(12)), ("▼ 缩量前列", X.tail(6))]:
        p(f"  {lab}")
        for s, r in sub.iterrows():
            note = ("增量买盘" if r["4周"] > 0.02 else
                    "增量卖盘" if r["4周"] < -0.02 else "方向不明")
            p(f"  {s:<6}{r['中文名']:<16}{r['功能子类']:<10}"
              f"{r['成交额异动']:>10.2f}x{r['1周']:>+9.2%}{r['4周']:>+9.2%}"
              f"{r['日均成交额$M']:>10.0f}   {note}")
        p("")
    return X


# ---------------------------------------------------------------- Part C

def part_c(D, tax):
    p("")
    p("=" * 100)
    p("Part C —— 避险 vs 风险：钱究竟往哪边挪")
    p("=" * 100)
    grp = {}
    for s in D.index:
        k = tax.at[s, "功能子类"]
        grp[s] = "避险" if k in SAFE else "其他货币" if k in OTHER else "风险"
    D = D.assign(组=[grp[s] for s in D.index])
    p(f"  {'组':<8}{'只数':>5}{'近1周':>10}{'近4周':>10}{'近13周':>10}"
      f"{'距52周高':>11}{'成交额异动':>12}")
    p("  " + "-" * 66)
    for k in ["避险", "风险", "其他货币"]:
        sub = D[D["组"] == k]
        if not len(sub):
            continue
        p(f"  {k:<8}{len(sub):>5}{sub['1周'].median():>+10.2%}"
          f"{sub['4周'].median():>+10.2%}{sub['13周'].median():>+10.2%}"
          f"{sub['距52周高'].median():>+11.2%}{sub['成交额异动'].median():>11.2f}x")
    p("")
    p("  避险组逐只：")
    for s, r in D[D["组"] == "避险"].sort_values("4周", ascending=False).iterrows():
        p(f"    {s:<6}{r['中文名']:<18}{'近4周':>8}{r['4周']:>+8.2%}"
          f"{'  近13周':>9}{r['13周']:>+8.2%}{'  成交额异动':>11}{r['成交额异动']:>6.2f}x")
    p("")
    return D


# ---------------------------------------------------------------- 图

def fig(D, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    SURF, INK, INK2, MUT = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    GRID, BASE = "#e1e0d9", "#c3c2b7"
    S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # 槽位 1-4（已过 validator）
    V = ["国债短", "国债中长", "国债长", "债宽基", "信用债",
         "贵金属", "美元", "其他货币"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.4, 6.4), facecolor=SURF,
        gridspec_kw={"width_ratios": [1.25, 1.0]})
    for ax in (ax1, ax2):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(BASE)
        ax.tick_params(colors=MUT, labelsize=9.5, length=0)
        ax.grid(color=GRID, lw=0.7)
        ax.set_axisbelow(True)

    # ---- 左：13 周动量 vs 成交额异动 ----
    # 79 个点全标会糊成一团（大盘股全挤在 x≈0），所以只标 curated 一组，
    # 每个标签给手工偏移 —— 自动排布在这个密度下救不回来。
    ax1.axhline(1.0, color=BASE, lw=1.0, zorder=1)
    ax1.axvline(0, color=BASE, lw=1.0, zorder=1)
    sub = D[D["日均成交额$M"] >= 20]
    for s, r in sub.iterrows():
        big = r["日均成交额$M"] >= 150
        ax1.scatter(r["13周"], r["成交额异动"], s=140 if big else 42,
                    color=S[0] if big else MUT, alpha=0.9 if big else 0.45,
                    edgecolor=SURF, lw=1.4, zorder=3)
    LAB = {"USO": (-30, 7), "IBIT": (-33, 5), "HACK": (-36, 5), "XLE": (-30, -9),
           "XBI": (8, 3), "EWZ": (6, 9), "EWT": (7, -4), "DBC": (-31, -4),
           "UNG": (9, 3), "SHY": (-30, 6), "TLT": (9, 2), "IEF": (-30, -3),
           "LQD": (8, -9), "HYG": (8, 3), "GLD": (8, -2), "XLV": (8, -2),
           "XHB": (9, 3), "XLU": (9, -8), "XLI": (-28, -7), "XLK": (8, 1)}
    for s, (dx, dy) in LAB.items():
        if s not in sub.index:
            continue
        r = sub.loc[s]
        ax1.annotate(s, (r["13周"], r["成交额异动"]),
                     textcoords="offset points", xytext=(dx, dy),
                     fontsize=9.5, color=INK2, zorder=4,
                     ha="right" if dx < 0 else "left")
    ax1.set_xlabel("近 13 周收益", fontsize=10.5, color=INK2)
    ax1.set_ylabel("成交额异动（近 5 日 ÷ 前 60 日）", fontsize=10.5, color=INK2)
    ax1.set_title("热度地图：右上=已涨且放量（原油/比特币/网安），"
                  "左上=放量但没涨（国债）",
                  fontsize=12, color=INK, pad=12, loc="left")
    ax1.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}x")
    ax1.set_xlim(-0.20, 0.42)
    ax1.set_ylim(0.3, 2.8)
    ax1.legend(handles=[ax1.scatter([], [], s=140, color=S[0], label="日均额 ≥ $150M"),
                        ax1.scatter([], [], s=42, color=MUT, alpha=.45, label="$20M–150M")],
               loc="lower right", frameon=False, fontsize=10, labelcolor=INK2)

    # ---- 右：避险子类 vs 全池 ----
    ax2.axvline(0, color=BASE, lw=1.0, zorder=1)
    rows = []
    for k in V:
        s2 = D[D["功能子类"] == k]
        if len(s2):
            rows.append((k, s2["4周"].median(), s2["13周"].median(), len(s2)))
    rows.sort(key=lambda r: r[1])
    ys = np.arange(len(rows))
    for i, (k, a, b, n) in enumerate(rows):
        ax2.plot([a, b], [i, i], color=GRID, lw=1.6, zorder=1)
        ax2.scatter(a, i, s=110, color=S[0], edgecolor=SURF, lw=1.4,
                    zorder=3, label="近 4 周" if i == 0 else None)
        ax2.scatter(b, i, s=110, color=S[1], edgecolor=SURF, lw=1.4,
                    zorder=3, label="近 13 周" if i == 0 else None)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([f"{k} ({n})" for k, _, _, n in rows], fontsize=10)
    ax2.set_xlabel("中位收益", fontsize=10.5, color=INK2)
    ax2.set_title("避险类资产：只有美元为正，国债短端强于长端", fontsize=12,
                  color=INK, pad=12, loc="left")
    ax2.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax2.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK2)

    # 注意：标题里不要放 ⚠ / ❗ 这类符号 —— Microsoft YaHei 没有这些字形，
    # 渲染出来是豆腐块。用「注：」代替。
    fig.suptitle("池内热度与成交额异动（价格与成交额均到 2026-09-18）"
                 "　注：成交额异动 ≠ 资金净流入",
                 fontsize=13.5, color=INK, x=0.007, ha="left", y=0.982)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)
    p(f"  图  {os.path.basename(out)}")


def main():
    os.makedirs(OUT_CSV, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)
    p("=" * 100)
    p("全球资产配置模拟 —— 5c：池内热点与成交额异动")
    p("=" * 100)
    px, vol = load()
    tax = pd.read_csv(TAX_CSV,
                      encoding="utf-8-sig", index_col=0)

    sp = pd.read_csv(os.path.join(DATA_DIR, "splits.csv"), encoding="utf-8-sig",
                     parse_dates=["date"])
    cut = px.index.max() - pd.Timedelta(days=90)
    recent = sp[sp["date"] >= cut]
    p(f"  价格 {px.index.min().date()} ~ {px.index.max().date()}"
      f"（{px.shape[1]} 列）；成交额到 {vol.index.max().date()}")
    p(f"  拆股守卫：近 90 天内有 {len(recent)} 起"
      + (f" —— {', '.join(recent['symbol'])}（成交额比值可能失真）" if len(recent)
         else "，成交量窗口干净"))
    p("")

    D = heat(px, vol, tax, list(tax.index))
    p(f"  纳入 {len(D)} 只（需同时有 ≥{W13+5} 日价格与 ≥{V_BASE+V_RECENT} 日成交额）")
    if len(D) < len(tax):
        miss = [s for s in tax.index if s not in D.index]
        p(f"  剔除：{', '.join(miss)}")

    g = part_a(D)
    X = part_b(D)
    D2 = part_c(D, tax)
    fig(D2, os.path.join(OUT_FIG, "M_market_heat.png"))

    D2.sort_values("成交额异动", ascending=False).to_csv(
        os.path.join(OUT_CSV, "M_heat_all.csv"), encoding="utf-8-sig")
    g.to_csv(os.path.join(OUT_CSV, "M_heat_by_subclass.csv"), encoding="utf-8-sig")
    p("  表  M_heat_all.csv / M_heat_by_subclass.csv")
    p("=" * 100)


if __name__ == "__main__":
    main()
