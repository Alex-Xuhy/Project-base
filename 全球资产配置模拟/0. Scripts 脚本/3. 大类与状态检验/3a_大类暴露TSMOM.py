# -*- coding: utf-8 -*-
"""全球资产配置模拟 —— 3a：大类暴露与大类时序动量

赛期约束（决定了本脚本的全部口径，勿改）：
  * 比赛真实期限 = 2 个月，不是站点项目页写的 10 个月
  * 只能在周末调仓 → 2 个月只有 8-9 次调仓
  * **无交易成本** → 周频调仓免费，换手约束不适用
  ⇒ 评估一律用「所有历史 8 周窗口的收益分布」，不用年化
  ⇒ 单次比赛信噪比极低，故重点看 中位 / 5%分位 / 最差 / 跑赢基准占比

三部分：
  A. 大类暴露 —— 四个大类在 8 周窗口上的风险收益，以及各类之间的分散效果
  B. 大类时序动量（TSMOM）—— 用大类自己的历史收益决定持有与否
  C. 组合对比 + 当前读数（给出对本次比赛的可执行姿态）

方法学注意：
  * 8 周窗口逐周重叠 → 名义 n≈700，**有效独立窗口只有约 88 个**。
    所有 t 值给两栏：名义 t（严重高估）与有效 t（已按 √重叠倍数 折算）。
  * 大类收益 = 类内等权（只用当期有数据的标的），避免被单只 ETF 主导。

用法：
  python 3a_大类暴露TSMOM.py
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

GAME_WEEKS = 8          # 2 个月 ≈ 8 个周末
CLASSES = ["Equity", "FixedIncome", "Commodity", "Currency"]
CN = {"Equity": "股权", "FixedIncome": "固收", "Commodity": "商品", "Currency": "货币"}

p = print


# ---------------------------------------------------------------- 载入

def load():
    px = pd.read_csv(os.path.join(DATA_DIR, "wide_close_adj_usd.csv"),
                     encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    meta = pd.read_csv(os.path.join(DATA_DIR, "stocks.csv"),
                       encoding="utf-8-sig").set_index("symbol")
    last_valid = px.apply(lambda s: s.last_valid_index())
    keep = [c for c in px.columns
            if last_valid[c] >= px.index.max() - pd.Timedelta(days=10)]
    return px[keep], meta.loc[keep]


def week_ends(d):
    s = pd.Series(d.index, index=d.index).resample("W-FRI").max().dropna()
    return [pd.Timestamp(t) for t in s.tolist()]


def to_weekly(obj, weekly):
    """把**日频**复合到周频。⚠ window_dist 假定输入已是周频 ——
    日频直接喂进去会把「8 周」算成「8 天」，务必先过这个函数。"""
    rows, idx = [], []
    for j in range(len(weekly) - 1):
        t, t1 = weekly[j], weekly[j + 1]
        sub = obj.loc[(obj.index > t) & (obj.index <= t1)]
        if sub.empty:
            continue
        rows.append((1 + sub.fillna(0)).prod() - 1)
        idx.append(t1)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def class_daily(px, meta):
    """大类日收益 = 类内等权。"""
    out = {}
    for c in CLASSES:
        ks = [s for s in px.columns if meta.loc[s, "assetClass"] == c]
        out[c] = px[ks].pct_change().mean(axis=1)
    return pd.DataFrame(out)


def window_dist(r, n=GAME_WEEKS):
    v = (1 + r).values
    m = len(v) - n + 1
    return pd.Series([np.prod(v[j:j + n]) - 1 for j in range(m)], index=r.index[:m])


def summary(name, d, base=None):
    s = d if base is None else (d - base).dropna()
    if base is None:
        return {"策略": name, "中位": s.median(), "均值": s.mean(), "标准差": s.std(),
                "5%分位": s.quantile(.05), "最差": s.min(), "最好": s.max(),
                "负收益占比": (s < 0).mean()}
    t = s.mean() / (s.std(ddof=1) / sqrt(len(s))) if len(s) > 2 else np.nan
    return {"策略": name, "超额中位": s.median(), "超额均值": s.mean(),
            "胜率": (s > 0).mean(), "名义t": t,
            "有效t": t / sqrt(GAME_WEEKS) if pd.notna(t) else np.nan, "窗口数": len(s)}


def fmt(df, pct_cols, num_cols=()):
    o = df.copy()
    for c in o.columns:
        if c in pct_cols:
            o[c] = o[c].map(lambda x: f"{x:>9.2%}" if pd.notna(x) else "      n/a")
        elif c in num_cols:
            o[c] = o[c].map(lambda x: f"{x:>7.2f}" if pd.notna(x) else "    n/a")
        elif c == "窗口数":
            o[c] = o[c].astype(int).astype(str)
    return o


# ---------------------------------------------------------------- Part A

def part_a(px, meta, cd, weekly, bench_r):
    p("")
    p("=" * 100)
    p("Part A —— 大类暴露：四个大类在 8 周窗口上的风险收益")
    p("=" * 100)

    p(f"  全池等权的**实际大类权重**（按只数）:")
    tot = len(px.columns)
    for c in CLASSES:
        n = sum(1 for s in px.columns if meta.loc[s, "assetClass"] == c)
        p(f"    {CN[c]:<6}{c:<14} {n:>2} 只   {n/tot:>6.1%}")
    p("  ⚠ 全池等权 = 72% 股票。所谓「大类暴露」决策，实质是要不要偏离这个股票倾向。")

    cdw = to_weekly(cd, weekly)          # ⚠ 必须先转周频，见 to_weekly 的说明
    rows = []
    for c in CLASSES:
        r = summary(CN[c], window_dist(cdw[c]))
        r["年化波动"] = cd[c].std() * sqrt(252)
        rows.append(r)
    A = pd.DataFrame(rows)
    p("")
    p("  各大类单独的 8 周收益分布（等权持有该类）:")
    p(fmt(A, ["中位", "均值", "标准差", "5%分位", "最差", "最好",
              "负收益占比", "年化波动"]).to_string(index=False))

    # 8 周重叠窗口的相关
    p("")
    p("  大类之间的相关（日收益，全样本）:")
    corr = cd.corr()
    p("    " + "".join(f"{CN[c]:>10}" for c in CLASSES))
    for c in CLASSES:
        p(f"    {CN[c]:<6}" + "".join(f"{corr.loc[c, x]:>10.2f}" for x in CLASSES))

    # 静态配置对比
    p("")
    p("  静态配置的 8 周分布（周频再平衡，无成本）:")
    static = {
        "全池79只等权(基准)": bench_r,
        "四类等权(各25%)": cdw.mean(axis=1),
        "股票+商品各半": cdw[["Equity", "Commodity"]].mean(axis=1),
        "仅股票": cdw["Equity"],
        "仅商品": cdw["Commodity"],
    }
    rows = []
    for name, r in static.items():
        d = window_dist(r.fillna(0))
        rows.append(summary(name, d))
    S = pd.DataFrame(rows)
    p(fmt(S, ["中位", "均值", "标准差", "5%分位", "最差", "最好", "负收益占比"]).to_string(index=False))

    # 分散化的实际效果
    p("")
    p("  分散化有没有用？（看 5% 分位和最差窗口）")
    for name, r in static.items():
        d = window_dist(r.fillna(0))
        p(f"    {name:<22} 5%分位 {d.quantile(.05):>8.2%}   最差 {d.min():>8.2%}")

    return A, S


# ---------------------------------------------------------------- Part B

def tsmom_weekly(cd, weekly, L, mode="long_flat"):
    """大类时序动量：大类自身过去 L 日收益 > 0 则持有，否则空仓。"""
    idx = (1 + cd.fillna(0)).cumprod()
    trail = idx / idx.shift(L) - 1

    out, dates, hold = [], [], []
    for j, t in enumerate(weekly[:-1]):
        t1 = weekly[j + 1]
        if t not in trail.index:
            continue
        sig = trail.loc[t]
        if sig.isna().all():
            continue        # 回看窗口尚未成型 —— 不能当成"真实空仓"记 0 收益
        on = [c for c in CLASSES if pd.notna(sig[c]) and sig[c] > 0]
        if mode == "long_short":
            off = [c for c in CLASSES if pd.notna(sig[c]) and sig[c] <= 0]
            w = {c: (1 / len(on) if c in on else (-1 / len(off) if off else 0))
                 for c in CLASSES}
        else:
            w = {c: (1 / len(on) if (on and c in on) else 0.0) for c in CLASSES}
        sub = cd.loc[(cd.index > t) & (cd.index <= t1)]
        if sub.empty:
            continue
        r = sum(sub[c].fillna(0) * w[c] for c in CLASSES)
        out.append((1 + r).prod() - 1)
        dates.append(t1)
        hold.append(len(on))
    return pd.Series(out, index=pd.DatetimeIndex(dates)).sort_index(), \
        pd.Series(hold, index=pd.DatetimeIndex(dates)).sort_index()


def part_b(cd, weekly, bench_r):
    p("")
    p("=" * 100)
    p("Part B —— 大类时序动量（TSMOM）：大类自身动量为正才持有")
    p("=" * 100)

    p(f"  {'回看 L':<12}{'超额中位':>10}{'超额均值':>10}{'胜率':>8}"
      f"{'名义t':>8}{'有效t':>8}{'平均持仓类数':>13}")
    p("  " + "-" * 70)

    # ⚠ base 必须同样是「窗口收益」再相减。bench_r 是周收益，索引与窗口收益
    #   差 8 周，直接相减会错配（曾因此把超额中位从 -0.39% 误报成 +0.80%）。
    bw = window_dist(bench_r)
    best, rows = None, []
    for L in (21, 42, 63, 126, 189, 252):
        r, hold = tsmom_weekly(cd, weekly, L)
        d = window_dist(r)
        s = summary(f"TSMOM L={L}", d, bw)
        rows.append({**s, "L": L, "平均持仓": hold.mean()})
        p(f"  L={L:<10}{s['超额中位']:>+10.2%}{s['超额均值']:>+10.2%}{s['胜率']:>8.1%}"
          f"{s['名义t']:>+8.2f}{s['有效t']:>+8.2f}{hold.mean():>13.2f}")
        if best is None or s["有效t"] > best[1]["有效t"]:
            best = (L, s, r, hold)

    p("")
    p(f"  最佳回看 L={best[0]}（按有效 t）")
    d = window_dist(best[2])
    p(f"    8 周分布: 中位 {d.median():.2%}  5%分位 {d.quantile(.05):.2%}  "
      f"最差 {d.min():.2%}  负收益占比 {(d < 0).mean():.1%}")

    # 多回看集成（把几个 L 平均，降低参数敏感性）
    p("")
    p("  多回看集成（把 L=126/189/252 的持仓平均，降参数敏感度）:")
    sigs = {}
    for L in (126, 189, 252):
        idx = (1 + cd.fillna(0)).cumprod()
        trail = idx / idx.shift(L) - 1
        sigs[L] = trail
    out, dates, hold = [], [], []
    for j, t in enumerate(weekly[:-1]):
        t1 = weekly[j + 1]
        if t not in sigs[126].index:
            continue
        score = sum((sigs[L].loc[t] > 0).astype(float) for L in (126, 189, 252))
        on = [c for c in CLASSES if score[c] >= 2]
        if not on:
            out.append(0.0); dates.append(t1); hold.append(0); continue
        sub = cd.loc[(cd.index > t) & (cd.index <= t1)]
        if sub.empty:
            continue
        w = {c: (1 / len(on) if c in on else 0.0) for c in CLASSES}
        r = sum(sub[c].fillna(0) * w[c] for c in CLASSES)
        out.append((1 + r).prod() - 1)
        dates.append(t1); hold.append(len(on))
    r_ens = pd.Series(out, index=pd.DatetimeIndex(dates)).sort_index()
    d = window_dist(r_ens)
    s = summary("集成", d, bw)
    p(f"    超额中位 {s['超额中位']:+.2%}  胜率 {s['胜率']:.1%}  "
      f"名义t {s['名义t']:+.2f}  有效t {s['有效t']:+.2f}  平均持仓 {np.mean(hold):.2f} 类")
    p(f"    8 周分布: 中位 {d.median():.2%}  5%分位 {d.quantile(.05):.2%}  最差 {d.min():.2%}")

    return pd.DataFrame(rows), r_ens, best


# ---------------------------------------------------------------- Part C

def part_c(cd, weekly, bench_r, best_L, r_best):
    p("")
    p("=" * 100)
    p("Part C —— 组合对比 + 当前读数")
    p("=" * 100)

    cdw = to_weekly(cd, weekly)
    p("  三个候选放在一起:")
    cands = {"全池79只等权(基准)": bench_r,
             f"TSMOM L={best_L}": r_best,
             "四类等权(静态)": cdw.mean(axis=1)}
    rows = []
    for name, r in cands.items():
        d = window_dist(r.fillna(0))
        rows.append(summary(name, d))
    p(fmt(pd.DataFrame(rows),
          ["中位", "均值", "标准差", "5%分位", "最差", "最好", "负收益占比"]).to_string(index=False))

    p("")
    p("=" * 100)
    p("★ 当前读数（数据末日 " + f"{cd.index.max():%Y-%m-%d}" + "）—— 本次比赛的可执行姿态")
    p("=" * 100)
    idx = (1 + cd.fillna(0)).cumprod()
    p(f"  {'大类':<8}" + "".join(f"{'L=' + str(L):>12}" for L in (21, 63, 126, 189, 252))
      + f"{'信号':>10}")
    p("  " + "-" * 82)
    on_now = {}
    for c in CLASSES:
        line = f"  {CN[c]:<8}"
        votes = 0
        for L in (21, 63, 126, 189, 252):
            tr = idx[c].iloc[-1] / idx[c].iloc[-1 - L] - 1
            line += f"{tr:>11.1%} "
            if tr > 0:
                votes += 1
        line += f"{'持有' if votes >= 3 else ('观望' if votes >= 2 else '回避'):>10}"
        on_now[c] = votes
        p(line)

    p("")
    hold = [c for c in CLASSES if on_now[c] >= 3]
    if hold:
        p(f"  → 多数回看同向为正的大类: {'、'.join(CN[c] for c in hold)}")
        p(f"  → 等权持有这些大类，各 {1/len(hold):.0%}")
    else:
        p("  → 无大类获得多数回看支持，建议降低总仓位")

    p("")
    p("  各大类当前所处位置（过去 1 年收益）:")
    for c in CLASSES:
        r1y = idx[c].iloc[-1] / idx[c].iloc[-1 - 252] - 1 if len(idx) > 252 else np.nan
        p(f"    {CN[c]:<6} {r1y:>+8.1%}")


def main():
    os.makedirs(OUT_CSV, exist_ok=True)      # 本脚本只出表，没有图 → 不建 OUT_FIG
    p("=" * 100)
    p("全球资产配置模拟 —— 3a：大类暴露 与 大类时序动量")
    p("赛期 2 个月 / 周末调仓 / 无交易成本 → 评估口径 = 8 周窗口分布")
    p("=" * 100)

    px, meta = load()
    cd = class_daily(px, meta)
    weekly = week_ends(px)

    # 基准：全池 79 只等权，周频再平衡
    out, dates = [], []
    for j, t in enumerate(weekly[:-1]):
        t1 = weekly[j + 1]
        sub = px.loc[(px.index > t) & (px.index <= t1)]
        if sub.empty:
            continue
        out.append((1 + sub.pct_change().mean(axis=1).fillna(0)).prod() - 1)
        dates.append(t1)
    bench_r = pd.Series(out, index=pd.DatetimeIndex(dates)).sort_index()

    A, S = part_a(px, meta, cd, weekly, bench_r)
    B, r_ens, best = part_b(cd, weekly, bench_r)
    part_c(cd, weekly, bench_r, best[0], best[2])

    A.to_csv(os.path.join(OUT_CSV, "A_class_8w_dist.csv"),
             index=False, encoding="utf-8-sig")
    S.to_csv(os.path.join(OUT_CSV, "A_static_alloc.csv"),
             index=False, encoding="utf-8-sig")
    B.to_csv(os.path.join(OUT_CSV, "B_tsmom.csv"),
             index=False, encoding="utf-8-sig")

    p("")
    p("=" * 100)
    p(f"产物已写入 {OUT_CSV}")
    p("=" * 100)


if __name__ == "__main__":
    main()
