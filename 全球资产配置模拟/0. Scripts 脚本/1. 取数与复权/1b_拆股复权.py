#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全球资产配置模拟 —— 1b：拆股复权

为什么需要这一步（2026-09-20 实测，勿凭记忆改）：
    站点提供的 ETF 价格**未做拆股复权，且复权处理不一致**。两种矛盾形态都出现过：
      * 价格跳变与 changePercent 一致   → XLK 2025-12-05 两者都是 -49.6%（拆股没复权）
      * 价格跳变与 changePercent 矛盾   → PSP 2023-07-17 价格 +399.9% 但字段写 -0.03%
    反向证据：USO 真实 1:8 反拆日 2020-04-28 价格平滑无跳变 → 该标的**那次**复权过了。
    所以不能假设"全池统一未复权"，必须逐事件核验。

不修的后果（12 个月动量，终点 2026-09-16）：
    XLE 真实 +42.6%（排名 8/80）显示为 -28.7%（排名 73/80）
    XLK 真实 +35.1%（排名 12/80）显示为 -32.5%（排名 75/80）
    动量策略会把池内最强的资产当垫底剔除。

设计原则：
    * 原始数据 prices_long.csv **一个字都不动**，复权结果另存 prices_adj.csv
    * 复权方向：**前复权**（锚定最新价、缩放历史），与行情软件一致
    * SPLIT_EVENTS 是**唯一真源**，每个事件都附独立验证来源
    * 下方检测器只做**绊线告警**，绝不自动改数据 —— 自动应用有把真实崩盘
      （COVID 2020-03、油价崩盘）误判成拆股的风险，必须人工核验后入表

用法：
  python 1b_拆股复权.py              # 复权并质检
  python 1b_拆股复权.py --check-only # 只跑检测器，不动数据
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # 脚本在「0. Scripts 脚本/<步骤>/」，上跳两层才是项目根
DATA_DIR = os.path.join(ROOT, "1. Data 数据")     # 共享缓存（第 1 步产物），全步骤只读

F_RAW = "prices_long.csv"
F_ADJ = "prices_adj.csv"
F_SPLITS = "splits.csv"
F_FX = "fx_usdcny.csv"
F_WIDE_USD = "wide_close_adj_usd.csv"
F_WIDE_CNY = "wide_close_adj_cny.csv"

PRICE_COLS = ["openPrice", "highPrice", "lowPrice", "lastPrice"]

# ---------------------------------------------------------------------------
# 唯一真源：已独立验证的拆股事件
#
# factor = 前复权因子 = 施加在**事件日之前**价格上的乘数
#          （用理论精确值，不要用实测跳变比 —— 实测比含当日真实收益）
#          2:1 正拆 → 价格腰斩 → factor 0.5 ；1:5 反拆 → 价格 ×5 → factor 5
# ---------------------------------------------------------------------------
SPLIT_EVENTS = [
    # --- 2025-12-05 State Street 五只 Select Sector SPDR 2:1 拆分 ---
    # 公告明确 XLF 不在名单内，数据中 XLF 也确实没有跳变，互为印证
    {"symbol": "XLK", "date": "2025-12-05", "factor": 0.5, "ratio": "2:1 正拆",
     "source": "State Street 公告(2025-11-20)，2025-12-05 生效"},
    {"symbol": "XLY", "date": "2025-12-05", "factor": 0.5, "ratio": "2:1 正拆",
     "source": "State Street 公告(2025-11-20)，2025-12-05 生效"},
    {"symbol": "XLE", "date": "2025-12-05", "factor": 0.5, "ratio": "2:1 正拆",
     "source": "State Street 公告(2025-11-20)，2025-12-05 生效"},
    {"symbol": "XLU", "date": "2025-12-05", "factor": 0.5, "ratio": "2:1 正拆",
     "source": "State Street 公告(2025-11-20)，2025-12-05 生效"},
    {"symbol": "XLB", "date": "2025-12-05", "factor": 0.5, "ratio": "2:1 正拆",
     "source": "State Street 公告(2025-11-20)，2025-12-05 生效"},
    # --- 三只反向拆股 ---
    {"symbol": "PSP", "date": "2023-07-17", "factor": 5.0, "ratio": "1:5 反拆",
     "source": "Barchart/StockScan/Trading212 三方一致，2023-07-17 生效"},
    {"symbol": "UNG", "date": "2024-01-24", "factor": 4.0, "ratio": "1:4 反拆",
     "source": "MIAX 公司行动公告 + SEC 8-K，2024-01-24 生效"},
    {"symbol": "MJ", "date": "2025-02-21", "factor": 12.0, "ratio": "1:12 反拆",
     "source": "Amplify 半年度股东报告 + MIAX，2025-02-21 生效"},
]

# 检测器用的**真正标准**拆股比率：整数倍与其倒数
# ⚠ 刻意排除 3:2 / 5:4 / 3:4 / 5:6 这类非整数比 —— 放宽后会把 COVID 崩盘
#   （2020-03-12/13/16/18）与 2023-10-18 全市场下跌日误判成拆股
STANDARD_RATIOS = [n for n in (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20)] + \
                  [1 / n for n in (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20)]

TOL = 0.015          # 比率容差 1.5%


def p(*a):
    print(*a, flush=True)


def path(n):
    return os.path.join(DATA_DIR, n)


def detect(prices: pd.DataFrame) -> pd.DataFrame:
    """扫描标准拆股比率的跳变。只做告警，不自动应用。"""
    d = prices.sort_values(["symbol", "dataDatetime"]).copy()
    d["ratio"] = d["lastPrice"] / d.groupby("symbol")["lastPrice"].shift()

    hits = []
    for r in d[d["ratio"].notna()].itertuples():
        for s in STANDARD_RATIOS:
            if abs(r.ratio - s) / s < TOL:
                hits.append({"symbol": r.symbol, "date": r.dataDatetime[:10],
                             "ratio": r.ratio, "nearest": s})
                break
    if not hits:
        return pd.DataFrame(columns=["symbol", "date", "ratio", "nearest"])

    h = pd.DataFrame(hits)
    # 同日多只跳变 → 行情事件嫌疑（唯一例外：设计性同日拆分）
    same_day = h.groupby("date").size()
    h["sameDayCount"] = h["date"].map(same_day)
    return h


def main():
    ap = argparse.ArgumentParser(description="1b：拆股复权")
    ap.add_argument("--check-only", action="store_true", help="只跑检测器，不改数据")
    args = ap.parse_args()

    p("=" * 78)
    p("全球资产配置模拟 —— 1b 拆股复权")
    p(f"时间 {datetime.now():%Y-%m-%d %H:%M:%S}")
    p("=" * 78)

    if not os.path.exists(path(F_RAW)):
        p(f"找不到 {F_RAW}，请先跑 1a_拉取日线汇率.py")
        sys.exit(1)

    # utf-8-sig：1a 落盘时带 BOM，不指定会让首列名变成 '﻿stockId'
    raw = pd.read_csv(path(F_RAW), encoding="utf-8-sig")
    raw["date"] = raw["dataDatetime"].str.slice(0, 10)
    # volume 落盘时是 int64，除以 factor 会产出浮点并触发 LossySetitemError
    raw[PRICE_COLS + ["volume"]] = raw[PRICE_COLS + ["volume"]].astype(float)
    p(f"读入原始数据 {len(raw):,} 行 / {raw['symbol'].nunique()} 只")

    # ---------------- 检测器：绊线告警 ----------------
    p("")
    p("--- 检测器：标准拆股比率跳变扫描 ---")
    found = detect(raw)
    if found.empty:
        p("  ✓ 未发现标准拆股比率的跳变")
    else:
        p(f"  发现 {len(found)} 处：")
        for r in found.itertuples():
            known = any(e["symbol"] == r.symbol and e["date"] == r.date for e in SPLIT_EVENTS)
            if known:
                # 已在表内 = 已人工核验过，不必再报同日告警
                # （5 只 SPDR 就是设计性同日拆分，是本判据的唯一已知例外）
                tag = "已在表内（已验证）"
            else:
                tag = "★ 新事件，需人工核验后入表"
                if r.sameDayCount > 1:
                    tag += f"  ⚠ 同日另有 {int(r.sameDayCount) - 1} 只跳变，疑似行情而非拆股"
            p(f"    {r.symbol:<6} {r.date}  比率={r.ratio:.4f} ≈ {r.nearest:<6g} [{tag}]")

    unknown = [r for r in found.itertuples()
               if not any(e["symbol"] == r.symbol and e["date"] == r.date for e in SPLIT_EVENTS)]
    if unknown:
        p("")
        p(f"  ⚠⚠ 有 {len(unknown)} 处跳变不在 SPLIT_EVENTS 表内。")
        p("     **本脚本不会自动应用它们** —— 自动应用有把真实崩盘误判成拆股的风险。")
        p("     请逐个核验（查发行方公告/交易所公司行动），确认后手工加入 SPLIT_EVENTS 再重跑。")

    if args.check_only:
        p("")
        p("--check-only：未改动任何数据。")
        return

    # ---------------- 应用前复权 ----------------
    p("")
    p(f"--- 应用前复权（{len(SPLIT_EVENTS)} 个事件）---")
    adj = raw.copy()
    for ev in SPLIT_EVENTS:
        mask = (adj["symbol"] == ev["symbol"]) & (adj["date"] < ev["date"])
        n = int(mask.sum())
        if n == 0:
            p(f"  ⚠ {ev['symbol']} {ev['date']}：命中 0 行，请检查日期是否越界")
            continue
        # 价格按 factor 缩放，成交量反向缩放（拆分改变股数）
        adj.loc[mask, PRICE_COLS] = adj.loc[mask, PRICE_COLS].astype(float) * ev["factor"]
        adj.loc[mask, "volume"] = adj.loc[mask, "volume"] / ev["factor"]
        pre = adj.loc[mask, "lastPrice"].iloc[-1] if n else float("nan")
        p(f"  {ev['symbol']:<6} {ev['date']}  {ev['ratio']:<10} factor={ev['factor']:<5g} "
          f"调整 {n:>5} 行（事件前最后收盘 → {pre:,.2f}）")

    adj = adj.drop(columns=["date"]).sort_values(["stockId", "dataDatetime"]).reset_index(drop=True)
    adj.to_csv(path(F_ADJ), index=False, encoding="utf-8-sig")
    p(f"  ✓ 已写出 {F_ADJ}（{len(adj):,} 行）")

    # 事件表落盘
    ev_df = pd.DataFrame(SPLIT_EVENTS)[["symbol", "date", "ratio", "factor", "source"]]
    ev_df.to_csv(path(F_SPLITS), index=False, encoding="utf-8-sig")
    p(f"  ✓ 已写出 {F_SPLITS}（{len(ev_df)} 个事件）")

    # ---------------- 宽表 ----------------
    p("")
    p("--- 生成复权宽表 ---")
    a = adj.copy()
    a["date"] = a["dataDatetime"].str.slice(0, 10)
    usd = a.pivot_table(index="date", columns="symbol", values="lastPrice", aggfunc="last").sort_index()

    cny = usd.copy()
    if os.path.exists(path(F_FX)):
        fx = pd.read_csv(path(F_FX), encoding="utf-8-sig")
        fx["date"] = fx["date"].str.slice(0, 10)
        rate = (fx.set_index("date")["rate"].astype(float).sort_index()
                  .reindex(usd.index.union(fx["date"]))
                  .ffill().reindex(usd.index))
        cny = usd.mul(rate, axis=0)
        p(f"  汇率已对齐（{F_FX}）")
    else:
        p(f"  ⚠ 找不到 {F_FX}，跳过人民币宽表")

    usd.to_csv(path(F_WIDE_USD), encoding="utf-8-sig")
    cny.to_csv(path(F_WIDE_CNY), encoding="utf-8-sig")
    p(f"  ✓ {F_WIDE_USD}  {usd.shape[0]} 日 × {usd.shape[1]} 标的")
    p(f"  ✓ {F_WIDE_CNY}  {cny.shape[0]} 日 × {cny.shape[1]} 标的")

    # ---------------- 复权后质检 ----------------
    p("")
    p("=" * 78)
    p("复权后质检（验证拆股跳变已消除）")
    p("=" * 78)
    after = detect(adj)
    before = detect(raw)

    def summarize(h, label):
        if h.empty:
            p(f"  {label}: 0 处")
            return
        big = h[h.ratio.abs().sub(1).abs() > 0.25]
        p(f"  {label}: {len(h)} 处标准比率命中，其中 |涨跌|>25% 的 {len(big)} 处")
        return big

    p("")
    b_big = summarize(before, "复权前")
    a_big = summarize(after, "复权后")
    p("")
    if a_big is None or len(a_big) == 0:
        p("  ✓ 复权后已无大幅标准比率跳变")
    else:
        p(f"  ⚠ 复权后仍剩 {len(a_big)} 处大幅跳变：")
        for r in a_big.itertuples():
            p(f"    {r.symbol:<6} {r.date}  比率={r.ratio:.4f}")

    p("")
    p("对照检验 —— 5 只 SPDR 的 12 个月动量（终点取数据末日）：")
    end = usd.index.max()
    t0 = (pd.Timestamp(end) - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
    raw_px = raw.copy()
    raw_px["date"] = raw_px["dataDatetime"].str.slice(0, 10)
    raw_usd = raw_px.pivot_table(index="date", columns="symbol",
                                 values="lastPrice", aggfunc="last").sort_index()
    for sym in ["XLK", "XLE", "XLB", "XLY", "XLU"]:
        try:
            r_raw = raw_usd.loc[end, sym] / raw_usd.loc[raw_usd.index[raw_usd.index.searchsorted(t0)], sym] - 1
            r_adj = usd.loc[end, sym] / usd.loc[usd.index[usd.index.searchsorted(t0)], sym] - 1
            p(f"    {sym:<6} 复权前 {r_raw:>+8.1%}   →   复权后 {r_adj:>+8.1%}")
        except Exception as e:
            p(f"    {sym:<6} 计算失败：{e}")

    p("")
    p("=" * 78)
    p("完成。原始 prices_long.csv 未改动，复权数据见 prices_adj.csv")
    p("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        p("\n中断。")
        sys.exit(130)
