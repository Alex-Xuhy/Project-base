# -*- coding: utf-8 -*-
"""
全球资产配置模拟 —— 1a：数据管道

从 econlab.xmu.edu.cn 拉取「全球财富管理2027模拟投资」项目的资产池日线与汇率，
落地成本地缓存，供后续策略步骤使用。

接口事实（2026-09-20 实测，勿凭记忆改）：
  * 基址 https://econlab.xmu.edu.cn/trade/prod-api/tradesys/client
  * 鉴权 Authorization: TradesysBearer <token>
    token 存浏览器 Cookie `Client-Token`，约 30 分钟过期，过期后需重新提取
  * /stockDayData/latestList             资产池清单 + 最新快照（80 只，一次返回）
  * /stockDayData/dataList               单标的日线，**每次全量返回**；
                                         日期过滤与分页参数一律被忽略（已实测 9 种写法）
  * /exchangeRateDayData/listByStockId   按标的返回汇率；实测 80 只标的计价货币均为 USD，
                                         返回的 USD/CNY 序列逐字节相同 → 全池只拉一次

用法：
  python "0. Scripts 脚本/1. 取数与复权/1a_拉取日线汇率.py" --token <TOKEN>            # 全量拉取 / 增量刷新
  python "0. Scripts 脚本/1. 取数与复权/1a_拉取日线汇率.py" --token <TOKEN> --limit 5  # 只拉前 5 只（试跑）
  python 1a_拉取日线汇率.py --qc-only                  # 不联网，只跑数据质量检查

token 解析顺序：--token 参数 > 环境变量 TRADESYS_TOKEN > 项目根目录 _token.local
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd
import requests

# ---------------------------------------------------------------- 常量

BASE_URL = "https://econlab.xmu.edu.cn/trade/prod-api/tradesys/client"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # 脚本在「0. Scripts 脚本/<步骤>/」，上跳两层才是项目根
DATA_DIR = os.path.join(ROOT, "1. Data 数据")     # 共享缓存（第 1 步产物），全步骤只读

TIMEOUT = 60
RETRY = 3
RETRY_SLEEP = 3
POLITE_SLEEP = 0.15          # 请求间隔，对校园服务器客气一点
CHECKPOINT_EVERY = 20        # 每拉 N 只落一次盘

ASSET_CLASS = {1: "Equity", 2: "FixedIncome", 3: "Commodity", 4: "Currency"}
ASSET_CLASS_CH = {1: "股权类", 2: "固定收益类", 3: "大宗商品类", 4: "货币基金类"}

# 出场文件
F_STOCKS = "stocks.csv"
F_PRICES = "prices_long.csv"
F_FX = "fx_usdcny.csv"
F_SNAP = "latest_snapshot.csv"
# 宽表刻意带头缀 raw：与 1b 的 *_adj_* 区分开，避免误取未复权数据喂给策略
F_WIDE_USD = "wide_close_raw_usd.csv"
F_WIDE_CNY = "wide_close_raw_cny.csv"
F_LOG = "pull_log.csv"


def p(*a):
    print(*a, flush=True)


def path(name):
    return os.path.join(DATA_DIR, name)


# ---------------------------------------------------------------- 认证


class AuthError(RuntimeError):
    pass


def resolve_token(cli_token: str | None) -> str:
    if cli_token:
        return cli_token.strip()
    env = os.environ.get("TRADESYS_TOKEN")
    if env:
        return env.strip()
    f = os.path.join(ROOT, "_token.local")
    if os.path.exists(f):
        with open(f, encoding="utf-8") as fh:
            return fh.read().strip()
    raise AuthError(
        "没有拿到 token。请任选一种方式提供：\n"
        "  1) python \"0. Scripts 脚本/1. 取数与复权/1a_拉取日线汇率.py\" --token <TOKEN>\n"
        "  2) 设置环境变量 TRADESYS_TOKEN\n"
        "  3) 把 token 写进项目根目录 _token.local（已在 .gitignore 中）\n"
        "提取方法：浏览器登录站点后，F12 → Console 里执行\n"
        "  document.cookie.split('; ').find(c=>c.startsWith('Client-Token='))?.slice(13)"
    )


class Client:
    def __init__(self, token: str):
        self.s = requests.Session()
        self.s.headers.update(
            {
                "Authorization": "TradesysBearer " + token,
                "Accept": "application/json",
                # HTTP 头只能是 latin-1，这里不能出现中文，否则 requests 直接抛
                # UnicodeEncodeError：'latin-1' codec can't encode characters...
                "User-Agent": "Mozilla/5.0 (1a_daily_bars_fx)",
            }
        )

    def get(self, endpoint: str, **params):
        url = f"{BASE_URL}/{endpoint}"
        last = None
        for attempt in range(1, RETRY + 1):
            try:
                r = self.s.get(url, params=params or None, timeout=TIMEOUT)
                r.raise_for_status()
                body = r.json()
            except Exception as e:                       # 网络/解析类，可重试
                last = e
                if attempt < RETRY:
                    time.sleep(RETRY_SLEEP * attempt)
                    continue
                raise RuntimeError(f"{endpoint} 请求失败（{RETRY} 次）：{e}") from e

            code = body.get("code")
            if code == 401:
                raise AuthError(
                    f"token 失效或权限不足（{endpoint}）。\n"
                    f"服务端返回：{body.get('msg')}\n"
                    "→ 重新从浏览器提取 Client-Token 后再跑一次。"
                )
            if code != 200:
                raise RuntimeError(f"{endpoint} 返回 code={code}：{body.get('msg')}")

            time.sleep(POLITE_SLEEP)
            return body
        raise RuntimeError(f"{endpoint} 重试耗尽：{last}")


# ---------------------------------------------------------------- 抓取


def fetch_meta(cli: Client):
    """资产池元数据：主表 + 大类 + 子类。"""
    p("[1/4] 拉取资产池元数据 ...")
    snap = cli.get("stockDayData/latestList")
    rows = snap.get("rows") or []
    if not rows:
        raise RuntimeError("资产池为空 —— latestList 没返回任何标的，先确认账户状态。")

    classes = cli.get("assetClass/assetClassNameList")["data"]
    subclasses = []
    for cid in sorted(ASSET_CLASS):
        got = cli.get(f"assetClassSub/assetClassSubNameList", assetClassId=cid).get("data") or []
        subclasses.extend(got)

    cls_map = {c["id"]: c for c in classes}
    sub_map = {s["id"]: s for s in subclasses}

    stocks = []
    for r in rows:
        st = r["stock"]
        cid = st["assetClassId"]
        sid = st["assetClassSubId"]
        stocks.append(
            {
                "stockId": st["id"],
                "symbol": st["symbol"],
                "descriptionEn": st.get("descriptionEn", ""),
                "currencyId": st.get("currencyId"),
                "assetClassId": cid,
                "assetClass": cls_map.get(cid, {}).get("nameEn", ""),
                "assetClassCh": cls_map.get(cid, {}).get("nameCh", ""),
                "assetClassSubId": sid,
                "assetClassSub": sub_map.get(sid, {}).get("nameEn", ""),
                "assetClassSubCh": sub_map.get(sid, {}).get("nameCh", ""),
            }
        )
    stocks = pd.DataFrame(stocks).sort_values("stockId").reset_index(drop=True)

    snap_df = pd.DataFrame(
        [
            {
                "stockId": r["stock"]["id"],
                "symbol": r["stock"]["symbol"],
                "dataDatetime": r["dataDatetime"],
                "openPrice": r.get("openPrice"),
                "lastPrice": r.get("lastPrice"),
                "highPrice": r.get("highPrice"),
                "lowPrice": r.get("lowPrice"),
                "changePrice": r.get("changePrice"),
                "changePercent": r.get("changePercent"),
                "volume": r.get("volume"),
                "marketCap": r.get("marketCap"),
                "expenseRatio": r.get("expenseRatio"),
                "assetsUnderManagement": r.get("assetsUnderManagement"),
                "latestStockPriceCny": r.get("latestStockPriceCny"),
            }
            for r in rows
        ]
    )

    p(f"      资产池 {len(stocks)} 只 | 大类 {stocks['assetClass'].nunique()} 个 "
      f"| 子类 {stocks['assetClassSub'].nunique()} 个")
    return stocks, snap_df


def fetch_prices(cli: Client, stocks: pd.DataFrame, limit: int | None):
    """逐只拉日线。接口每次返回全量，增量只能靠本地合并。"""
    todo = stocks if limit is None else stocks.head(limit)
    n = len(todo)
    p(f"[2/4] 拉取日线（{n} 只，接口每次全量返回）...")

    old = pd.read_csv(path(F_PRICES)) if os.path.exists(path(F_PRICES)) else pd.DataFrame()
    if not old.empty:
        p(f"      已有缓存 {len(old):,} 行，将合并去重")

    frames = []
    t0 = time.time()
    for i, row in enumerate(todo.itertuples(), 1):
        body = cli.get("stockDayData/dataList", stockId=int(row.stockId))
        rows = body.get("rows") or []
        if not rows:
            p(f"      [{i:>3}/{n}] {row.symbol:<8} ⚠ 无数据，跳过")
            continue
        df = pd.DataFrame(
            [
                {
                    "stockId": r["stockId"],
                    "symbol": row.symbol,
                    "dataDatetime": r["dataDatetime"],
                    "openPrice": r.get("openPrice"),
                    "lastPrice": r.get("lastPrice"),
                    "highPrice": r.get("highPrice"),
                    "lowPrice": r.get("lowPrice"),
                    "changePercent": r.get("changePercent"),
                    "volume": r.get("volume"),
                }
                for r in rows
            ]
        )
        frames.append(df)
        p(f"      [{i:>3}/{n}] {row.symbol:<8} {len(df):>5} 行  "
          f"{df['dataDatetime'].iloc[0][:10]} ~ {df['dataDatetime'].iloc[-1][:10]}")

        if i % CHECKPOINT_EVERY == 0:
            _save_prices(old, frames)
            p(f"      --- 已落盘检查点（{i}/{n}）---")

    if not frames:
        raise RuntimeError("一只标的都没拉到，中止。")

    prices = _save_prices(old, frames)
    p(f"      完成，用时 {time.time() - t0:.0f}s，合计 {len(prices):,} 行")
    return prices


def _save_prices(old: pd.DataFrame, frames: list[pd.DataFrame]) -> pd.DataFrame:
    new = pd.concat(frames, ignore_index=True)
    if not old.empty:
        new = pd.concat([old, new], ignore_index=True)
    new = (
        new.drop_duplicates(subset=["stockId", "dataDatetime"], keep="last")
        .sort_values(["stockId", "dataDatetime"])
        .reset_index(drop=True)
    )
    os.makedirs(DATA_DIR, exist_ok=True)
    new.to_csv(path(F_PRICES), index=False, encoding="utf-8-sig")
    return new


def fetch_fx(cli: Client, stocks: pd.DataFrame):
    """汇率。实测全池标的返回同一份 USD/CNY，只取一次。"""
    p("[3/4] 拉取汇率 ...")
    probe = cli.get("exchangeRateDayData/listByStockId", stockId=int(stocks["stockId"].iloc[0]))
    data = probe.get("data") or []
    if not data:
        p("      ⚠ 汇率接口没返回数据")
        return pd.DataFrame()

    per = pd.DataFrame(
        [
            {
                "date": r["dataDatetime"],
                "rate": r["rate"],
                "fxTypeId": r.get("typeId"),
                "fxSymbol": (r.get("exchangeRateType") or {}).get("symbol", ""),
            }
            for r in data
        ]
    )
    symbols = sorted(set(per["fxSymbol"]))
    if len(symbols) > 1:
        p(f"      ⚠ 同一标的返回多个汇率品种 {symbols}，需人工确认")
    p(f"      {symbols} {len(per)} 行  {per['date'].iloc[0][:10]} ~ {per['date'].iloc[-1][:10]}")

    old = pd.read_csv(path(F_FX)) if os.path.exists(path(F_FX)) else pd.DataFrame()
    out = pd.concat([old, per], ignore_index=True) if not old.empty else per
    out = (
        out.drop_duplicates(subset=["date", "fxSymbol"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    out.to_csv(path(F_FX), index=False, encoding="utf-8-sig")
    return out


def build_wide(prices: pd.DataFrame, fx: pd.DataFrame):
    """宽表：date × symbol 的收盘价（美元 / 人民币）。"""
    p("[4/4] 生成宽表 ...")
    px = prices.copy()
    px["date"] = px["dataDatetime"].str.slice(0, 10)

    usd = px.pivot_table(index="date", columns="symbol", values="lastPrice", aggfunc="last")
    usd = usd.sort_index()

    cny = usd.copy()
    if not fx.empty:
        f = fx.copy()
        f["date"] = f["date"].str.slice(0, 10)
        f = f.set_index("date")["rate"].astype(float).sort_index()
        # 汇率交易日与股票交易日不完全重合（汇率多 2 天），对齐后前向填充
        rate = f.reindex(usd.index.union(f.index)).ffill().reindex(usd.index)
        missing = int(rate.isna().sum())
        if missing:
            p(f"      ⚠ {missing} 个股票交易日在汇率序列之前，人民币价将为空")
        cny = usd.mul(rate, axis=0)

    os.makedirs(DATA_DIR, exist_ok=True)
    usd.to_csv(path(F_WIDE_USD), encoding="utf-8-sig")
    cny.to_csv(path(F_WIDE_CNY), encoding="utf-8-sig")
    p(f"      美元宽表 {usd.shape[0]} 日 × {usd.shape[1]} 标的")
    p(f"      人民币宽表 {cny.shape[0]} 日 × {cny.shape[1]} 标的")
    return usd, cny


# ---------------------------------------------------------------- 质检


def quality_check(prices: pd.DataFrame, fx: pd.DataFrame, stamps: dict):
    p("")
    p("=" * 78)
    p("数据质量检查")
    p("=" * 78)

    if prices.empty:
        p("无价格数据。")
        return

    prices = prices.copy()
    prices["date"] = prices["dataDatetime"].str.slice(0, 10)
    pool_last = prices["date"].max()
    pool_first = prices["date"].min()
    p(f"样本区间     : {pool_first} ~ {pool_last}")
    p(f"交易日数     : {prices['date'].nunique():,}")
    p(f"总行数       : {len(prices):,}")
    p(f"标的数       : {prices['symbol'].nunique()}")

    # --- 陈旧标的 ---
    last_by = prices.groupby("symbol")["date"].max()
    stale = last_by[last_by < pool_last].sort_values()
    p("")
    if stale.empty:
        p("✓ 无陈旧标的，全部更新到池内最新交易日。")
    else:
        p(f"⚠ 陈旧标的 {len(stale)} 只（最新数据落后于池内最新的 {pool_last}）：")
        for sym, d in stale.items():
            if pd.Timestamp(d) == pd.Timestamp(pool_last):
                continue
            gap = (pd.Timestamp(pool_last) - pd.Timestamp(d)).days
            p(f"    {sym:<8} 停在 {d}  落后 {gap} 天")
        p("  → 这类标的若停更较久，动量信号会被冻结的旧价格污染，建池时需剔除或特殊处理。")

    # --- 起始日参差 ---
    first_by = prices.groupby("symbol")["date"].min()
    late = first_by[first_by > first_by.min()].sort_values()
    if not late.empty:
        p("")
        p(f"ℹ 起始日晚于池内最早的 {first_by.min()} 的标的 {len(late)} 只（新上市 ETF）：")
        for sym, d in late.head(12).items():
            p(f"    {sym:<8} 起于 {d}")
        if len(late) > 12:
            p(f"    ... 另有 {len(late) - 12} 只")

    # --- 重复 ---
    dup = prices.duplicated(subset=["stockId", "date"]).sum()
    p("")
    p(f"{'✓' if dup == 0 else '⚠'} 重复 (stockId, date) : {dup}")

    # --- 缺失/异常价格 ---
    bad = prices[(prices["lastPrice"].isna()) | (prices["lastPrice"] <= 0)]
    p(f"{'✓' if bad.empty else '⚠'} 非正/缺失收盘价 : {len(bad)} 行")

    # --- 周末交易日（美股不应有）---
    wd = pd.to_datetime(prices["date"]).dt.dayofweek
    wk = int((wd >= 5).sum())
    p(f"{'✓' if wk == 0 else '⚠'} 落在周末的交易日 : {wk} 行")

    # --- 异常跳变（复权/拆股嫌疑）---
    p("")
    p("异常单日跳变（|单日涨跌| > 25%，ETF 罕见，疑似未复权拆股）：")
    tmp = prices.sort_values(["symbol", "date"]).copy()
    tmp["ret"] = tmp.groupby("symbol")["lastPrice"].pct_change()
    jumps = tmp[tmp["ret"].abs() > 0.25]
    if jumps.empty:
        p("    ✓ 无")
    else:
        p(f"    ⚠ 共 {len(jumps)} 处，涉及 {jumps['symbol'].nunique()} 只：")
        for sym, g in list(jumps.groupby("symbol"))[:15]:
            worst = g.loc[g["ret"].abs().idxmax()]
            p(f"    {sym:<8} {len(g):>2} 处，最大 {worst['date']} 单日 {worst['ret'] * 100:+.1f}%")
        p("  → 需确认是否为拆股/分红未复权；若是，动量与收益计算都要先复权。")

    # --- 汇率 ---
    if not fx.empty:
        f = fx.copy()
        f["date"] = f["date"].str.slice(0, 10)
        p("")
        p(f"汇率序列     : {f['fxSymbol'].iloc[0]}  {f['date'].min()} ~ {f['date'].max()}  "
          f"{len(f):,} 行")
        fwd = int((pd.to_datetime(f["date"]).max() - pd.to_datetime(pool_last)).days)
        p(f"  汇率比股票数据晚 {fwd} 天（正常：汇率更新到周五，股票数据到周三）")
        rate_now = float(f["rate"].iloc[-1])
        p(f"  最新汇率     : {rate_now}")

    # --- 快照对账 ---
    if "latest" in stamps:
        snap = stamps["latest"]
        p("")
        p("最新快照 vs 日线末行对账（应完全一致）：")
        last_rows = prices.sort_values("date").groupby("symbol").tail(1).set_index("symbol")
        mismatch = 0
        for r in snap.itertuples():
            if r.symbol not in last_rows.index:
                continue
            a, b = float(r.lastPrice), float(last_rows.loc[r.symbol, "lastPrice"])
            if abs(a - b) > 1e-6:
                mismatch += 1
                p(f"    ⚠ {r.symbol:<8} 快照 {a} vs 日线 {b}")
        p(f"    {'✓ 全部一致' if mismatch == 0 else f'⚠ {mismatch} 处不一致'}")


# ---------------------------------------------------------------- 主流程


def main():
    ap = argparse.ArgumentParser(description="1a 数据管道：资产池日线 + 汇率")
    ap.add_argument("--token", help="Client-Token（缺省走环境变量或 _token.local）")
    ap.add_argument("--limit", type=int, help="只拉前 N 只标的，用于试跑")
    ap.add_argument("--qc-only", action="store_true", help="不联网，只跑数据质量检查")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    p("=" * 78)
    p("全球资产配置模拟 —— 1a 数据管道")
    p(f"时间 {datetime.now():%Y-%m-%d %H:%M:%S}   数据目录 {DATA_DIR}")
    p("=" * 78)

    if args.qc_only:
        prices = pd.read_csv(path(F_PRICES)) if os.path.exists(path(F_PRICES)) else pd.DataFrame()
        fx = pd.read_csv(path(F_FX)) if os.path.exists(path(F_FX)) else pd.DataFrame()
        quality_check(prices, fx, {})
        return

    cli = Client(resolve_token(args.token))
    t0 = time.time()

    stocks, snap = fetch_meta(cli)
    stocks.to_csv(path(F_STOCKS), index=False, encoding="utf-8-sig")
    snap.to_csv(path(F_SNAP), index=False, encoding="utf-8-sig")

    prices = fetch_prices(cli, stocks, args.limit)
    fx = fetch_fx(cli, stocks)
    build_wide(prices, fx)

    quality_check(prices, fx, {"latest": snap})

    # 拉取日志
    log = pd.DataFrame(
        [
            {
                "pulledAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stocks": len(stocks),
                "priceRows": len(prices),
                "fxRows": len(fx),
                "poolLastDate": prices["dataDatetime"].max()[:10] if not prices.empty else "",
                "elapsedSec": round(time.time() - t0, 1),
            }
        ]
    )
    if os.path.exists(path(F_LOG)):
        log = pd.concat([pd.read_csv(path(F_LOG)), log], ignore_index=True)
    log.to_csv(path(F_LOG), index=False, encoding="utf-8-sig")

    p("")
    p("=" * 78)
    p(f"完成，用时 {time.time() - t0:.0f}s。产物在「1. Data 数据/」：")
    for f in (F_STOCKS, F_PRICES, F_FX, F_SNAP, F_WIDE_USD, F_WIDE_CNY, F_LOG):
        fp = path(f)
        if os.path.exists(fp):
            p(f"    {f:<22} {os.path.getsize(fp) / 1024:>9,.0f} KB")
    p("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except AuthError as e:
        p("")
        p("【鉴权失败】")
        p(str(e))
        sys.exit(2)
    except KeyboardInterrupt:
        p("\n中断。已拉取的部分已落盘，重跑即可断点续传。")
        sys.exit(130)
