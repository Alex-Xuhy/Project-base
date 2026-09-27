# -*- coding: utf-8 -*-
"""全球资产配置模拟 —— 1c：外部源补最近几日的价格缺口

为什么需要这个：
  站点股票数据**固定滞后到周三**（汇率到周五）。在周末调仓的约束下，
  周三到周末之间发生的行情站点完全没有 —— 而调仓决策正好要落在那个时点。
  例：2026-09-16（周三）之后的 09-17/09-18 两个交易日在站点里根本不存在。

数据源：stockanalysis.com 的公开 history 接口（无需 token）
  GET https://stockanalysis.com/api/symbol/s/{SYM}/history?range=1M&period=Day
  返回 [{t, o, h, l, c, a, v, ch}, ...]  c = 未复权收盘, a = 复权收盘

⚠ 三条必须遵守的口径纪律：
  1. **先对账再用**。外部序列必须与站点在**重叠日期**上逐只对齐（默认取站点末日
     2026-09-16）。差异 > 0.5% 的标的一律标记为可疑，不参与结论 —— 那通常意味着
     复权基准不同或期间发生了拆股，两条序列不可拼接。
  2. **只补缺口，不改历史**。本脚本产出的 external_gap_usd.csv 是**独立**文件，
     绝不写回 prices_adj.csv。站点那份仍是唯一权威历史。
  3. **复权基准**：站点是复权价、外部 c 是原始收盘价。两者只有在
     「站点末日之后无拆股」时才可比。脚本用 `c` 对账、用 `a` 算外部日收益，
     并在末尾显式检查两个口径是否自洽。

用法：
  python 1c_补站点缺口.py              # 补最近 1 个月
  python 1c_补站点缺口.py --range 3M   # 拉更长
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # 脚本在「0. Scripts 脚本/<步骤>/」，上跳两层才是项目根
DATA_DIR = os.path.join(ROOT, "1. Data 数据")     # 共享缓存（第 1 步产物），全步骤只读
GAP_CSV = os.path.join(DATA_DIR, "external_gap_usd.csv")
RECON_CSV = os.path.join(DATA_DIR, "external_gap_recon.csv")
EXT_CSV = os.path.join(DATA_DIR, "wide_close_extgap_usd.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
API = "https://stockanalysis.com/api/symbol/s/{sym}/history?range={rng}&period=Day"
TOL = 0.005          # 对账容差 0.5%
MAX_MOVE = 0.25      # 缺口期单日绝对收益上限，超过说明拼接口径错了
p = print


def stitch(site: pd.DataFrame, ext: pd.DataFrame,
           tolerate: set[str] | None = None) -> pd.DataFrame | None:
    """把外部缺口接到站点序列尾部。

    ⚠ 唯一正确的接法：**只用外部的相对收益，绝对水平锚定站点末日**。
      写成 `ext[gap] × site[se]` 是错的 —— 那会把绝对价乘上去，量级直接爆掉
      （本轮真实踩过：缺口期收益从 +0.7% 变成 +36246%，而「历史未变」的断言
      照样通过，因为它只覆盖 se 及以前）。所以这里加了一道量级断言。

    tolerate: 对账不通过的标的，其缺口值一律丢弃（保留 NaN）。
    """
    se = site.index.max()
    gap = [d for d in ext.index if d > se]
    if not gap:
        return None
    anchor = site.loc[se]
    g = ext.loc[gap].div(ext.loc[se], axis=1).mul(anchor, axis=1)
    if tolerate:
        for s in tolerate:
            if s in g.columns:
                g[s] = np.nan

    out = pd.concat([site, g]).sort_index()
    # 断言 1：历史必须逐点不变
    assert out.loc[:se].equals(site.loc[:se]), "拼接改动了站点历史"
    # 断言 2：缺口段量级必须合理（首日相对锚点）
    first = (g.iloc[0] / anchor - 1).abs().max()
    assert first < MAX_MOVE, f"缺口首日变动 {first:.1%}，量级不合理，检查接法"
    return out


def fetch(sym: str, rng: str, tries: int = 4) -> list[dict] | None:
    url = API.format(sym=sym, rng=rng)
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "application/json",
        "Referer": f"https://stockanalysis.com/etf/{sym.lower()}/",
    })
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                body = json.loads(r.read().decode("utf-8"))
            return body.get("data") or None
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):        # 限流 → 退避重试
                time.sleep(1.5 * (k + 1))
                continue
            p(f"    ! {sym}: HTTP {e.code}")
            return None
        except Exception as e:              # noqa: BLE001
            time.sleep(0.8 * (k + 1))
            last = e
    p(f"    ! {sym}: 重试 {tries} 次仍失败 ({last})")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", default="1M", help="1M / 3M / 6M / 1Y / 5Y")
    ap.add_argument("--sleep", type=float, default=0.35, help="每只之间的间隔秒")
    args = ap.parse_args()

    site = pd.read_csv(os.path.join(DATA_DIR, "wide_close_adj_usd.csv"),
                       encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    syms = [c for c in site.columns]
    site_end = site.index.max()

    p("=" * 100)
    p(f"1c —— 外部源补缺口（stockanalysis.com, range={args.range}）")
    p(f"  站点数据末日 {site_end:%Y-%m-%d}；目标：补齐其后的交易日")
    p(f"  标的数 {len(syms)}")
    p("=" * 100)

    frames, failed = {}, []
    for i, s in enumerate(syms, 1):
        d = fetch(s, args.range)
        if not d:
            failed.append(s)
            continue
        df = pd.DataFrame(d)
        df["t"] = pd.to_datetime(df["t"])
        df = df.set_index("t").sort_index()
        frames[s] = df
        if i % 20 == 0:
            p(f"  ... {i}/{len(syms)}")
        time.sleep(args.sleep)

    if not frames:
        p("  ✗ 一只都没取到，检查网络/源是否可用")
        return

    # 外部宽表：复权收盘 a
    ext_adj = pd.DataFrame({s: d["a"] for s, d in frames.items()}).sort_index()
    ext_raw = pd.DataFrame({s: d["c"] for s, d in frames.items()}).sort_index()
    ext_vol = pd.DataFrame({s: d["v"] for s, d in frames.items()}).sort_index()

    # ---------------------------------------------------------- 对账
    p("")
    p("=" * 100)
    p("对账 —— 重叠日期上「外部未复权收盘 vs 站点复权收盘」")
    p("=" * 100)
    rows = []
    for s in ext_raw.columns:
        if site_end not in ext_raw.index or pd.isna(ext_raw.at[site_end, s]):
            rows.append({"symbol": s, "站点价": site.at[site_end, s]
                         if site_end in site.index else None,
                         "外部价": None, "相对差": None, "结论": "外部无此日"})
            continue
        sv, ev = site.at[site_end, s], ext_raw.at[site_end, s]
        if pd.isna(sv) or pd.isna(ev) or ev == 0:
            rows.append({"symbol": s, "站点价": sv, "外部价": ev,
                         "相对差": None, "结论": "缺值"})
            continue
        rel = ev / sv - 1
        rows.append({"symbol": s, "站点价": sv, "外部价": ev, "相对差": rel,
                     "结论": "一致" if abs(rel) <= TOL else "⚠ 不一致"})
    R = pd.DataFrame(rows)
    bad = R[R["结论"] == "⚠ 不一致"]
    ok = R[R["结论"] == "一致"]

    p(f"  一致      {len(ok):>3} / {len(R)}   (|相对差| <= {TOL:.1%})")
    p(f"  不一致    {len(bad):>3}")
    if len(bad):
        p("")
        for _, r in bad.sort_values("相对差", key=abs, ascending=False).iterrows():
            p(f"    {r['symbol']:<6} 站点 {r['站点价']:>9.4f}  外部 {r['外部价']:>9.4f}"
              f"  差 {r['相对差']:>+8.2%}   ← 不可拼接")
        p("  ⚠ 这些标的**不参与后续结论**：两条序列复权基准不同或期间拆股。")

    # 复权口径自洽性：外部 a 与 c 在同一天应当等比例（无分红日为 1）
    p("")
    last = ext_raw.index.max()
    ratio = (ext_adj.loc[last] / ext_raw.loc[last]).dropna()
    p(f"  外部 a/c 比值（{last:%Y-%m-%d}）：中位 {ratio.median():.4f}，"
      f"范围 {ratio.min():.4f} ~ {ratio.max():.4f}")
    p("    ↑ 偏离 1 说明该标的有分红除息，a 序列才是可比的收益口径。")

    # ---------------------------------------------------------- 缺口
    gap_dates = [d for d in ext_adj.index if d > site_end]
    p("")
    p("=" * 100)
    p("缺口 —— 站点末日之后、外部源有的交易日")
    p("=" * 100)
    if not gap_dates:
        p("  无缺口（外部源也没有更新）")
    else:
        g = ext_adj.loc[gap_dates]
        p(f"  补到 {len(gap_dates)} 个交易日: "
          + "、".join(f"{d:%m-%d}" for d in gap_dates))
        p("")
        # 缺口期收益（复权口径，仅用对账通过的标的）
        good = [s for s in ok["symbol"] if s in ext_adj.columns]
        base = pd.concat([site.loc[[site_end], good], ext_adj[good].loc[gap_dates]])
        tot = base.iloc[-1] / base.iloc[0] - 1
        p(f"  缺口期（{site_end:%m-%d} → {gap_dates[-1]:%m-%d}）"
          f"收益，按复权口径、仅对账通过的 {len(good)} 只:")
        p("")
        p("    【涨幅前 12】")
        for k, v in tot.nlargest(12).items():
            p(f"      {k:<6}{v:>+9.2%}")
        p("    【跌幅前 12】")
        for k, v in tot.nsmallest(12).items():
            p(f"      {k:<6}{v:>+9.2%}")
        p("")
        p(f"    全池等权 {tot.mean():>+8.2%}   中位 {tot.median():>+8.2%}"
          f"   上涨 {(tot > 0).sum()}/{len(tot)}")

    # ---------------------------------------------------------- 落盘
    out = ext_adj.stack().rename("close_adj_usd").reset_index()
    out.columns = ["date", "symbol", "close_adj_usd"]
    out["source"] = "stockanalysis.com"
    out.to_csv(GAP_CSV, index=False, encoding="utf-8-sig")

    stitched = stitch(site, ext_adj, tolerate=set(bad["symbol"]) if len(bad) else set())
    if stitched is not None:
        stitched.to_csv(EXT_CSV, encoding="utf-8-sig")
        p(f"  ✓ {EXT_CSV}   （站点历史 + 外部缺口，仅「当前读数」类分析可用）")
    else:
        # 无缺口（站点没滞后）时 stitch() 返回 None。但下游 3c/3d/4a/4b/5a/5b/5c
        # 无条件读这个文件 —— 这里曾经静默跳过、退出码照样是 0，于是第 1 步报成功、
        # 到第 3 步才 FileNotFoundError。无缺口即「站点历史 + 空缺口」，照写不误。
        site.to_csv(EXT_CSV, encoding="utf-8-sig")
        p(f"  ✓ {EXT_CSV}   （本日无缺口，即站点原始序列）")
    R.to_csv(RECON_CSV, index=False, encoding="utf-8-sig")
    ext_vol.stack().rename("volume").reset_index().rename(
        columns={"level_0": "date", "level_1": "symbol"}).to_csv(
        os.path.join(DATA_DIR, "external_gap_volume.csv"),
        index=False, encoding="utf-8-sig")

    p("")
    p("=" * 100)
    p(f"  ✓ {GAP_CSV}")
    p(f"  ✓ {RECON_CSV}")
    if failed:
        p(f"  ⚠ 未取到: {'、'.join(failed)}")
    p("=" * 100)


if __name__ == "__main__":
    main()
