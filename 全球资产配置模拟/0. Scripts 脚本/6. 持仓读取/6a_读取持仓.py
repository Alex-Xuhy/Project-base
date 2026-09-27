# -*- coding: utf-8 -*-
"""全球资产配置模拟 —— 6a：从交易系统实读当前持仓

为什么需要这个：
  第 4 步的 `4b_持仓体检.py` 里，持仓是**硬编码**的 `HOLDINGS` 字典 —— 每次调仓
  都要手动去改它，改完还得回头核对。而站点其实有现成接口（2026-09-27 从
  `app.js` 逆向前端挖出并实测）：

    GET /account/info   账户（initialCash / cash / clientId / projectId）
    GET /position/list  当前持仓（股数、成本、市值、浮盈）
    GET /order/list     成交流水（type：1=买 0=卖）

  本脚本把它们读出来、配上项目自己的功能分类（`T_taxonomy.csv`），
  按大类 / 子类汇总暴露。**只读，不下单、不写任何站点数据。**

token：
  与 1a 同一套 —— 浏览器登录后 F12 → Console：
    document.cookie.split('; ').find(c=>c.startsWith('Client-Token='))?.slice(13)
  **约 30 分钟过期**。过期后所有接口返回 code=401，脚本会明确提示你换一个。

⚠ 口径提醒：
  * `/summary/list` 是**每日收盘快照**，会落后于 `/account/info` 的实时现金。
    本脚本一律以 `account/info` + `position/list` 为准，不读 summary。
  * 站点行情通常滞后（股票到周三、汇率到周五）。持仓市值由**站点**给出，
    可能不是你下单时的真实市价 —— 尤其刚调仓完，成本与市值会短暂相等。

用法：
  python "0. Scripts 脚本/6. 持仓读取/6a_读取持仓.py"              # 读持仓
  python "0. Scripts 脚本/6. 持仓读取/6a_读取持仓.py" --orders     # 连成交流水一起看
  python "0. Scripts 脚本/6. 持仓读取/6a_读取持仓.py" --csv        # 落盘到本步骤的 csv/
  python "0. Scripts 脚本/6. 持仓读取/6a_读取持仓.py" --token XXX  # 临时指定 token

产物（仅当 --csv）：`6. 持仓读取/csv/P_positions.csv`、`P_exposure.csv`、
  `P_orders.csv`（若 --orders）
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # 脚本在「0. Scripts 脚本/<步骤>/」，上跳两层才是项目根
OUT_CSV = os.path.join(HERE, "csv")
TAX_CSV = os.path.join(ROOT, "2. 资产池认知", "csv", "T_taxonomy.csv")

BASE_URL = "https://econlab.xmu.edu.cn/trade/prod-api/tradesys/client"
TIMEOUT = 25


def p(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 认证


class AuthError(RuntimeError):
    pass


def resolve_token(cli_token: str | None) -> str:
    """与 1a 同序：--token > 环境变量 TRADESYS_TOKEN > 项目根 _token.local。"""
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
        "  1) python \"0. Scripts 脚本/6. 持仓读取/6a_读取持仓.py\" --token <TOKEN>\n"
        "  2) 设置环境变量 TRADESYS_TOKEN\n"
        "  3) 把 token 写进项目根目录 _token.local（已在 .gitignore 中）\n"
        "提取方法：浏览器登录站点后，F12 → Console 里执行\n"
        "  document.cookie.split('; ').find(c=>c.startsWith('Client-Token='))?.slice(13)"
    )


def get(endpoint: str, token: str, **params):
    """GET 一个接口。401 单独抛 AuthError（那是 token 过期，不是网络问题）。"""
    url = f"{BASE_URL}/{endpoint}"
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url, headers={
        "Authorization": "TradesysBearer " + token,
        "Accept": "application/json",
        # HTTP 头只能是 latin-1，这里不能出现中文，否则 requests/urllib 直接抛
        # UnicodeEncodeError：'latin-1' codec can't encode characters...（1a 踩过同一个坑）
        "User-Agent": "Mozilla/5.0 (6a_positions_reader)",
    })
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            body = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{endpoint} HTTP {e.code}") from e
    except Exception as e:
        raise RuntimeError(f"{endpoint} 请求失败：{e}") from e
    if body.get("code") == 401:
        raise AuthError(
            f"token 失效（{endpoint}）。token 约 30 分钟过期。\n"
            "→ 重新从浏览器提取 Client-Token 后重跑。"
        )
    return body


# ---------------------------------------------------------------- 取数


def load_tax():
    """功能分类；第 2 步没跑过就返回空表，不中断。"""
    if os.path.exists(TAX_CSV):
        return pd.read_csv(TAX_CSV, encoding="utf-8-sig", index_col=0)
    return None


def fetch_positions(token):
    """持仓 + 站点最新价 → 一张宽表。"""
    info = get("account/info", token)["data"]
    rows = get("position/list", token).get("rows") or []
    latest = {r["stockId"]: r for r in get("stockDayData/latestList", token).get("rows", [])}

    out = []
    for r in rows:
        sym = (r.get("stock") or {}).get("symbol", "?")
        sid = r.get("stockId")
        n = r.get("positionNum") or 0
        cost = r.get("dilutedCostPriceCny") or 0
        mv = r.get("totalMarketValueCny") or 0
        lat = latest.get(sid, {})
        out.append({
            "标的": sym,
            "股数": n,
            "成本价CNY": cost,
            "成本额CNY": r.get("totalDilutedCostPriceCny") or 0,
            "市值CNY": mv,
            "浮盈亏CNY": r.get("totalRealFloatingProfitCny") or 0,
            "收益率": r.get("floatingProfitRateCny") or 0,
            "最新价CNY": lat.get("latestStockPriceCny"),
            "行情日": (lat.get("dataDatetime") or "")[:10],
        })
    df = pd.DataFrame(out)
    return info, df


def fetch_orders(token):
    rows = get("order/list", token).get("rows") or []
    out = []
    for r in rows:
        out.append({
            "时间": (r.get("createTime") or "")[:16],
            "标的": (r.get("stock") or {}).get("symbol", "?"),
            "方向": "买入" if str(r.get("type")) == "1" else "卖出",
            "数量": r.get("volume"),
            "价格": r.get("price"),
            "金额CNY": r.get("totalPriceCny"),
        })
    return pd.DataFrame(out).sort_values("时间").reset_index(drop=True) if out else pd.DataFrame()


# ---------------------------------------------------------------- 展示


def show(info, pos, tax, orders=None):
    init = info.get("initialCash") or 0
    cash = info.get("cash") or 0
    mv = pos["市值CNY"].sum() if len(pos) else 0
    total = cash + mv

    p("=" * 92)
    p(f"  账户：{(info.get('client') or {}).get('nickName','')}"
      f" · {(info.get('project') or {}).get('projectName','')}"
      f" · {(info.get('group') or {}).get('groupName','')}")
    p("=" * 92)
    p(f"  初始资金 {init:>14,.0f}      现金 {cash:>14,.0f}      持仓市值 {mv:>14,.0f}")
    p(f"  总资产   {total:>14,.0f}      总收益 {total-init:>+13,.0f}"
      f"     收益率 {(total/init-1):>+9.2%}" if init else "")
    p(f"  持仓浮盈 {pos['浮盈亏CNY'].sum() if len(pos) else 0:>+14,.0f}"
      f"      现金占比 {cash/total:.2%}" if total else "")

    if not len(pos):
        p("\n  （当前无持仓）")
        return

    # 补中文名 / 功能分类
    if tax is not None:
        pos["中文名"] = [tax.at[c, "中文名"] if c in tax.index else "" for c in pos["标的"]]
        pos["功能大类"] = [tax.at[c, "功能大类"] if c in tax.index else "其他" for c in pos["标的"]]
        pos["功能子类"] = [tax.at[c, "功能子类"] if c in tax.index else "其他" for c in pos["标的"]]
    else:
        pos["中文名"] = pos["功能大类"] = pos["功能子类"] = ""
    pos["占账户"] = pos["市值CNY"] / total

    p("")
    p("=" * 92)
    p("  当前持仓（按市值降序）")
    p("=" * 92)
    d = pos.sort_values("市值CNY", ascending=False)
    p(f"  {'标的':<6}{'中文名':<14}{'子类':<10}{'股数':>7}{'成本价':>11}"
      f"{'市值':>13}{'浮盈亏':>11}{'收益率':>9}{'占账户':>9}")
    p("  " + "-" * 88)
    for _, r in d.iterrows():
        p(f"  {r['标的']:<6}{r['中文名']:<14}{r['功能子类']:<10}{r['股数']:>7,.0f}"
          f"{r['成本价CNY']:>11,.2f}{r['市值CNY']:>13,.0f}"
          f"{r['浮盈亏CNY']:>+11,.0f}{r['收益率']:>+9.2%}{r['占账户']:>9.2%}")
    p("  " + "-" * 88)
    p(f"  {'现金':<6}{'':<14}{'':<10}{'':>7}{'':>11}{cash:>13,.0f}"
      f"{'':>11}{'':>9}{cash/total:>9.2%}")
    p(f"  {'合计':<6}{'':<14}{'':>32}{'':>13}{'':>11}{pos['占账户'].sum()+cash/total:>9.2%}")

    # 暴露
    p("")
    p("=" * 92)
    p("  暴露汇总")
    p("=" * 92)
    p("  按功能大类：")
    for k, v in pos.groupby("功能大类")["市值CNY"].sum().sort_values(ascending=False).items():
        p(f"    {k:<10}{v:>13,.0f}{v/total:>9.2%}")
    p(f"    {'现金':<10}{cash:>13,.0f}{cash/total:>9.2%}")
    p("")
    p("  按功能子类（集中度视角）：")
    for k, v in pos.groupby("功能子类")["市值CNY"].sum().sort_values(ascending=False).items():
        p(f"    {k:<10}{v:>13,.0f}{v/total:>9.2%}")

    if orders is not None and len(orders):
        p("")
        p("=" * 92)
        p(f"  成交流水（{len(orders)} 笔，时间正序）")
        p("=" * 92)
        for _, r in orders.iterrows():
            p(f"  {r['时间']}  {r['标的']:<6}{r['方向']:<5}{r['数量']:>7,.0f}"
              f" @ {r['价格']:>10,.4f}   {r['金额CNY']:>12,.0f}")


def main():
    ap = argparse.ArgumentParser(
        description="从交易系统实读当前持仓",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", help="Client-Token（缺省走环境变量或 _token.local）")
    ap.add_argument("--orders", action="store_true", help="连成交流水一起读")
    ap.add_argument("--csv", action="store_true", help="把结果落盘到本步骤的 csv/")
    args = ap.parse_args()

    token = resolve_token(args.token)
    info, pos = fetch_positions(token)
    orders = fetch_orders(token) if args.orders else None
    tax = load_tax()

    show(info, pos, tax, orders)

    if args.csv:
        os.makedirs(OUT_CSV, exist_ok=True)
        pos.to_csv(os.path.join(OUT_CSV, "P_positions.csv"), index=False,
                   encoding="utf-8-sig")
        if tax is not None and len(pos):
            exp = pd.concat([
                pos.groupby("功能大类")["市值CNY"].sum().rename("市值CNY"),
                pos.groupby("功能子类")["市值CNY"].sum().rename("市值CNY"),
            ])
            exp.to_frame().to_csv(os.path.join(OUT_CSV, "P_exposure.csv"),
                                  encoding="utf-8-sig")
        if orders is not None and len(orders):
            orders.to_csv(os.path.join(OUT_CSV, "P_orders.csv"), index=False,
                          encoding="utf-8-sig")
        p("")
        p(f"  ✓ 已落盘到 {OUT_CSV}")


if __name__ == "__main__":
    try:
        main()
    except AuthError as e:
        raise SystemExit(f"\n!! {e}")
