# -*- coding: utf-8 -*-
"""
KOOK 登录页「游戏台词」抓取

KOOK 网页版进入时加载框会闪现一句游戏台词（游戏名 + 角色 + 台词），一闪而过。
这些文案有两个来源，本脚本把两处都抓下来并合并去重：

  1. 远端资源  saying.json
     —— 由 https://www.kookapp.cn/api/file/config 下发 url/etag，是权威列表
  2. 前端兜底  打包在 JS 入口里的默认数组
     —— 远端加载完成前先用它，可能含远端没有的条目

输出（默认为脚本同目录）:
  kook_台词.txt    人可读，按游戏名分组
  kook_台词.csv    表格，Excel 直接打开（带 BOM）
  kook_台词.json   结构化原始数据，含来源标记与 etag

用法:
  python kook登入语抓取.py                # 正常抓取
  python kook登入语抓取.py --deep         # 额外扫描懒加载 chunk 找兜底数组（更慢更全）
  python kook登入语抓取.py -o out         # 指定输出目录
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import requests

HOME = "https://www.kookapp.cn/"
# 注意：根路径 / 是营销下载页，没有 SPA 脚本；要拿打包后的 JS 必须走 /app/*
APP_ENTRIES = (
    "https://www.kookapp.cn/app/home",
    "https://www.kookapp.cn/app/home/friend/online",
    "https://www.kookapp.cn/app/login",
)
CONFIG_API = "https://www.kookapp.cn/api/file/config"
SAYING_FALLBACK_URL = "https://img.kookapp.cn/assets/saying.json"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

SESSION = requests.Session()
SESSION.headers["User-Agent"] = UA
SESSION.headers["Accept-Language"] = "zh-CN,zh;q=0.9"


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# 1. 远端 saying.json
# --------------------------------------------------------------------------
def fetch_remote_saying() -> tuple[list[dict], str, str]:
    """返回 (条目列表, 实际使用的 url, etag)。任何一步失败都能降级。"""
    url, etag = SAYING_FALLBACK_URL, ""
    try:
        r = SESSION.get(CONFIG_API, timeout=20)
        r.raise_for_status()
        saying = (r.json().get("data") or {}).get("saying") or {}
        url = saying.get("url") or url
        etag = saying.get("etag") or ""
        log(f"  /api/file/config 下发 url = {url}")
        log(f"                     etag = {etag}")
    except Exception as e:
        log(f"  [!] 读取 /api/file/config 失败({e})，回退到已知地址")

    try:
        # requests 会自动处理 gzip/deflate/br
        r = SESSION.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log(f"  [!] 下载 saying.json 失败: {e}")
        return [], url, etag

    if not isinstance(data, list):
        log(f"  [!] saying.json 结构异常（{type(data).__name__}），跳过")
        return [], url, etag

    items = [d for d in data if isinstance(d, dict) and d.get("words")]
    log(f"  远端 saying.json 解析到 {len(items)} 条")
    return items, url, etag


# --------------------------------------------------------------------------
# 2. 前端兜底数组
#    JS 里形如： let tV=[{game:"...",role_name:"...",words:"..."},...]
# --------------------------------------------------------------------------
OBJ_RE = re.compile(
    r'\{game\s*:\s*"((?:[^"\\]|\\.)*)"\s*,'
    r'\s*role_name\s*:\s*"((?:[^"\\]|\\.)*)"\s*,'
    r'\s*words\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
)

_JS_ESCAPE = {
    "n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f",
    "v": "\v", "0": "\0", "\\": "\\", "'": "'", '"': '"', "/": "/", "\n": "",
}


def js_unescape(s: str) -> str:
    """还原 JS 字符串字面量里的转义（\\xNN / \\uNNNN / \\n 等）。"""
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c != "\\" or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        nxt = s[i + 1]
        if nxt == "x" and i + 3 < n:
            try:
                out.append(chr(int(s[i + 2:i + 4], 16)))
                i += 4
                continue
            except ValueError:
                pass
        elif nxt == "u":
            if i + 2 < n and s[i + 2] == "{":  # \u{1F600}
                j = s.find("}", i + 3)
                if j != -1:
                    try:
                        out.append(chr(int(s[i + 3:j], 16)))
                        i = j + 1
                        continue
                    except ValueError:
                        pass
            elif i + 5 < n:
                try:
                    out.append(chr(int(s[i + 2:i + 6], 16)))
                    i += 6
                    continue
                except ValueError:
                    pass
        elif nxt in _JS_ESCAPE:
            out.append(_JS_ESCAPE[nxt])
            i += 2
            continue
        out.append(nxt)
        i += 2
    return "".join(out)


def extract_from_js(text: str) -> list[dict]:
    items = []
    for game, role, words in OBJ_RE.findall(text):
        words = js_unescape(words)
        if words.strip():
            items.append({
                "game": js_unescape(game).strip(),
                "role_name": js_unescape(role).strip(),
                "words": words,
            })
    return items


def list_scripts(html: str) -> list[str]:
    urls = re.findall(r'<script[^>]+src="([^"]+\.js[^"]*)"', html)
    return [u for u in urls if "/static/js/" in u]


def extract_from_frontend(deep: bool) -> list[dict]:
    urls: list[str] = []
    for entry_url in APP_ENTRIES:
        try:
            html = SESSION.get(entry_url, timeout=20).text
        except Exception as e:
            log(f"  [!] 拉取 {entry_url} 失败: {e}")
            continue
        urls = list_scripts(html)
        if urls:
            log(f"  入口页 {entry_url}")
            break
    if not urls:
        log("  [!] 没找到任何 SPA 脚本，跳过前端兜底数组")
        return []
    # 入口包优先；兜底数组历史上一直在 index.*.js 里
    entry = [u for u in urls if re.search(r"/index\.[0-9a-f]+\.js$", u)]
    others = [u for u in urls if u not in entry and "vendors" not in u and "locales" not in u]

    found: list[dict] = []
    for u in entry + others:
        try:
            js = SESSION.get(u, timeout=30).text
        except Exception:
            continue
        hits = extract_from_js(js)
        if hits:
            log(f"  命中 {len(hits)} 条 <- {os.path.basename(u)}")
            found.extend(hits)
            if u in entry:
                break  # 入口包命中就够了

    if found or not deep:
        return found

    # 兜底数组若被挪进懒加载 chunk：解析 runtime 的 chunk 表再逐个找
    log("  顶层脚本没找到，改扫懒加载 chunk ...")
    runtime = next((u for u in urls if "runtime" in u), None)
    if not runtime:
        return []
    try:
        rt = SESSION.get(runtime, timeout=20).text
    except Exception:
        return []

    m = re.search(r'l\.u\s*=\s*e\s*=>\s*"static/js/"\s*\+.*?\(\{([^}]*)\}\)\[e\]\s*\+\s*"\.chunk\.js"', rt, re.S)
    if not m:
        return []
    hashes = dict(re.findall(r'(\d+)\s*:\s*"([^"]+)"', m.group(1)))
    base = runtime.rsplit("/", 1)[0]
    for cid, h in hashes.items():
        try:
            js = SESSION.get(f"{base}/{cid}.{h}.chunk.js", timeout=30).text
        except Exception:
            continue
        hits = extract_from_js(js)
        if hits:
            log(f"  命中 {len(hits)} 条 <- chunk {cid}")
            found.extend(hits)
    return found


# --------------------------------------------------------------------------
# 3. 合并 / 输出
# --------------------------------------------------------------------------
def merge(remote: list[dict], local: list[dict]) -> list[dict]:
    merged: dict[tuple, dict] = {}

    def key(d):
        # 台词为主体去重；忽略角色名里的多余空格差异
        norm = lambda s: re.sub(r"\s+", "", s or "")
        return (norm(d.get("game")), norm(d.get("words")))

    for d in remote:
        merged[key(d)] = {
            "game": (d.get("game") or "").strip(),
            "role_name": (d.get("role_name") or "").strip(),
            "words": (d.get("words") or "").strip(),
            "source": ["remote"],
        }
    for d in local:
        k = key(d)
        if k in merged:
            merged[k]["source"].append("js")
            # 远端 role_name 为空/为 NA 时用本地的补上
            if merged[k]["role_name"] in ("", "NA") and d["role_name"]:
                merged[k]["role_name"] = d["role_name"]
        else:
            merged[k] = {
                "game": d["game"], "role_name": d["role_name"],
                "words": d["words"], "source": ["js"],
            }
    for v in merged.values():
        v["source"] = "+".join(dict.fromkeys(v["source"]))
    return list(merged.values())


def write_outputs(items: list[dict], outdir: str, etag: str, src_url: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    items = sorted(items, key=lambda d: (d["game"], d["words"]))
    p = lambda name: os.path.join(outdir, name)

    with open(p("kook_台词.json"), "w", encoding="utf-8") as f:
        json.dump({
            "source_url": src_url, "etag": etag, "count": len(items), "items": items,
        }, f, ensure_ascii=False, indent=2)

    with open(p("kook_台词.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["游戏", "角色", "台词", "来源"])
        for d in items:
            w.writerow([d["game"], d["role_name"], d["words"], d["source"]])

    with open(p("kook_台词.txt"), "w", encoding="utf-8") as f:
        f.write(f"KOOK 加载页游戏台词   共 {len(items)} 条\n")
        f.write(f"来源: {src_url}    etag: {etag}\n")
        f.write("=" * 68 + "\n\n")
        cur = None
        for d in items:
            if d["game"] != cur:
                cur = d["game"]
                f.write(f"\n【{cur}】\n")
            who = f"（{d['role_name']}）" if d["role_name"] and d["role_name"] != "NA" else ""
            f.write(f"  {d['words']}{who}\n")
            if d["source"] == "js":
                f.write("      ^ 仅前端内置，远端 saying.json 暂无\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="抓取 KOOK 登录页游戏台词")
    ap.add_argument("-o", "--outdir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="输出目录（默认脚本所在目录）")
    ap.add_argument("--deep", action="store_true",
                    help="顶层脚本没命中时，继续扫描懒加载 chunk")
    args = ap.parse_args()

    log("[1/3] 抓远端 saying.json（权威列表）")
    remote, src_url, etag = fetch_remote_saying()

    log("[2/3] 抓前端内置兜底数组")
    local = extract_from_frontend(args.deep)
    log(f"  前端内置 {len(local)} 条")

    log("[3/3] 合并去重并写出")
    items = merge(remote, local)
    if not items:
        log("没抓到任何条目。检查网络后重试。")
        return 1
    write_outputs(items, args.outdir, etag, src_url)

    only_js = sum(1 for d in items if d["source"] == "js")
    log(f"\n完成：共 {len(items)} 条  ->  {args.outdir}")
    log(f"  远端 {len(remote)} 条 / 内置 {len(local)} 条 / 仅内置独有 {only_js} 条")
    log("  kook_台词.txt  kook_台词.csv  kook_台词.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
