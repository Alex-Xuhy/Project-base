# -*- coding: utf-8 -*-
"""
用免费替代字体替换「华文系列」

华文系列（华文楷体/行楷/隶书/新魏/中宋/彩云）是常州华文（SinoType）的商业字体，
Windows 不自带、也没有官方免费下载，正规渠道是随 Microsoft Office 分发。
本脚本装的是**风格相近的免费替代字体**，不是原版。

安装位置：%LOCALAPPDATA%\\Microsoft\\Windows\\Fonts（用户级，无需管理员，不污染系统）

用法:
  python 安装中文字体.py --list     只看方案，不下载
  python 安装中文字体.py            执行安装
  python 安装中文字体.py --check    检查哪些已装
  python 安装中文字体.py --uninstall 卸载本脚本装的字体
"""
from __future__ import annotations

import argparse
import ctypes
import io
import os
import re
import struct
import sys
import tarfile
import urllib.request

# --------------------------------------------------------------------------
# 方案表
#   npm     : npm 包名，脚本自动解析最新版 tarball
#   member  : tarball 内的字体文件路径
#   url     : 直接下载地址（与 npm 二选一）
# --------------------------------------------------------------------------
SPECS = [
    {
        "target": "华文楷体",
        "name": "霞鹜文楷 LXGW WenKai",
        "license": "SIL OFL 1.1（可商用）",
        "npm": "@fontpkg/lxgw-wen-kai",
        "member": "package/LXGWWenKai-Regular.ttf",
        "note": "最接近的免费楷体，字形温润",
    },
    {
        "target": "华文行楷",
        "name": "演示夏行楷",
        "license": "免费商用",
        "npm": "@fontpkg/slidexiaxing",
        "member": "package/演示夏行楷.ttf",
        "note": "行楷风格，简体",
    },
    {
        "target": "华文隶书",
        "name": "阿里妈妈刀隶体",
        "license": "免费商用",
        "npm": "@fontpkg/alimama-dao-li-ti",
        "member": "package/AlimamaDaoLiTi.ttf",
        "note": "隶楷之间，取爨宝子碑笔意",
    },
    {
        "target": "华文新魏",
        "name": "王汉宗中魏碑简体",
        "license": "GNU GPL v2（真正开源）",
        "url": "https://raw.githubusercontent.com/cghio/wangfonts/master/fonts/wts43.ttf",
        "member": None,
        "note": "魏碑风格，简体字形",
    },
    {
        "target": "华文中宋",
        "name": "思源宋体 SC Medium",
        "license": "SIL OFL 1.1（可商用）",
        "npm": "@fontpkg/source-han-serif-sc",
        "member": "package/SourceHanSerifSC-Medium.otf",
        "note": "中宋对应「中等字重宋体」，Medium 最贴近",
    },
]

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128.0.0.0 Safari/537.36"
FONT_DIR = os.path.join(os.environ["LOCALAPPDATA"], "Microsoft", "Windows", "Fonts")
REG_KEY = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"


def log(msg=""):
    print(msg, flush=True)


def http_get(url: str, timeout=90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def mirrors(url: str) -> list[str]:
    """raw.githubusercontent.com 在国内常年不通，补一个 jsDelivr 镜像。"""
    m = re.match(r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)", url)
    if m:
        owner, repo, br, path = m.groups()
        return [url, f"https://cdn.jsdelivr.net/gh/{owner}/{repo}@{br}/{path}"]
    return [url]


def http_get_first(url: str) -> bytes:
    last = None
    for u in mirrors(url):
        try:
            return http_get(u)
        except Exception as e:
            last = e
            log(f"    [!] {u.split('/')[2]} 失败: {e}")
    raise last if last else RuntimeError("无可用地址")


# --------------------------------------------------------------------------
# 字体名解析：直接读 sfnt 的 name 表，不依赖 fontTools
# --------------------------------------------------------------------------
def sfnt_kind(data: bytes) -> str | None:
    if len(data) < 4:
        return None
    tag = data[:4]
    if tag == b"\x00\x01\x00\x00":
        return "ttf"
    if tag == b"OTTO":
        return "otf"
    if tag == b"ttcf":
        return "ttc"
    if tag == b"true":
        return "ttf"
    return None


def read_names(data: bytes) -> dict[int, str]:
    """返回 {nameID: 字符串}，优先取 Windows 平台的记录。"""
    if len(data) < 12:
        return {}
    num_tables = struct.unpack(">H", data[4:6])[0]
    name_off = None
    for i in range(num_tables):
        rec = 12 + i * 16
        if rec + 16 > len(data):
            break
        tag = data[rec:rec + 4]
        off, length = struct.unpack(">II", data[rec + 8:rec + 16])
        if tag == b"name":
            name_off = (off, length)
            break
    if not name_off:
        return {}
    off = name_off[0]
    if off + 6 > len(data):
        return {}
    count, str_off = struct.unpack(">HH", data[off + 2:off + 6])
    str_base = off + str_off
    best: dict[int, tuple[int, str]] = {}
    for i in range(count):
        rec = off + 6 + i * 12
        if rec + 12 > len(data):
            break
        pid, eid, lid, nid, length, soff = struct.unpack(">HHHHHH", data[rec:rec + 12])
        raw = data[str_base + soff: str_base + soff + length]
        if pid == 3:              # Windows
            try:
                s = raw.decode("utf-16-be")
            except UnicodeDecodeError:
                continue
            # 中文系统上优先用 zh-CN 的名字（Windows 自己也是这么选的）
            score = {0x804: 4, 0x409: 3}.get(lid, 2)
        elif pid == 1:            # Mac
            try:
                s = raw.decode("mac-roman")
            except UnicodeDecodeError:
                continue
            score = 1
        else:
            continue
        s = s.replace("\x00", "").strip()
        if s and (nid not in best or score > best[nid][0]):
            best[nid] = (score, s)
    return {k: v[1] for k, v in best.items()}


def registry_name(data: bytes, kind: str) -> tuple[str, str]:
    """返回 (注册表值名, 字体族名)。"""
    names = read_names(data)
    family = names.get(16) or names.get(1) or "Unknown"
    sub = names.get(17) or names.get(2) or "Regular"
    suffix = "OpenType" if kind == "otf" else "TrueType"
    if sub.lower() in ("regular", "normal", ""):
        return f"{family} ({suffix})", family
    # Windows 对非 Regular 字重命名为 "族名 字重"
    if family.lower().endswith(sub.lower()):
        return f"{family} ({suffix})", family
    return f"{family} {sub} ({suffix})", f"{family} {sub}"


# --------------------------------------------------------------------------
# 下载
# --------------------------------------------------------------------------
def fetch_spec(spec: dict) -> tuple[bytes, str]:
    """返回 (字体字节, 文件名)。"""
    if spec.get("url"):
        data = http_get_first(spec["url"])
        return data, os.path.basename(spec["url"])

    pkg = spec["npm"].replace("/", "%2f")
    meta = http_get(f"https://registry.npmjs.org/{pkg}")
    import json
    info = json.loads(meta)
    ver = info["dist-tags"]["latest"]
    tarball = info["versions"][ver]["dist"]["tarball"]
    log(f"    包 {spec['npm']}@{ver}")
    blob = http_get(tarball)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        try:
            f = tf.extractfile(spec["member"])
        except KeyError:
            names = [m.name for m in tf.getmembers() if m.name.lower().endswith((".ttf", ".otf"))]
            raise SystemExit(f"    [!] tarball 里没有 {spec['member']}，实际有: {names}")
        if f is None:
            raise SystemExit(f"    [!] 无法读取 {spec['member']}")
        return f.read(), os.path.basename(spec["member"])


# --------------------------------------------------------------------------
# 安装 / 卸载
# --------------------------------------------------------------------------
def installed_families() -> set[str]:
    import winreg
    out = set()
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_KEY) as k:
            i = 0
            while True:
                try:
                    n, v, _ = winreg.EnumValue(k, i)
                except OSError:
                    break
                out.add(n)
                i += 1
    except FileNotFoundError:
        pass
    return out


def install_one(spec: dict, dry: bool) -> bool:
    log(f"\n▸ {spec['target']}  →  {spec['name']}")
    log(f"    许可: {spec['license']}")
    try:
        data, fname = fetch_spec(spec)
    except Exception as e:
        log(f"    [!] 下载失败: {e}")
        return False

    kind = sfnt_kind(data)
    if not kind:
        log("    [!] 下载到的不是有效字体文件，跳过")
        return False

    reg_name, family = registry_name(data, kind)
    log(f"    文件: {fname}  ({len(data)/1048576:.1f} MB, {kind.upper()})")
    log(f"    字体族: {family}")

    if dry:
        log(f"    [dry-run] 将注册为: {reg_name}")
        return True

    os.makedirs(FONT_DIR, exist_ok=True)
    dest = os.path.join(FONT_DIR, fname)
    with open(dest, "wb") as f:
        f.write(data)

    import winreg
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, REG_KEY) as k:
        winreg.SetValueEx(k, reg_name, 0, winreg.REG_SZ, dest)

    log(f"    已安装 → {dest}")
    return True


def broadcast_fontchange():
    """通知系统字体有变动。

    必须用 SendNotifyMessage 而不是 SendMessage：SendMessage 对 HWND_BROADCAST
    是同步的，会一直等到所有顶层窗口处理完消息，只要有一个窗口卡住这里就死等。
    SendNotifyMessage 对跨线程窗口立即返回，不会阻塞。
    """
    try:
        HWND_BROADCAST, WM_FONTCHANGE = 0xFFFF, 0x001D
        ctypes.windll.user32.SendNotifyMessageW(HWND_BROADCAST, WM_FONTCHANGE, 0, 0)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="安装华文系列的免费替代字体")
    ap.add_argument("--list", action="store_true", help="只列出方案")
    ap.add_argument("--check", action="store_true", help="检查已装情况")
    ap.add_argument("--uninstall", action="store_true", help="卸载本脚本装的字体")
    ap.add_argument("--dry-run", action="store_true", help="下载并识别，但不写入")
    ap.add_argument("--only", default="", help="只处理目标名包含该串的项，如 --only 新魏")
    args = ap.parse_args()

    specs = [s for s in SPECS if args.only in s["target"]] if args.only else SPECS
    if args.only and not specs:
        log(f"没有匹配「{args.only}」的项。可用: {[s['target'] for s in SPECS]}")
        return 1

    log("华文系列（常州华文 SinoType 商业字体）的免费替代方案")
    log("=" * 66)
    log(f"{'目标':<8}{'替代字体':<22}{'许可'}")
    log("-" * 66)
    for s in SPECS:
        log(f"{s['target']:<8}{s['name']:<22}{s['license']}")
    log("-" * 66)
    log("华文彩云：空心装饰体，无风格相近的免费替代，未列入。")
    log(f"\n安装目录: {FONT_DIR}")

    if args.list:
        return 0

    if args.uninstall:
        import winreg
        removed = 0
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_KEY, 0, winreg.KEY_ALL_ACCESS) as k:
            for name in [s["name"] for s in SPECS]:
                i = 0
                while True:
                    try:
                        rn, rv, _ = winreg.EnumValue(k, i)
                    except OSError:
                        break
                    if os.path.basename(rv).lower() in (
                        os.path.basename(s["member"] or s.get("url", "")).lower()
                        for s in SPECS
                    ):
                        winreg.DeleteValue(k, rn)
                        if os.path.exists(rv):
                            os.remove(rv)
                        log(f"  已卸载 {rn}")
                        removed += 1
                        continue
                    i += 1
        broadcast_fontchange()
        log(f"\n共卸载 {removed} 项。")
        return 0

    if args.check:
        have = installed_families()
        log("")
        for s in SPECS:
            hit = [n for n in have if s["name"].split()[0].lower() in n.lower()]
            log(f"  {'✓' if hit else '✗'} {s['target']:<8} {s['name']}")
        return 0

    ok = 0
    for s in specs:
        if install_one(s, args.dry_run):
            ok += 1

    if not args.dry_run:
        broadcast_fontchange()

    log(f"\n{'=' * 66}")
    if args.dry_run:
        log(f"dry-run：{ok}/{len(specs)} 个字体可正常下载解析，未做任何写入。")
        log("去掉 --dry-run 即可真正安装。")
    else:
        log(f"完成：{ok}/{len(specs)} 个字体已安装。")
        log("新开的程序能直接用；已经开着的 Word/PS 等需要重启才能看到。")
    return 0 if ok == len(specs) else 1


if __name__ == "__main__":
    sys.exit(main())
