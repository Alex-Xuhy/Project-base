#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 5 步 —— 专题检验：回答具体问题：利率传导、加卫星、池内热点

⚠ 依赖第 2 步的 `T_taxonomy.csv`。这三个是**追加的专题**，
  互不依赖，可以只跑其中一个。

用法：
  python "0. Scripts 脚本/5. 专题检验/run.py"                  # 按序跑本步全部
  python "0. Scripts 脚本/5. 专题检验/run.py" --list           # 只列清单，不跑
  python "0. Scripts 脚本/5. 专题检验/run.py" --only 5a   # 只跑某一个
  python "0. Scripts 脚本/5. 专题检验/run.py" --keep-going     # 出错也继续（默认遇错即停）
  python "0. Scripts 脚本/5. 专题检验/run.py" -- --token XXX   # `--` 之后的参数透传给每个子脚本

子脚本一律以**独立进程**运行 —— 每个脚本自己用 `__file__` 解析项目根，
互不共享模块状态，所以从任何工作目录调用都可以。

产物：`5. 专题检验/` 下的 csv/ 与 fig/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (匹配键, 文件名, 一句话说明)
STEPS = [
    ("5a", "5a_利率美元黄金.py",
     "加息 → 美元 / 黄金：同期 vs 预测，两段弱链接"),
    ("5b", "5b_加TLT_UUP.py",
     "卫星加 TLT / UUP 值不值：保底检验与账户口径代价"),
    ("5c", "5c_池内热点异动.py",
     "池内热点与成交额异动（成交额 ≠ 净流入）"),
]


def resolve(key):
    """按键 / 文件名 / 文件名前缀匹配，返回唯一条目。"""
    hits = [s for s in STEPS
            if s[0] == key or s[1] == key or s[1].startswith(key)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        sys.exit(f"!! 没有匹配「{key}」的子脚本。可用："
                 + "、".join(s[0] for s in STEPS))
    sys.exit(f"!! 「{key}」匹配到多个："
             + "、".join(s[1] for s in hits))


def main():
    ap = argparse.ArgumentParser(
        description="第 5 步 —— 专题检验",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="只列清单，不运行")
    ap.add_argument("--only", metavar="KEY", help="只运行指定子脚本（如 5a）")
    ap.add_argument("--keep-going", action="store_true",
                    help="某个子脚本失败也继续（默认遇错即停）")
    args, passthru = ap.parse_known_args()
    passthru = [x for x in passthru if x != "--"]

    todo = [resolve(args.only)] if args.only else STEPS

    print("=" * 78)
    print(f"第 5 步 —— 专题检验")
    print(f"  产物目录 {os.path.join(os.path.dirname(os.path.dirname(HERE)), '5. 专题检验')}")
    if passthru:
        print(f"  透传参数 {' '.join(passthru)}")
    print("=" * 78)
    for key, fn, desc in todo:
        print(f"  {key:<4}{fn:<26}{desc}")
    if args.list:
        return 0

    failed = []
    for key, fn, desc in todo:
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            sys.exit(f"!! 找不到 {path}")
        print("")
        print("-" * 78)
        print(f">>> {key}  {fn} —— {desc}")
        print("-" * 78)
        r = subprocess.run([sys.executable, path] + passthru)
        if r.returncode != 0:
            failed.append(key)
            print(f"\n!! {fn} 退出码 {r.returncode}")
            if not args.keep_going:
                print("   已中止本步剩余脚本（下游会读到半截产物，算出静默错误的数字）。")
                print("   要强行跑完加 --keep-going。")
                return 1

    print("")
    print("=" * 78)
    if failed:
        print(f"第 5 步结束 —— {len(failed)} 个失败：{'、'.join(failed)}")
        return 1
    print(f"第 5 步全部完成 —— {len(todo)} 个子脚本")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
