#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全球资产配置模拟 —— 按 1→5 顺序运行全部 5 个步骤。

每个步骤都是一个独立的 run.py，这里只是顺序调用它们。

用法：
  python "0. Scripts 脚本/run_all.py"                 # 跑全部 5 步
  python "0. Scripts 脚本/run_all.py" --list          # 只列清单，不跑
  python "0. Scripts 脚本/run_all.py" --only 3        # 只跑第 3 步
  python "0. Scripts 脚本/run_all.py" --skip 1        # 跳过第 1 步（没 token 时最常用）
  python "0. Scripts 脚本/run_all.py" -- --token XXX  # `--` 之后的参数透传给每一步

⚠ 第 1 步要联网、1a 还要 token。只想刷新分析结果就加 `--skip 1`。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (序号, 文件夹, 一句话说明)
STEPS = [
    ("1", "1. 取数与复权",
     "拉数据、拆股复权、补站点滞后的最近几日"),
    ("2", "2. 资产池认知",
     "认识这 79 只标的：描述统计、分类、冗余度"),
    ("3", "3. 大类与状态检验",
     "大类配置、加息冲击、缺口复核、状态分档 —— 全是研究"),
    ("4", "4. 组合构建",
     "把研究结论落成一个能下单的持仓"),
    ("5", "5. 专题检验",
     "回答具体问题：利率传导、加卫星、池内热点"),
]


def main():
    ap = argparse.ArgumentParser(description="按 1→5 顺序运行全部 5 个步骤")
    ap.add_argument("--list", action="store_true", help="只列清单，不运行")
    ap.add_argument("--only", metavar="N", help="只运行第 N 步")
    ap.add_argument("--skip", metavar="N", action="append", default=[],
                    help="跳过第 N 步，可重复")
    ap.add_argument("--keep-going", action="store_true", help="某步失败也继续")
    args, passthru = ap.parse_known_args()
    passthru = [x for x in passthru if x != "--"]

    todo = [s for s in STEPS
            if (not args.only or s[0] == args.only) and s[0] not in args.skip]
    if not todo:
        sys.exit("!! 没有要跑的步骤（检查 --only / --skip）")

    print("=" * 78)
    print("全球资产配置模拟 —— 5 个步骤")
    for n, folder, blurb in STEPS:
        mark = "  " if (n, folder, blurb) in todo else "跳过"
        print(f"  {mark} {n}. {folder.split('. ', 1)[1]:<16}{blurb}")
    if args.list:
        return 0

    for n, folder, blurb in todo:
        path = os.path.join(HERE, folder, "run.py")
        if not os.path.exists(path):
            sys.exit(f"!! 找不到 {path}")
        print("")
        print("#" * 78)
        print(f"# 第 {n} 步 —— {blurb}")
        print("#" * 78)
        r = subprocess.run([sys.executable, path] + passthru)
        if r.returncode != 0:
            print(f"\n!! 第 {n} 步失败（退出码 {r.returncode}）")
            if not args.keep_going:
                print("   已中止。要强行跑完加 --keep-going。")
                return 1

    print("")
    print("=" * 78)
    print("全部完成")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
