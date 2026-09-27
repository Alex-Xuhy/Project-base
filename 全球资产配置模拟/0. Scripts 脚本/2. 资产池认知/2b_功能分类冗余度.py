# -*- coding: utf-8 -*-
"""全球资产配置模拟 —— 2b：资产池分类、明细清单与冗余度

用户需求：80 只标的全英文、多数没见过 → 要一张**分类图** + 一份**中文明细清单**。

本脚本做三件事：
  1. 建**功能分类**（不是照抄站点分类 —— 站点分类有 15 处会误导策略，见 SITE_TRAP）
  2. 算每只标的的真实特征：年化波动 / 年化收益 / 流动性 / 与全池的相关（= 冗余度）
  3. 出两张图：① 功能子类结构（只数 + 波动区间）  ② 功能子类相关热图

产出：
  T_taxonomy.csv        79 只标的的完整明细
  T_structure.png       分类图（左：只数；右：子类内波动区间）
  T_corr_cluster.png    功能子类相关热图（按聚类排序）
  T_redundancy.csv      功能子类的冗余度汇总

用法：
  python 2b_功能分类冗余度.py
"""

from __future__ import annotations

import os
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # 脚本在「0. Scripts 脚本/<步骤>/」，上跳两层才是项目根
DATA_DIR = os.path.join(ROOT, "1. Data 数据")     # 共享缓存（第 1 步产物），全步骤只读
STEP_DIR = os.path.join(ROOT, "2. 资产池认知")   # 本步骤的产物目录
# 产物按类型分家：表进 csv/，图进 fig/
OUT_CSV, OUT_FIG = os.path.join(STEP_DIR, "csv"), os.path.join(STEP_DIR, "fig")

p = print

# ---- 绘图约定（沿用本项目既有口径） --------------------------------------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
PLANE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"

# 配色：参考调色板的官方**槽位顺序**（顺序本身就是 CVD 安全机制，别改序）
# 已用 validate_palette.js 验证：4 槽 / 5 槽 在 adjacent 语境下全部 PASS
CLR = {
    "权益": "#2a78d6",   # slot 1 blue
    "商品": "#eb6834",   # slot 2 orange
    "固收": "#1baf7a",   # slot 3 aqua
    "货币": "#eda100",   # slot 4 yellow
    "加密": "#4a3aa7",   # slot 7 violet
}
CLS_ORDER = ["权益", "商品", "固收", "货币", "加密"]
# 子类的**阅读顺序**：从「最熟悉的美股」一路走到「最陌生的货币」，而不是按波动排
SUB_ORDER = ["美股宽基", "发达市场", "新兴市场", "美股行业", "资源主题股", "科技主题股",
             "债宽基", "信用债", "国债短", "国债中长", "国债长",
             "贵金属", "能源", "农产品", "商品宽基",
             "加密", "美元", "其他货币"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


# ================================================================ 分类体系
# 站点只给了 4 大类 / 11 子类，粒度太粗，且有 5 处会误导策略（见 SITE_TRAP）。
# 下面按「**这只 ETF 实际靠什么赚钱**」重建功能分类 —— 这正是策略要用的粒度。
#
# 格式: 代码 -> (功能大类, 功能子类, 中文名, 一句话说明)
TAXONOMY: dict[str, tuple[str, str, str, str]] = {
    # --- 权益 / 美股宽基 (8) ---
    "SPY":  ("权益", "美股宽基", "标普500",        "美国大盘蓝筹，全池的风险中枢"),
    "QQQ":  ("权益", "美股宽基", "纳斯达克100",    "美国科技龙头，与 SPY 高度重叠但更偏成长"),
    "ONEQ": ("权益", "美股宽基", "纳斯达克综合",   "比 QQQ 更宽，含小盘，与 QQQ 几乎同涨同跌"),
    "DIA":  ("权益", "美股宽基", "道琼斯工业30",   "30 只老牌工业蓝筹，价格加权，与 SPY 重叠"),
    "IWM":  ("权益", "美股宽基", "罗素2000 小盘",  "美国小盘股，弹性大于 SPY"),
    "IWN":  ("权益", "美股宽基", "罗素2000 小盘价值", "小盘里的便宜股，偏金融/工业"),
    "DSI":  ("权益", "美股宽基", "标普400 社会责任", "中盘 ESG 筛选，实质仍是美股中盘"),
    "VYM":  ("权益", "美股宽基", "美股高股息",     "高分红蓝筹，偏防守，利率敏感"),

    # --- 权益 / 发达市场 (9) ---
    "EFA":  ("权益", "发达市场", "EAFE 发达市场",  "欧澳远东（不含美国），组合里的非美分散"),
    "VGK":  ("权益", "发达市场", "欧洲",           "欧洲整体，与 EFA 高度重叠"),
    "EWG":  ("权益", "发达市场", "德国",           "德国 DAX，出口/制造业"),
    "EWU":  ("权益", "发达市场", "英国",           "英国富时，偏能源与金融"),
    "EPP":  ("权益", "发达市场", "亚太除日本",     "澳洲+香港+新加坡，实为澳港星"),
    "EWJ":  ("权益", "发达市场", "日本",           "日本大盘，与日元汇率强相关"),
    "EWY":  ("权益", "发达市场", "韩国",           "韩国，实质是三星+海力士的半导体代理"),
    "EWT":  ("权益", "发达市场", "台湾",           "台湾，实质是台积电的半导体代理"),
    "EWA":  ("权益", "发达市场", "澳大利亚",       "澳洲，银行+矿业，商品货币属性"),

    # --- 权益 / 新兴市场 (5) ---
    "EEM":  ("权益", "新兴市场", "新兴市场整体",   "新兴市场宽基，与全球风险偏好同步"),
    "BKF":  ("权益", "新兴市场", "金砖四国",       "中巴印俄，成交极清淡，是 EEM 的窄化版"),
    "FXI":  ("权益", "新兴市场", "中国大盘",       "港股中概大盘，政策敏感"),
    "EWZ":  ("权益", "新兴市场", "巴西",           "巴西，铁矿石+石油，商品属性重"),
    "INDA": ("权益", "新兴市场", "印度",           "印度，内需驱动，与其它新兴市场相关低"),

    # --- 权益 / 美股行业 (14) ---
    "XLF":  ("权益", "美股行业", "金融",           "银行+保险，利率上行受益"),
    "XLE":  ("权益", "美股行业", "能源",           "石油天然气公司，跟随油价"),
    "XLI":  ("权益", "美股行业", "工业",           "制造业+运输+军工，周期属性"),
    "XLB":  ("权益", "美股行业", "材料",           "化工+金属，跟随工业需求"),
    "XLP":  ("权益", "美股行业", "必需消费",       "食品饮料日化，典型防守"),
    "XLY":  ("权益", "美股行业", "可选消费",       "零售+汽车+餐饮，典型进攻"),
    "XLC":  ("权益", "美股行业", "通信",           "谷歌/Meta/奈飞，实为互联网平台"),
    "XLK":  ("权益", "美股行业", "科技",           "软件+硬件+半导体，与 QQQ 高度重叠"),
    "XLV":  ("权益", "美股行业", "医疗",           "药企+器械+保险，防守偏成长"),
    "XLU":  ("权益", "美股行业", "公用事业",       "电力水务，类债券，利率敏感"),
    "XLRE": ("权益", "美股行业", "房地产(标普)",   "REITs，利率敏感，行为夹在股债之间"),
    "VNQ":  ("权益", "美股行业", "房地产(全市场)", "REITs 宽基，与 XLRE 高度重叠"),
    "XHB":  ("权益", "美股行业", "住宅建筑",       "美国建商，利率敏感度全池最高之一"),
    "XBI":  ("权益", "美股行业", "生物科技",       "中小型生科，并购预期驱动，波动极高"),

    # --- 权益 / 资源主题（名义是股票，实际是商品代理）(8) ---
    "XME":  ("权益", "资源主题股", "金属与矿业",   "美加矿企，跟随金属价格"),
    "GDX":  ("权益", "资源主题股", "金矿",         "金矿股，是黄金的**加杠杆版**"),
    "SIL":  ("权益", "资源主题股", "银矿",         "银矿股，白银的高波动代理"),
    "SLX":  ("权益", "资源主题股", "钢铁",         "全球钢铁，跟随钢价与基建预期"),
    "URA":  ("权益", "资源主题股", "铀",           "铀矿+核电，供给端事件驱动"),
    "MOO":  ("权益", "资源主题股", "农业",         "种子+化肥+农机，跟随农产品"),
    "IGE":  ("权益", "资源主题股", "自然资源",     "油气+矿业宽基，是 XLE 与 XME 的混合"),
    "PIO":  ("权益", "资源主题股", "水",           "水务公用事业，流动性极差，实为低波防守"),

    # --- 权益 / 科技与其他主题 (13) ---
    "BLOK": ("权益", "科技主题股", "区块链/数字支付", "交易所+矿企+支付，是比特币的股票代理"),
    "HACK": ("权益", "科技主题股", "网络安全",     "网安软件，与 XLK 同向"),
    "IGF":  ("权益", "科技主题股", "全球基础设施", "收费公路/机场/电网，类债券防守"),
    "MJ":   ("权益", "科技主题股", "大麻",         "大麻种植商，监管驱动，波动全池最高档"),
    "ROBO": ("权益", "科技主题股", "机器人与自动化", "工业自动化，与 XLI 同向"),
    "SOCL": ("权益", "科技主题股", "社交媒体",     "社交平台，与 XLC 高度重叠"),
    "ERTH": ("权益", "科技主题股", "环保可持续",   "绿色主题宽基，流动性全场最差"),
    "PSP":  ("权益", "科技主题股", "上市私募股权", "PE 公司，金融属性重于科技"),
    "PBD":  ("权益", "科技主题股", "清洁能源",     "可再生能源宽基，与 TAN/FAN 重叠"),
    "TAN":  ("权益", "科技主题股", "太阳能",       "光伏，利率与补贴双敏感"),
    "FAN":  ("权益", "科技主题股", "风能",         "风电，流动性与 PBD 同档偏低"),
    "LIT":  ("权益", "科技主题股", "锂电与电池",   "锂矿+电池厂，跟随锂价"),
    "REMX": ("权益", "科技主题股", "稀土",         "稀土开采，跟着锂电与供给政策走"),

    # --- 固收 (6) ---
    "AGG":  ("固收", "债宽基", "美国综合债券",     "全市场债券宽基，久期约 6 年"),
    "LQD":  ("固收", "信用债", "投资级公司债",     "高评级公司债，久期长于 AGG"),
    "HYG":  ("固收", "信用债", "高收益债",         "垃圾债，行为更像股票而非债券"),
    "SHY":  ("固收", "国债短", "1-3年美债",       "超短国债，接近现金，全池最稳"),
    "IEF":  ("固收", "国债中长", "7-10年美债",     "中久期国债，纯利率风险"),
    "TLT":  ("固收", "国债长", "20年+美债",       "长久期国债，**波动接近股票**，别当避险用"),

    # --- 商品 (7) ---
    "GLD":  ("商品", "贵金属", "黄金",             "无息资产，避险+抗通胀"),
    "SLV":  ("商品", "贵金属", "白银",             "白银，工业属性重，波动约黄金 1.5 倍"),
    "USO":  ("商品", "能源", "原油",               "WTI 原油期货，展期损耗大"),
    "UNG":  ("商品", "能源", "天然气",             "天然气期货，波动与损耗均为全池极端"),
    "DBA":  ("商品", "农产品", "农产品",           "玉米/大豆/小麦/糖"),
    "USCI": ("商品", "商品宽基", "商品指数(USCI)", "多品种商品指数，与 DBC 重叠"),
    "DBC":  ("商品", "商品宽基", "商品指数(DBC)",  "能源权重约 60%，实质是原油代理"),

    # --- 加密 (1) ---
    "IBIT": ("加密", "加密", "比特币",             "**站点误归到商品**，实为独立风险源，与股/债/金都低相关"),

    # --- 货币 (8) ---
    "UUP":  ("货币", "美元", "美元(做多)",         "做多美元指数，与风险资产负相关"),
    "UDN":  ("货币", "美元", "美元(做空)",         "**做空**美元 —— 与 UUP 完全负相关，不能同格等权"),
    "FXY":  ("货币", "其他货币", "日元",           "日元，套息交易的反向指标"),
    "FXE":  ("货币", "其他货币", "欧元",           "欧元，美元指数的最大对手盘"),
    "FXB":  ("货币", "其他货币", "英镑",           "英镑"),
    "FXF":  ("货币", "其他货币", "瑞郎",           "瑞郎，传统避险货币"),
    "FXC":  ("货币", "其他货币", "加元",           "加元，跟随油价"),
    "FXA":  ("货币", "其他货币", "澳元",           "澳元，跟随大宗商品，流动性极差"),
}

# 站点分类里会误导策略的 5 处（写进清单的「备注」列）
SITE_TRAP = {
    "IBIT": "站点归「商品/其他商品」—— 实为加密资产，风险来源与商品无关",
    "GDX":  "站点归「股权/其他市场」—— 实际是黄金的杠杆代理，不是股票",
    "SIL":  "站点归「股权/其他市场」—— 实际是白银的杠杆代理",
    "XME":  "站点归「股权/发达市场」—— 实际跟随金属价格",
    "SLX":  "站点归「股权/其他市场」—— 实际跟随钢价",
    "URA":  "站点归「股权/其他市场」—— 实际是铀价/核电政策代理",
    "MOO":  "站点归「股权/其他市场」—— 实际跟随农产品",
    "IGE":  "站点归「股权/发达市场」—— 实为 XLE+XME 的混合，非宽基股票",
    "TLT":  "站点归「固收/国家债券」—— 但久期 20 年+，波动接近股票，不是避险资产",
    "XLRE": "站点归「股权」—— 但 REITs 是利率敏感资产，行为夹在股债之间",
    "VNQ":  "站点归「股权」—— 同 XLRE，且与之高度重叠",
    "UDN":  "站点归「货币/美元」与 UUP 同格 —— 但它是**做空**美元，两者完全负相关",
    "HYG":  "站点归「固收/公司债券」—— 但高收益债行为更像股票，与 LQD 不同类",
    "BKF":  "站点归「新兴市场」—— 成交极清淡（日均不足 20 万美元），实为 EEM 的窄化版",
    "DSI":  "站点归「发达市场」—— 实为美股中盘 ESG，与 SPY/IWM 重叠",
}

CLC = "  "          # 控制台列间距
NUM = "  "          # fig 编号


# ================================================================ 载入

def load():
    px = pd.read_csv(os.path.join(DATA_DIR, "wide_close_adj_usd.csv"),
                     encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    meta = pd.read_csv(os.path.join(DATA_DIR, "stocks.csv"),
                       encoding="utf-8-sig").set_index("symbol")
    last_valid = px.apply(lambda s: s.last_valid_index())
    keep = [c for c in px.columns
            if last_valid[c] >= px.index.max() - pd.Timedelta(days=10)]
    px, meta = px[keep], meta.loc[keep]

    long = pd.read_csv(os.path.join(DATA_DIR, "prices_adj.csv"),
                       encoding="utf-8-sig",
                       usecols=["symbol", "dataDatetime", "lastPrice", "volume"],
                       parse_dates=["dataDatetime"])
    return px, meta, long


# ================================================================ 特征

def per_asset_stats(px, long):
    """每只标的的真实特征。近 3 年为主口径（匹配当前市场状态）。"""
    ret = px.pct_change()
    t3 = px.index.max() - pd.DateOffset(years=3)
    r3, p3 = ret.loc[ret.index >= t3], px.loc[px.index >= t3]

    pool = ret.mean(axis=1)                       # 全池等权日收益
    spy = ret["SPY"] if "SPY" in ret else pool

    rows = []
    for c in px.columns:
        s3 = r3[c].dropna()
        ok = len(s3) >= 250                       # 上市满 1 年的才给 3 年口径
        # 年化收益：用区间累计，避免日频均值被极端值污染
        cum = (1 + s3).prod() - 1
        yrs = len(s3) / 252
        rows.append({
            "symbol": c,
            "年化波动": s3.std() * sqrt(252) if ok else np.nan,
            "年化收益": (1 + cum) ** (1 / yrs) - 1 if ok and yrs > 0.5 else np.nan,
            "与SPY相关": s3.corr(spy.reindex(s3.index)) if ok else np.nan,
            "与全池相关": s3.corr(pool.reindex(s3.index)) if ok else np.nan,
            "历史长度": len(ret[c].dropna()) / 252,
        })
    S = pd.DataFrame(rows).set_index("symbol")

    # 流动性：日均成交额（美元）。volume 已在 1b 做拆股反向缩放，
    # 故 volume × 未复权价 = 真实美元成交额；这里用复权价×复权量同样等于真实额。
    l3 = long[long["dataDatetime"] >= t3].copy()
    dv = (l3["lastPrice"] * l3["volume"]).groupby(l3["symbol"]).mean() / 1e6
    S["日均成交额"] = S.index.map(dv)
    return S


def cluster_order(corr):
    """按层次聚类给相关矩阵排个序，让高相关的挤在一起。"""
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        d = 1 - corr.values
        np.fill_diagonal(d, 0.0)
        d = (d + d.T) / 2
        return [corr.columns[i] for i in leaves_list(linkage(squareform(d, checks=False), "average"))]
    except Exception as e:                                    # scipy 缺失时退化为人工顺序
        p(f"  [注意] scipy 聚类不可用（{e}），改用平均相关排序")
        return list(corr.mean().sort_values(ascending=False).index)


# ================================================================ 图 1：结构

def fig_structure(S, tax, order, sub_meta):
    ncol = 2
    fig, axes = plt.subplots(1, ncol, figsize=(13.5, 8.4), dpi=150,
                             gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.05})
    fig.patch.set_facecolor(PLANE)
    y = np.arange(len(order))[::-1]

    # ---- 左：只数 ----
    ax = axes[0]
    ax.set_facecolor(PLANE)
    cnt = [sub_meta[s]["n"] for s in order]
    cols = [CLR[sub_meta[s]["cls"]] for s in order]
    ax.barh(y, cnt, height=0.62, color=cols, zorder=3)
    for yi, c in zip(y, cnt):
        ax.text(c + 0.22, yi, f"{c}", va="center", ha="left",
                fontsize=9.5, color=INK, fontweight="bold", zorder=4)
    ax.set_xlim(0, max(cnt) + 2.2)
    ax.set_title("只数", fontsize=11, color=INK2, pad=10, loc="left")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s}{CLC}{sub_meta[s]['cls']}" for s in order],
                       fontsize=10, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8.5, colors=INK3, length=0)
    ax.grid(axis="x", color="#e6e5e1", lw=1, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#dcdbd6")

    # ---- 右：子类内每只标的的波动 ----
    ax = axes[1]
    ax.set_facecolor(PLANE)
    for k, (yi, s) in enumerate(zip(y, order)):
        mem = [m for m in tax.index if tax.loc[m, "功能子类"] == s]
        v = S.loc[mem, "年化波动"].dropna().sort_values()
        if v.empty:
            continue
        c = CLR[sub_meta[s]["cls"]]
        ax.plot([v.min(), v.max()], [yi, yi], color=c, lw=1.6, alpha=.20,
                solid_capstyle="round", zorder=2)
        # 成员各画一个点（上下交错，避免同值重叠），直接展示子类内的离散程度。
        # 大号实心点 = 中位，是这一行的代表值。
        for j, (nm, x) in enumerate(v.items()):
            yj = yi + (0.17 if j % 2 else -0.17)
            ax.scatter(x, yj, s=17, color=c,
                       alpha=.62, zorder=3, edgecolors=PLANE, linewidths=0.7)
            if x > 0.55:      # 只标真正的极端值（MJ/UNG），否则整张图都是字
                ax.text(x, yj - 0.30, nm, fontsize=7.6, color=INK2,
                        ha="center", va="center", zorder=6)
        ax.scatter(v.median(), yi, s=62, color=c, zorder=5,
                   edgecolors=PLANE, linewidths=1.7)
        # ⚠ 标签必须紧跟**中位点**。挂在区间右端会被读成最大值（曾如此）。
        ax.text(v.median() + 0.013, yi, f"{v.median():.1%}", va="center",
                ha="left", fontsize=8.6, color=INK2, zorder=6)
    ax.set_xlim(0, 0.78)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_title("子类内每只标的的年化波动（大点=中位，小点=成员）", fontsize=11,
                 color=INK2, pad=10, loc="left")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8.5, colors=INK3, length=0)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(axis="x", color="#e6e5e1", lw=1, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#dcdbd6")

    # 大类分隔线
    prev = None
    for yi, s in zip(y, order):
        if prev is not None and sub_meta[s]["cls"] != prev:
            for a in axes:
                a.axhline(yi + 0.5, color="#e0dfda", lw=1, zorder=1)
        prev = sub_meta[s]["cls"]

    handles = [Patch(facecolor=CLR[c], label=c) for c in CLS_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.005), labelcolor=INK2)
    fig.suptitle(f"全球资产池结构 —— {sum(sub_meta[s]['n'] for s in order)} 只 ETF / "
                 f"{len(order)} 个功能子类",
                 fontsize=15, color=INK, x=0.055, ha="left", y=0.975, fontweight="bold")
    fig.text(0.055, 0.933,
             "颜色 = 功能大类；分隔线 = 大类边界。近 3 年年化波动，数据截至 "
             f"{S.attrs.get('asof', '')}",
             fontsize=9.6, color=INK2, ha="left")
    fig.subplots_adjust(left=0.115, right=0.975, top=0.895, bottom=0.075)
    path = os.path.join(OUT_FIG, "T_structure.png")
    fig.savefig(path, facecolor=PLANE)
    plt.close(fig)
    return path


# ================================================================ 图 2：相关热图

def fig_corr(corr, order, sub_meta):
    o = cluster_order(corr.loc[order, order])
    C = corr.loc[o, o]
    fig, ax = plt.subplots(figsize=(10.2, 8.6), dpi=150)
    fig.patch.set_facecolor(PLANE)
    ax.set_facecolor(PLANE)

    n = len(o)
    vals = C.values
    # 单色蓝序列（magnitude 用 sequential，不用分类色）。
    # ⚠ 方向不能反：SEQ_BLUE 本身是 浅→深，直接喂给 imshow 才能「值越大越深」。
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("bl", SEQ_BLUE)
    im = ax.imshow(vals, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(o, rotation=45, ha="right", fontsize=9, color=INK2)
    ax.set_yticklabels([f"{s}{CLC}{sub_meta[s]['cls']}" for s in o], fontsize=9, color=INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # 2px 表面色间隙：用白线分隔格子
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color=PLANE, lw=2)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            v = vals[i, j]
            # 阈值 0.42 ≈ SEQ_BLUE 第 5 档，再深就该换白字（深灰压中蓝读不清）
            ax.text(j, i, f"{v:.2f}".lstrip("0") if abs(v) < 1 else "1",
                    ha="center", va="center", fontsize=7.2,
                    color="#ffffff" if v > 0.42 else INK2)

    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, colors=INK3, length=0)
    cb.set_label("日收益相关（近 3 年）", fontsize=9.5, color=INK2)

    fig.suptitle("功能子类之间的相关 —— 池子看似 79 只，真正独立的主线远少于此",
                 fontsize=14, color=INK, x=0.03, ha="left", y=0.975, fontweight="bold")
    fig.text(0.03, 0.935,
             "按层次聚类排序，高相关子类被排到一起。深色块 = 换汤不换药的重复暴露",
             fontsize=9.6, color=INK2, ha="left")
    fig.subplots_adjust(left=0.16, right=0.93, top=0.90, bottom=0.135)
    path = os.path.join(OUT_FIG, "T_corr_cluster.png")
    fig.savefig(path, facecolor=PLANE)
    plt.close(fig)
    return path


# ================================================================ Markdown 清单

def write_md(T, sub_order, by_sub, uniq, corr):
    L = []
    A = L.append
    A("# 全球资产池清单 —— 79 只 ETF\n")
    A(f"> 数据截至 **{T.attrs['asof']}**。年化波动/收益、相关系数为**近 3 年**日频口径；"
      "日均成交额为近 3 年日均美元成交额。\n>\n"
      "> 分类是**功能分类** —— 按「这只 ETF 实际靠什么赚钱」重建，"
      "不是照抄站点分类。站点分类有若干处会误导策略，已在条目下用 ⚠ 标出。\n")

    A("\n## 一、总览\n")
    A("| 功能大类 | 功能子类 | 只数 | 子类波动中位 | 与全池平均相关 |")
    A("|---|---|---:|---:|---:|")
    for _, r in sub_order.iterrows():
        A(f"| {r['功能大类']} | {r['功能子类']} | {int(r['n'])} | "
          f"{r['vol']:.1%} | {r['corr']:.2f} |")

    A("\n### 独特性排名（平均相关越低 = 越能提供别人给不了的东西）\n")
    A("| 功能子类 | 与其它子类平均相关 | 只数 |")
    A("|---|---:|---:|")
    for s, r in uniq.iterrows():
        A(f"| {s} | {r['与其它子类平均相关']:.2f} | {int(r['只数'])} |")

    A("\n## 二、明细\n")
    for _, r in sub_order.iterrows():
        s, cls = r["功能子类"], r["功能大类"]
        A(f"\n### {cls} · {s}（{int(r['n'])} 只）")
        A(f"子类波动中位 {r['vol']:.1%} · 与全池平均相关 {r['corr']:.2f}\n")
        A("| 代码 | 中文名 | 年化波动 | 年化收益 | 日均成交(百万$) | 与SPY相关 | 与全池相关 | 说明 |")
        A("|---|---|---:|---:|---:|---:|---:|---|")
        for c in by_sub[s]:
            row = T.loc[c]
            def f(x, fmt, dash="n/a"):
                return f"{x:{fmt}}" if pd.notna(x) else dash
            A(f"| `{c}` | {row['中文名']} | {f(row['年化波动'], '.1%')} | "
              f"{f(row['年化收益'], '+.1%')} | {f(row['日均成交额'], '.1f')} | "
              f"{f(row['与SPY相关'], '.2f')} | {f(row['与全池相关'], '.2f')} | "
              f"{row['说明']} |")
            if isinstance(row["备注"], str) and row["备注"].strip():
                A(f"| | | | | | | | ⚠ **{row['备注']}** |")

    path = os.path.join(STEP_DIR, "T_资产池清单.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path


# ================================================================ main

def main():
    os.makedirs(OUT_CSV, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)
    p("=" * 118)
    p("全球资产配置模拟 —— 2b：资产池分类与明细清单")
    p("=" * 118)

    px, meta, long = load()
    S = per_asset_stats(px, long)
    S.attrs["asof"] = px.index.max().strftime("%Y-%m-%d")

    # 站点分类 vs 功能分类
    tax = pd.DataFrame.from_dict(TAXONOMY, orient="index",
                                 columns=["功能大类", "功能子类", "中文名", "说明"])
    missing = [c for c in px.columns if c not in tax.index]
    extra = [c for c in tax.index if c not in px.columns]
    if extra:
        p(f"  [注意] 分类表里有 {len(extra)} 只已不在池中（如停更）：{'、'.join(extra)}")
        tax = tax.loc[[c for c in tax.index if c in px.columns]]
    if missing:
        raise SystemExit(f"分类表缺少: {missing}")

    T = tax.join(S)
    T["站点大类"] = T.index.map(meta["assetClassCh"])
    T["站点子类"] = T.index.map(meta["assetClassSubCh"])
    T["备注"] = T.index.map(lambda s: SITE_TRAP.get(s, ""))

    p(f"\n  池内 {len(T)} 只（站点列 80 只，其中 HTEC 已停更剔除）")
    p(f"  数据截至 {S.attrs['asof']}；波动/相关用近 3 年日收益；流动性用近 3 年日均成交额")

    # ---------- 明细清单 ----------
    p("\n" + "=" * 118)
    p("明细清单（按功能大类 → 子类排列）")
    p("=" * 118)

    sub_order = T.groupby(["功能大类", "功能子类"], sort=False).agg(
        n=("中文名", "size"),
        vol=("年化波动", "median"),
        corr=("与全池相关", "mean")).reset_index()
    sub_order["rank"] = sub_order["功能子类"].map(
        {s: i for i, s in enumerate(SUB_ORDER)})
    sub_order = sub_order.sort_values("rank")
    order = list(sub_order["功能子类"])

    by_sub: dict[str, list[str]] = {}
    for s in order:
        by_sub[s] = list(T[T["功能子类"] == s]
                         .sort_values("年化波动").index)

    for _, r in sub_order.iterrows():
        s = r["功能子类"]
        p(f"\n  【{r['功能大类']}】{s}   {r['n']} 只   "
          f"子类波动中位 {r['vol']:.1%}   与全池平均相关 {r['corr']:.2f}")
        p(f"    {'代码':<7}{'中文名':<17}{'年化波动':>9}{'年化收益':>9}"
          f"{'日均成交(百万$)':>17}{'与SPY':>8}{'与全池':>8}")
        p("    " + "-" * 78)
        for c in by_sub[s]:
            row = T.loc[c]
            def f(x, fmt):
                return f"{x:{fmt}}" if pd.notna(x) else "n/a"
            p(f"    {c:<7}{row['中文名']:<17}"
              f"{f(row['年化波动'], '.1%'):>9}{f(row['年化收益'], '+.1%'):>9}"
              f"{f(row['日均成交额'], '.1f'):>17}"
              f"{f(row['与SPY相关'], '.2f'):>8}{f(row['与全池相关'], '.2f'):>8}")
            if isinstance(row["备注"], str) and row["备注"].strip():
                p(f"          ⚠ {row['备注']}")

    # ---------- 冗余度 ----------
    p("\n" + "=" * 118)
    p("功能子类之间的独立性（用于判断「79 只到底有几条真主线」）")
    p("=" * 118)
    tax["key"] = T["功能子类"]
    sub_ret = {}
    for s in order:
        sub_ret[s] = px[by_sub[s]].pct_change().mean(axis=1)
    SR = pd.DataFrame(sub_ret)
    t3 = SR.index.max() - pd.DateOffset(years=3)
    corr = SR.loc[SR.index >= t3].corr()

    off = corr.values[np.triu_indices(len(corr), 1)]
    p(f"  {len(corr)} 个子类两两相关（{len(off)} 对）: 中位 {np.median(off):+.2f}   "
      f"最高 {off.max():+.2f}   最低 {off.min():+.2f}")
    p(f"  相关 > 0.80 的「近乎重复」对数: {(off > 0.80).sum()} / {len(off)}")
    p(f"  相关 < 0.20 的「真正分散」对数: {(off < 0.20).sum()} / {len(off)}")

    # 每个子类的「代表性」= 与其它子类的平均相关，越低越独特
    uniq = pd.DataFrame({
        "与其它子类平均相关": corr.mean(),
        "只数": [len(by_sub[s]) for s in corr.columns],
    }, index=corr.columns).sort_values("与其它子类平均相关")
    p("\n  独特性排名（平均相关越低 = 越能提供别人给不了的东西）:")
    p(f"    {'子类':<14}{'平均相关':>10}{'只数':>7}")
    for s, r in uniq.iterrows():
        p(f"    {s:<14}{r['与其它子类平均相关']:>10.2f}{int(r['只数']):>7}")

    # ---------- 出图 ----------
    sub_meta = {s: {"cls": r["功能大类"], "n": int(r["n"])}
                for s, r in sub_order.set_index("功能子类").iterrows()}
    f1 = fig_structure(S, tax, order, sub_meta)
    f2 = fig_corr(corr, order, sub_meta)
    T.attrs["asof"] = S.attrs["asof"]
    f3 = write_md(T, sub_order, by_sub, uniq, corr)

    # ---------- 落盘 ----------
    T.drop(columns=["备注"]).to_csv(
        os.path.join(OUT_CSV, "T_taxonomy.csv"), encoding="utf-8-sig")
    T[["功能大类", "功能子类", "中文名", "说明", "站点大类", "站点子类", "备注"]].to_csv(
        os.path.join(OUT_CSV, "T_taxonomy_notes.csv"), encoding="utf-8-sig")
    uniq.to_csv(os.path.join(OUT_CSV, "T_redundancy.csv"), encoding="utf-8-sig")
    corr.to_csv(os.path.join(OUT_CSV, "T_subclass_corr.csv"), encoding="utf-8-sig")

    p("\n" + "=" * 118)
    p(f"  图 1  {f1}")
    p(f"  图 2  {f2}")
    p(f"  清单  {f3}")
    p(f"  明细  {os.path.join(OUT_CSV, 'T_taxonomy.csv')}")
    p(f"  备注  {os.path.join(OUT_CSV, 'T_taxonomy_notes.csv')}")
    p("=" * 118)


if __name__ == "__main__":
    main()
