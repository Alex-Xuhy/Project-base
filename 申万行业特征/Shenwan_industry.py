# -*- coding: utf-8 -*-
"""
申万一级行业（31个）收益与风险特征分析
时间窗口: 2021-2025, 月频
分析维度: 收益、波动率、最大回撤、夏普比率，按板块输出图表
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import warnings, os

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 0. 全局配置
# ============================================================
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR  = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

START_DATE = "2021-01-01"
END_DATE   = "2025-12-31"

# 申万31个一级行业 —— 名称与代码（2021新版分类）
# 代码参考申万研究所官方编码
SW_INDUSTRIES = {
    "农林牧渔": "801010", "基础化工": "801030", "钢铁":     "801040",
    "有色金属": "801050", "电子":     "801080", "汽车":     "801880",
    "家用电器": "801110", "食品饮料": "801120", "纺织服饰": "801130",
    "轻工制造": "801140", "医药生物": "801150", "公用事业": "801160",
    "交通运输": "801170", "房地产":   "801180", "商贸零售": "801200",
    "社会服务": "801210", "综合":     "801230", "建筑材料": "801710",
    "建筑装饰": "801720", "电力设备": "801730", "国防军工": "801740",
    "计算机":   "801750", "传媒":     "801760", "通信":     "801770",
    "银行":     "801780", "非银金融": "801790", "机械设备": "801890",
    "煤炭":     "801950", "石油石化": "801960", "环保":     "801970",
    "美容护理": "801980",
}

# 板块映射（金融/周期/消费/成长/稳定/综合）
SECTOR_MAP = {
    "农林牧渔": "消费", "基础化工": "周期", "钢铁":     "周期",
    "有色金属": "周期", "电子":     "成长", "汽车":     "消费",
    "家用电器": "消费", "食品饮料": "消费", "纺织服饰": "消费",
    "轻工制造": "消费", "医药生物": "消费", "公用事业": "稳定",
    "交通运输": "稳定", "房地产":   "金融", "商贸零售": "消费",
    "社会服务": "消费", "综合":     "综合", "建筑材料": "周期",
    "建筑装饰": "周期", "电力设备": "成长", "国防军工": "成长",
    "计算机":   "成长", "传媒":     "成长", "通信":     "成长",
    "银行":     "金融", "非银金融": "金融", "机械设备": "周期",
    "煤炭":     "周期", "石油石化": "周期", "环保":     "稳定",
    "美容护理": "消费",
}

SECTOR_COLORS = {
    "金融": "#C0392B", "周期": "#E67E22", "消费": "#27AE60",
    "成长": "#2980B9", "稳定": "#8E44AD", "综合": "#7F8C8D",
}

SECTOR_ORDER = ["金融", "周期", "消费", "成长", "稳定", "综合"]

# 关键市场事件（月频标注）
MARKET_EVENTS = [
    ("2021-02", "春节后抱团瓦解"),
    ("2021-07", "教育双减"),
    ("2021-09", "恒大危机"),
    ("2022-03", "上海疫情封控"),
    ("2022-04", "沪指触底2863"),
    ("2022-10", "二十大召开"),
    ("2022-11", "疫情放开二十条"),
    ("2023-03", "硅谷银行危机"),
    ("2023-08", "活跃资本市场"),
    ("2024-01", "市场深度调整"),
    ("2024-02", "维稳反弹"),
    ("2024-09", "924政策组合拳"),
    ("2025-04", "中美关税战升级"),
]
EVENTS_DF = pd.DataFrame(MARKET_EVENTS, columns=["date", "label"])
EVENTS_DF["date"] = pd.to_datetime(EVENTS_DF["date"])

# ============================================================
# 1. 数据获取
# ============================================================
def fetch_sw_index(code):
    """
    通过 AKShare 拉取申万行业指数月线。
    AKShare index_hist_sw 返回列 (位置索引):
      0:代码  1:日期  2:开盘  3:最高  4:最低  5:收盘  6:成交量  7:成交额
    """
    try:
        raw = ak.index_hist_sw(symbol=code, period="month")
    except Exception:
        return None
    if raw is None or raw.empty:
        return None
    # 使用位置索引避免中文列名编码问题
    df = pd.DataFrame({
        "date":  pd.to_datetime(raw.iloc[:, 1]),
        "close": pd.to_numeric(raw.iloc[:, 5], errors="coerce"),
    })
    df = df.dropna(subset=["close"])
    df.set_index("date", inplace=True)
    df = df.loc[START_DATE:END_DATE].copy()
    return df if len(df) >= 6 else None


def fetch_all():
    """批量拉取31个行业月频收盘价，支持 parquet 缓存"""
    cache_path = os.path.join(OUTPUT_DIR, "sw_monthly_cache.parquet")
    if os.path.exists(cache_path):
        print("[CACHE] Reading cached parquet ...")
        all_df = pd.read_parquet(cache_path)
        all_df.columns.name = "industry"
        return all_df

    print("[FETCH] Pulling Shenwan industry monthly data via AKShare ...")
    closes = {}
    for i, (name, code) in enumerate(SW_INDUSTRIES.items()):
        print(f"  [{i+1:02d}/31] {name} ({code}) ... ", end="", flush=True)
        df = fetch_sw_index(code)
        if df is not None:
            closes[name] = df["close"]
            print(f"OK ({len(df)} obs)")
        else:
            print("FAILED")
    if not closes:
        raise RuntimeError("All industry data fetch failed.")
    all_df = pd.DataFrame(closes).sort_index()
    all_df.index = pd.to_datetime(all_df.index)
    all_df.to_parquet(cache_path)
    print(f"[CACHE] Saved to {cache_path}")
    return all_df


# ============================================================
# 2. 统计指标计算
# ============================================================
def compute_stats(returns: pd.DataFrame, rf_annual=0.025):
    """计算各行业的收益与风险统计"""
    stats_list = []
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) < 12:
            continue
        total_ret   = (1 + r).prod() - 1
        ann_ret     = (1 + r).prod() ** (12 / len(r)) - 1
        ann_vol     = r.std() * np.sqrt(12)
        sharpe      = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else np.nan
        # 最大回撤
        cum = (1 + r).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        max_dd = dd.min()
        # 最大回撤结束日期
        dd_end = dd.idxmin()
        # 胜率
        win_rate = (r > 0).sum() / len(r)
        # 正负均值
        pos_mean = r[r > 0].mean() if (r > 0).sum() > 0 else 0
        neg_mean = r[r < 0].mean() if (r < 0).sum() > 0 else 0
        # 累计收益序列（最后值）
        stats_list.append({
            "行业": col,
            "板块": SECTOR_MAP.get(col, "综合"),
            "累计收益(%)":     round(total_ret * 100, 2),
            "年化收益(%)":     round(ann_ret * 100, 2),
            "年化波动(%)":     round(ann_vol * 100, 2),
            "夏普比率":        round(sharpe, 2),
            "最大回撤(%)":     round(max_dd * 100, 2),
            "最大回撤时点":    dd_end.strftime("%Y-%m") if hasattr(dd_end, 'strftime') else str(dd_end),
            "月胜率(%)":       round(win_rate * 100, 1),
            "月均正收益(%)":   round(pos_mean * 100, 2),
            "月均负收益(%)":   round(neg_mean * 100, 2),
        })
    return pd.DataFrame(stats_list)


def sector_summary_table(stats_df):
    """输出板块层面的汇总统计"""
    rows = []
    for sector in SECTOR_ORDER:
        sub = stats_df[stats_df["板块"] == sector]
        if sub.empty:
            continue
        rows.append({
            "板块": sector,
            "行业数": len(sub),
            "平均累计收益(%)":    round(sub["累计收益(%)"].mean(), 2),
            "平均年化收益(%)":    round(sub["年化收益(%)"].mean(), 2),
            "平均年化波动(%)":    round(sub["年化波动(%)"].mean(), 2),
            "平均夏普比率":       round(sub["夏普比率"].mean(), 2),
            "平均最大回撤(%)":    round(sub["最大回撤(%)"].mean(), 2),
            "最佳行业":           sub.loc[sub["累计收益(%)"].idxmax(), "行业"],
            "最佳累计收益(%)":    sub["累计收益(%)"].max(),
            "最差行业":           sub.loc[sub["累计收益(%)"].idxmin(), "行业"],
            "最差累计收益(%)":    sub["累计收益(%)"].min(),
        })
    return pd.DataFrame(rows)


# ============================================================
# 3. 绘图
# ============================================================
def _add_events(ax, date_index, y_lim, text_offset=0.92, show_label=True):
    """在 ax 上标注市场事件竖线"""
    for _, ev in EVENTS_DF.iterrows():
        nearest_idx = np.argmin(np.abs(date_index - ev["date"]))
        nearest = date_index[nearest_idx]
        if abs((nearest - ev["date"]).days) <= 45:
            ax.axvline(x=nearest, color="gray", linestyle="--",
                       alpha=0.45, linewidth=0.8)
            if show_label:
                ax.text(nearest, y_lim[1] * text_offset, ev["label"],
                        rotation=90, fontsize=6.5, color="gray",
                        verticalalignment="top")


def plot_sector_returns(returns, sector, members, stats_df):
    """绘制单个板块内所有行业的累计收益走势子图"""
    n = len(members)
    if n == 0:
        return
    cols = min(3, n)
    rows = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    sector_stats = stats_df[stats_df["板块"] == sector].set_index("行业")

    for i, ind in enumerate(members):
        ax = axes[i]
        cum = (1 + returns[ind].dropna()).cumprod() - 1
        ax.fill_between(cum.index, 0, cum.values,
                        where=(cum.values >= 0), color="#E74C3C", alpha=0.12)
        ax.fill_between(cum.index, 0, cum.values,
                        where=(cum.values < 0), color="#27AE60", alpha=0.12)
        ax.plot(cum.index, cum.values, color=SECTOR_COLORS.get(sector, "#333"),
                linewidth=1.5)

        y_lim = ax.get_ylim()
        _add_events(ax, cum.index, y_lim, show_label=(i < cols))

        # 标题附关键统计
        if ind in sector_stats.index:
            s = sector_stats.loc[ind]
            title = (f"{ind}  |  累收{s['累计收益(%)']:.1f}%  |  "
                     f"波动{s['年化波动(%)']:.0f}%  |  "
                     f"回撤{s['最大回撤(%)']:.0f}%  |  "
                     f"Sharpe {s['夏普比率']:.2f}")
        else:
            title = ind
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.axhline(y=0, color="black", linewidth=0.6)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"[{sector}] 申万一级行业累计收益走势 (2021-2025)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(CHART_DIR, f"sector_{sector}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def plot_sector_summary(returns, stats_df):
    """板块汇总：每个板块等权累计收益曲线 + 事件标注"""
    fig, ax = plt.subplots(figsize=(14, 7))
    for sector in SECTOR_ORDER:
        members = [c for c in returns.columns if SECTOR_MAP.get(c) == sector]
        if not members:
            continue
        eq_ret = returns[members].mean(axis=1)
        cum = (1 + eq_ret).cumprod() - 1
        ax.plot(cum.index, cum.values, color=SECTOR_COLORS[sector],
                linewidth=2.2, label=f"{sector} ({len(members)}个)")

    yl = ax.get_ylim()
    _add_events(ax, returns.index, yl, text_offset=0.95, show_label=True)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    ax.set_title("申万一级行业板块等权累计收益对比 (2021-2025)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "sector_summary.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def plot_risk_return_scatter(stats_df):
    """风险-收益散点图 (年化波动 vs 年化收益)"""
    fig, ax = plt.subplots(figsize=(13, 8))
    for sector in SECTOR_ORDER:
        color = SECTOR_COLORS.get(sector, "#7F8C8D")
        sub = stats_df[stats_df["板块"] == sector]
        if sub.empty:
            continue
        ax.scatter(sub["年化波动(%)"], sub["年化收益(%)"],
                   c=color, label=sector, s=90, edgecolors="white",
                   linewidth=0.8, zorder=5)
        for _, row in sub.iterrows():
            ax.annotate(row["行业"],
                        (row["年化波动(%)"], row["年化收益(%)"]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7.5, alpha=0.85)
    ax.set_xlabel("Annualized Volatility (%)", fontsize=12)
    ax.set_ylabel("Annualized Return (%)", fontsize=12)
    ax.set_title("Shenwan Level-1 Industries: Risk-Return (2021-2025)",
                 fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, frameon=True)
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "risk_return_scatter.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def plot_heatmap(returns, stats_df):
    """月度收益热力图"""
    order = stats_df.sort_values(
        ["板块", "累计收益(%)"], ascending=[True, False]
    )["行业"].tolist()
    piv = returns[order].T * 100
    vmax = min(max(piv.max().max(), 20), 35)
    vmin = max(min(piv.min().min(), -20), -35)

    fig, ax = plt.subplots(figsize=(22, 14))
    cmap = LinearSegmentedColormap.from_list(
        "rd_wh_gn", ["#C0392B", "#F5F5F5", "#27AE60"]
    )
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    # 板块分隔线
    prev_sector = None
    for i, ind in enumerate(order):
        s = SECTOR_MAP.get(ind, "")
        if prev_sector and s != prev_sector:
            ax.axhline(y=i - 0.5, color="black", linewidth=1.2)
        prev_sector = s

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    date_labels = [d.strftime("%Y-%m") for d in piv.columns]
    step = max(1, len(date_labels) // 24)
    ax.set_xticks(range(0, len(date_labels), step))
    ax.set_xticklabels([date_labels[i] for i in range(0, len(date_labels), step)],
                       rotation=45, fontsize=8)
    ax.set_title("Shenwan Level-1 Industries: Monthly Return Heatmap (2021-2025)",
                 fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.015)
    cbar.set_label("Monthly Return (%)", fontsize=10)
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "heatmap.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def plot_sector_boxplot(returns):
    """各板块月度收益分布箱线图"""
    df_list = []
    for col in returns.columns:
        sector = SECTOR_MAP.get(col, "综合")
        df_list.append(pd.DataFrame({
            "sector": sector,
            "industry": col,
            "return": returns[col].values * 100,
        }))
    plot_df = pd.concat(df_list, ignore_index=True)
    plot_df = plot_df[plot_df["sector"] != "综合"]

    fig, ax = plt.subplots(figsize=(14, 7))
    sectors_order = ["金融", "周期", "消费", "成长", "稳定"]
    data_groups = [plot_df[plot_df["sector"] == s]["return"].dropna().values
                   for s in sectors_order]
    bp = ax.boxplot(data_groups, patch_artist=True,
                    showfliers=True, widths=0.5)
    ax.set_xticklabels(sectors_order, fontsize=11)
    for patch, sector in zip(bp["boxes"], sectors_order):
        patch.set_facecolor(SECTOR_COLORS[sector])
        patch.set_alpha(0.35)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Monthly Return (%)", fontsize=12)
    ax.set_title("Shenwan Sector Monthly Return Distribution (2021-2025)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "sector_boxplot.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


# ============================================================
# 4. 主流程
# ============================================================
def main():
    print("=" * 60)
    print("  Shenwan Level-1 Industry (31) Return & Risk Analysis")
    print("  Window: 2021-2025, Monthly Frequency")
    print("=" * 60)

    # --- 数据 ---
    closes = fetch_all()
    valid_cols = [c for c in SW_INDUSTRIES if c in closes.columns]
    closes = closes[valid_cols]
    print(f"\n[INFO] {len(valid_cols)} industries, {len(closes)} monthly observations")

    # --- 月度收益 ---
    returns = closes.pct_change().dropna(how="all")
    print(f"[INFO] Return series: {returns.shape[0]} periods x {returns.shape[1]} industries")

    # --- 统计表 ---
    stats_df = compute_stats(returns)
    stats_path = os.path.join(OUTPUT_DIR, "sw_industry_stats.csv")
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"\n[STATS] Saved: {stats_path}")
    print(stats_df.to_string(index=False))

    # --- 板块汇总 ---
    sector_sum = sector_summary_table(stats_df)
    sum_path = os.path.join(OUTPUT_DIR, "sw_sector_summary.csv")
    sector_sum.to_csv(sum_path, index=False, encoding="utf-8-sig")
    print(f"\n[SECTOR] Sector summary:")
    print(sector_sum.to_string(index=False))

    # --- 绘图 ---
    print("\n[PLOT] Generating charts ...")
    for sector in SECTOR_ORDER:
        members = stats_df[stats_df["板块"] == sector]["行业"].tolist()
        if members:
            plot_sector_returns(returns, sector, members, stats_df)

    plot_sector_summary(returns, stats_df)
    plot_risk_return_scatter(stats_df)
    plot_heatmap(returns, stats_df)
    plot_sector_boxplot(returns)

    print(f"\n[DONE] All charts saved to: {CHART_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
