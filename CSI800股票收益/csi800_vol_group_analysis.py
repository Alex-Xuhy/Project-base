"""
中证800 2024年二次变差波动率分组分析
按QV波动率中位数划分为小波动股/大波动股两组，
分别计算等权组合的年化收益和年化波动

数据源: akshare (成分股) + baostock (个股日线)
"""
import akshare as ak
import baostock as bs
import pandas as pd
import numpy as np

YEAR = 2024
START = f'{YEAR}-01-01'
END   = f'{YEAR}-12-31'
INDEX_CODE = '000906'  # 中证800

# ========== Step 1: 登录 baostock + 获取成分股列表 ==========
print(f'[1/3] 获取中证800成分股列表...')
# akshare 获取成分股
df_cons = ak.index_stock_cons_csindex(symbol=INDEX_CODE)
stock_codes = df_cons.iloc[:, 4].unique().tolist()  # col 4 = 成分券代码
print(f'  成分股数量: {len(stock_codes)}')

# 确定 baostock 格式的交易所前缀
def to_bs_code(code):
    """转换为 baostock 格式: sh.600000 或 sz.000001"""
    if code.startswith(('6', '9')):
        return f'sh.{code}'
    else:
        return f'sz.{code}'

# ========== Step 2: baostock 登录，批量获取日线 ==========
print(f'\n[2/3] baostock 获取个股{YEAR}年日线数据...')
lg = bs.login()
print(f'  baostock login: {lg.error_msg}')

stock_returns = {}  # code -> Series of daily returns
stock_closes = {}   # code -> last close (for reference)
failed = 0
n_total = len(stock_codes)

for i, code in enumerate(stock_codes):
    try:
        bs_code = to_bs_code(code)
        rs = bs.query_history_k_data_plus(
            bs_code, 'date,close,pctChg',
            start_date=START.replace('-', '-'),
            end_date=END.replace('-', '-'),
            frequency='d', adjustflag='2'  # 前复权
        )
        if rs.error_code != '0':
            failed += 1
            continue

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        if len(data_list) < 30:
            failed += 1
            continue

        df = pd.DataFrame(data_list, columns=rs.fields)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        df['ret'] = pd.to_numeric(df['pctChg'], errors='coerce') / 100.0
        df = df.dropna(subset=['ret'])

        if len(df) >= 30:
            stock_returns[code] = df['ret']
        else:
            failed += 1
    except Exception:
        failed += 1

    if (i + 1) % 80 == 0 or i == n_total - 1:
        pct = (i + 1) * 100 // n_total
        print(f'  {i+1}/{n_total} ({pct}%) - 成功 {len(stock_returns)}, 失败 {failed}', flush=True)

bs.logout()
print(f'  成功获取 {len(stock_returns)} 只股票数据, 失败/数据不足 {failed} 只\n')

# ========== Step 3: 计算QV波动率、分组、组合指标 ==========
print('[3/3] 计算QV波动率并分组...')

qv_vols = {}
ann_returns = {}
for code, ret in stock_returns.items():
    qv_vols[code] = np.sqrt(np.sum(ret ** 2))
    ann_returns[code] = (1 + ret).prod() - 1

qv_series = pd.Series(qv_vols)
median_vol = qv_series.median()

low_vol_stocks = qv_series[qv_series <= median_vol].index.tolist()
high_vol_stocks = qv_series[qv_series > median_vol].index.tolist()

print(f'  QV波动率中位数: {median_vol*100:.2f}%')
print(f'  小波动组(<=中位数): {len(low_vol_stocks)} 只')
print(f'  大波动组(>中位数):   {len(high_vol_stocks)} 只')

# ---- 两组分布摘要 ----
print(f'\n  小波动组 QV vol 范围: {qv_series[low_vol_stocks].min()*100:.1f}% ~ {qv_series[low_vol_stocks].max()*100:.1f}%')
print(f'  大波动组 QV vol 范围: {qv_series[high_vol_stocks].min()*100:.1f}% ~ {qv_series[high_vol_stocks].max()*100:.1f}%')


def group_metrics(stock_list, label):
    """等权组合指标"""
    all_rets = []
    for code in stock_list:
        all_rets.append(stock_returns[code])

    ret_matrix = pd.concat(all_rets, axis=1)
    daily_port_ret = ret_matrix.mean(axis=1).dropna()

    cum_ret = (1 + daily_port_ret).prod() - 1
    qv_vol = np.sqrt(np.sum(daily_port_ret ** 2))
    n = len(daily_port_ret)
    ann_ret_from_cum = ((1 + cum_ret) ** (252 / n) - 1) if n > 0 else 0

    # 个股统计
    ind_rets = [ann_returns[c] for c in stock_list]
    ind_vols = [qv_vols[c] for c in stock_list]

    print(f'\n  {"="*55}')
    print(f'  {label}')
    print(f'  {"="*55}')
    print(f'  股票数量:         {len(stock_list):>6d}')
    print(f'  交易日数:         {n:>6d}')
    print(f'  年化收益 (累计):  {cum_ret*100:>7.2f}%  (按252d折算: {ann_ret_from_cum*100:.2f}%)')
    print(f'  年化波动 (QV):    {qv_vol*100:>7.2f}%')
    print(f'  近似夏普:         {(cum_ret-0.028)/qv_vol:>7.3f}')
    print(f'  日收益均值:       {daily_port_ret.mean()*100:>7.4f}%')
    print(f'  日收益标准差:     {daily_port_ret.std()*100:>7.3f}%')
    print(f'  个股收益均值:     {np.mean(ind_rets)*100:>7.1f}%')
    print(f'  个股波动均值:     {np.mean(ind_vols)*100:>7.1f}%')

    return {
        'label': label, 'n_stocks': len(stock_list), 'n_days': n,
        'cum_ret': cum_ret, 'qv_vol': qv_vol, 'daily_ret': daily_port_ret,
    }


low_r = group_metrics(low_vol_stocks, '小波动组 (波动率 <= 中位数)')
high_r = group_metrics(high_vol_stocks, '大波动组 (波动率 > 中位数)')

# ========== 汇总对比 ==========
print(f'\n{"="*65}')
print(f'中证800 {YEAR}年 — 波动率分组对比 (QV等权)')
print(f'{"="*65}')
print(f'{"指标":<22s} {"小波动组":>18s} {"大波动组":>18s} {"差值":>18s}')
print(f'{"-"*76}')
print(f'{"股票数量":<22s} {low_r["n_stocks"]:>18d} {high_r["n_stocks"]:>18d} {low_r["n_stocks"]-high_r["n_stocks"]:>18d}')
print(f'{"年化收益(累计)":<22s} {low_r["cum_ret"]*100:>17.2f}% {high_r["cum_ret"]*100:>17.2f}% {(low_r["cum_ret"]-high_r["cum_ret"])*100:>17.2f}%')
print(f'{"年化波动(QV)":<22s} {low_r["qv_vol"]*100:>17.2f}% {high_r["qv_vol"]*100:>17.2f}% {(low_r["qv_vol"]-high_r["qv_vol"])*100:>17.2f}%')
sr_low = (low_r["cum_ret"]-0.028)/low_r["qv_vol"]
sr_high = (high_r["cum_ret"]-0.028)/high_r["qv_vol"]
print(f'{"近似夏普":<22s} {sr_low:>18.3f} {sr_high:>18.3f} {sr_low-sr_high:>18.3f}')
print(f'{"日收益均值":<22s} {low_r["daily_ret"].mean()*100:>16.4f}% {high_r["daily_ret"].mean()*100:>16.4f}% {(low_r["daily_ret"].mean()-high_r["daily_ret"].mean())*100:>16.4f}%')
print(f'{"日收益标准差":<22s} {low_r["daily_ret"].std()*100:>16.3f}% {high_r["daily_ret"].std()*100:>16.3f}% {(low_r["daily_ret"].std()-high_r["daily_ret"].std())*100:>16.3f}%')
