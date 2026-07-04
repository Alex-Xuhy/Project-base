# -*- coding: utf-8 -*-
# 用二次变差衡量2000-2025年上证综指的波动

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# 导入原数据（来源：CSMAR库-指数收益率）
origin = pd.read_csv('TRD_Index.csv')

dates = pd.date_range('2000-01', '2025-12', freq='MS').strftime('%Y-%m').tolist() 
df = pd.DataFrame({'Year-Month': dates})   # 构建2000-01到2025-12的空数据框

# 数据处理（计算月度波动率，这里没有年化处理，以保证数据直观性）
origin['Year-Month'] = pd.to_datetime(origin['Trddt']).dt.strftime('%Y-%m')   # 将交易日期转为 datetime，提取年-月
monthly_qv = origin.groupby('Year-Month')['Retindex'].apply(lambda x: (x ** 2).sum())   # 计算月度二次变差（使用groupby聚合和匿名函数）

df['Volatility'] = df['Year-Month'].map(monthly_qv)   # 填入月度波动数据（由于使用groupby命令创建列表，所以可以直接使用map命令映射）

# 数据检查（是否有错误或可疑结果）
print(df.head(24))
print(df['Volatility'].describe())   # 结果显示不存在异常值
print('共有' + str(len(df)) + '个数据')

# 数据可视化（折线图）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决中文字体显示问题
plt.figure(figsize=(12, 6))
plt.plot(df['Year-Month'], df['Volatility']*100, marker='o', linewidth=1, markersize=2)
plt.title('月度股指已实现波动率（二次变差）', fontsize=16, fontweight='bold')
plt.xlabel('月份', fontsize=12)
plt.ylabel('波动率（%）', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(range(0, len(df['Year-Month']), 12), df['Year-Month'][::12], rotation=45)   # 时间跨度太长，对横坐标做微处理
plt.axhline(y=0.4279, color='grey', linestyle=':', alpha=0.7, label='月度波动均值')   # 创建一条参考线——26年来上证综指月度波动的均值
plt.tight_layout()
plt.legend()  # 添加图例
plt.show()

df.to_csv('Quadratic Variation.csv')