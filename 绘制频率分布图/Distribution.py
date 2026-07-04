# -*- coding: utf-8 -*-
# 对股票指数收益率和股指期权价格绘制频率分布

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']  # 优先使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# ---------------------------------------------------------------------------- #
#                                    指数收益率分布                                   #
# ---------------------------------------------------------------------------- #

# 数据导入和处理
index_yield = pd.read_csv('TRD_Index.csv')
index_yield['Daily_yield'] = index_yield['Retindex'] * 100

plt.figure(figsize=(10, 12))

# ---------------------------------- 频率分布直方图 --------------------------------- #

plt.subplot(2, 2, 1)
plt.hist(index_yield['Daily_yield'], bins=100, alpha=0.7)
plt.title('直方图——股指收益率')
plt.xlabel('日收益率（%）')
plt.ylabel('频率分布')

# ---------------------------------- KDE 核密度 --------------------------------- #

plt.subplot(2, 2, 2)
sns.histplot(index_yield['Daily_yield'], bins=100, kde=True)
plt.title('KDE核密度估计——股指收益率')
plt.xlabel('日收益率（%）')
plt.ylabel('频率分布')

print(index_yield['Daily_yield'].skew())

# ---------------------------------------------------------------------------- #
#                                   股指期权价格分布                                   #
# ---------------------------------------------------------------------------- #

price_option = pd.read_excel('Option_data.xlsx')
price_option['price'] = (price_option['best_bid'] + price_option['best_offer']) / 2

# ---------------------------------- 频率分布直方图 --------------------------------- #

plt.subplot(2, 2, 3)
plt.hist(price_option['price'], bins=100, alpha=0.7)
plt.title('直方图——股指期权价格')
plt.xlabel('期权价格')
plt.ylabel('频率分布')

# ---------------------------------- KDE 核密度 --------------------------------- #

plt.subplot(2, 2, 4)
sns.histplot(price_option['price'], bins=100, kde=True)
plt.title('KDE核密度估计——股指期权价格')
plt.xlabel('期权价格')
plt.ylabel('频率分布')

print(price_option['price'].skew())

plt.show()