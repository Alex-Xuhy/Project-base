# -*- coding: utf-8 -*-
# 计算《魔兽世界》时光服一个角色在一个版本内做到装备毕业的概率

import pandas as pd
import math
import matplotlib.pyplot as plt

# 创建数据框，输入基础数据：各部位装备掉落概率
slots_drop = pd.DataFrame(list({
'head':5/9, # 套装掉落概率
'shoulders':math.comb(33,3)/math.comb(34,4), # 萨塔里奥的套装外掉落为 34 件
'chest':math.comb(33,3)/math.comb(34,4),
'wrists':math.comb(26,4)/math.comb(27,5),   # 格罗布鲁斯的掉落为 27 件
'hands':5/9,   # 套装掉落概率
'waist':math.comb(26,4)/math.comb(27,5),   # 格罗布鲁斯的掉落为 27 件
'legs':math.comb(27,4)/math.comb(28,5),   # 迈可斯纳的掉落为 28 件
'feet':math.comb(26,4)/math.comb(27,5),   # 药剂师诺斯的掉落为 27 件
'mainhand':math.comb(10,2)/math.comb(11,3),   # 天启四骑士的套装外掉落为 11 件
'ranged':math.comb(18,3)/math.comb(19,4),   # 克尔苏加德的套装外掉落为 19 件
'cloak':math.comb(27,4)/math.comb(28,5),    # 教官拉苏维奥斯的掉落为 28 件
'ring':math.comb(33,3)/math.comb(34,4),   # 萨塔里奥的套装外掉落为 34 件
'trinket':math.comb(7,1)/math.comb(8,2)   # 盲眼者的套装外掉落为 8 件
}.items()), columns = ['Slot','DropProbability'])

slots_drop['NotDropProbability'] = 1 - slots_drop['DropProbability']

print(slots_drop)

# 绘制毕业概率折线图
bisprob_list = []

for i in range(1,9):
    BIS_slot = 1 - slots_drop['NotDropProbability']**i
    BIS_whole = BIS_slot.prod()
    bisprob_list.append(BIS_whole) # 计算从第一个cd开始,能获得全身BIS的概率(随着时间推移逐渐升高)
    
BIS_Prob = pd.DataFrame({
    'No.CD':range(1,9),
    'BisProb':bisprob_list
}) # CD-BIS概率表格

print(BIS_Prob)

fig, ax = plt.subplots(figsize=(12, 10))

ax.plot(BIS_Prob['No.CD'],BIS_Prob['BisProb'],'b-', linewidth=2, alpha=0.7)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% line')
final_prob_value = BIS_Prob['BisProb'].iloc[-1]
final_prob_percent = final_prob_value * 100
ax.axhline(y=final_prob_value, color='orange', linestyle=':', alpha=0.7, 
           label=f'Final Probability: {final_prob_percent:.1f}%')
ax.set_xlabel('CD', fontsize=12)
ax.set_ylabel('Probability of BIS', fontsize=12)
ax.set_title('Probability of Getting Every BIS at Every CD', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()  # 确保图例显示
ax.set_ylim([0, 0.55])

plt.show()