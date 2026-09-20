import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('150_4h_50_5_15_5_0.0005_future_money.csv.run')
# df['money_diff']=df['money'].diff()
# df['clear_money_diff']=df['clear_money'].diff()
df.plot(kind='line', x='date', y=['money','clear_money'], title='money')
# df1=pd.read_csv('150_4h_5_50_30_5_0.001_money.csv')
# # 常用参数：kind (图表类型), x (横轴), y (纵轴), title (标题)
# df.plot(kind='line', x='date', y=['clear_money_diff'], title='diff')
# df1['clear_money_diff']=df1['clear_money'].diff()
# df1.plot(kind='line', x='date', y=['clear_money_diff'], title='df1diff')
# df1['diff']=df1['clear_money_diff']-df['clear_money_diff']
# df1.plot(kind='line', x='date', y=['diff'], title='diffdiff')

plt.show()
