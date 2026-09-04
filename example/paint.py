import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('300_1h_50_money.csv.best')
# 常用参数：kind (图表类型), x (横轴), y (纵轴), title (标题)
df.plot(kind='line', x='date', y='money', title='示例图表')
plt.show()
