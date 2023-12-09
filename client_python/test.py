import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV
import joblib
import time

# 假设df是包含所有指标的DataFrame
stockID = 8
df = pd.read_csv(f'../stock{stockID}.csv')  # 如果数据来自CSV文件
df.fillna(method='bfill', inplace=True)
df.replace([np.inf, -np.inf], 0, inplace=True)

# 选择输入特征
input_features = [
    'BidPrice1', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5',
    'AskPrice1', 'AskPrice2', 'AskPrice3', 'AskPrice4', 'AskPrice5',
    'BidVolume1', 'BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5',
    'AskVolume1', 'AskVolume2', 'AskVolume3', 'AskVolume4', 'AskVolume5',
    'Spread', 'tempVolume', 'tempValue', 'VOI',
    'QR1', 'QR2', 'QR3', 'QR4', 'QR5',
    'HR1', 'HR2', 'HR3', 'HR4', 'HR5', 'Press'
]

# 提取目标变量
target = 'Return'

# 划分数据集
X = df[input_features]
y = df[target]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据规范化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

# 创建SVR模型实例
svr = SVR(kernel='poly',degree = 3,C= 0.4,epsilon = 0.006,gamma=0.001)

# 训练模型
svr.fit(X_train_scaled, y_train)

# 预测
y_pred = svr.predict(X_test_scaled)
y_pred_t = svr.predict(X_train_scaled)
y_for_test = svr.predict(X_scaled)

joblib.dump(svr,f'svr_model_{stockID}.joblib')
joblib.dump(scaler,f'scaler_model_{stockID}.joblib')

# 计算均方根误差（RMSE）
rmse_train = sqrt(mean_squared_error(y_train, y_pred_t))
rmse_test = sqrt(mean_squared_error(y_test, y_pred))
print('Train RMSE:', rmse_train)
print('Test RMSE:', rmse_test)
print(pd.DataFrame(y_pred).describe())

# 交易模拟
wealth = 1000000
stock_num = 0
stock_value = 0
total_wealth_list = []
stock_num_list = []
buy_interval = 0
sell_interval = 0

time_t = time.ctime()
print(time_t)
for i in range(144000):
    if i % 1440 == 0:
        print(f'finished {i/1440}%')
    if i % 14400 > 14340:
        if stock_num > 0:
            wealth += stock_num * price
            stock_num = 0
            continue
    price = df.iloc[i].LastPrice  # 获取测试集中每个时间点的价格
    prediction = y_for_test[i]
    if prediction > 0 and wealth > 2000 * price and buy_interval >= 100:  # 如果预测为正，考虑买入
        stock_num += 2000
        wealth -= 2000 * price * 1.0002
        buy_interval = 0
    elif prediction < 0 and stock_num >= 2000 and sell_interval >= 100:  # 如果预测为负，考虑卖出
        wealth += 2000 * price
        stock_num -= 2000
        sell_interval = 0
    
    buy_interval += 1
    sell_interval += 1
    stock_value = stock_num * price
    total_wealth_list.append(stock_value + wealth)
    stock_num_list.append(stock_num)
time_tmp = time.ctime()
print(time_tmp)
# 输出最终财富值
print('Final wealth:', total_wealth_list[-1])