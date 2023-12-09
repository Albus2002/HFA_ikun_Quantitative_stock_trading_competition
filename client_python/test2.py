import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout,Reshape
from tensorflow.keras.optimizers import legacy

# 假设df是包含所有指标的DataFrame
df = pd.read_csv('../stock1.csv')  # 从CSV文件加载数据
df.fillna(method='bfill', inplace=True)

# 提取所需的特征列
feature_columns = [
    'BidPrice1', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5',
    'AskPrice1', 'AskPrice2', 'AskPrice3', 'AskPrice4', 'AskPrice5',
    'BidVolume1', 'BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5',
    'AskVolume1', 'AskVolume2', 'AskVolume3', 'AskVolume4', 'AskVolume5',
    'Spread', 'tempVolume', 'tempValue', 'VOI', 'QR1', 'QR2', 'QR3', 'QR4', 'QR5',
    'HR1', 'HR2', 'HR3', 'HR4', 'HR5', 'Press'
]

# 提取目标列（Return）
target_column = 'Return'

# 构建输入数据集
X = df[feature_columns].values
y = df[target_column].values

# 归一化特征
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)

# 定义时间步长
time_step = 60
# 重塑数据以适应模型的输入格式 (samples, time_steps, features)
X_scaled = np.array([X_scaled[i - time_step:i, :] for i in range(time_step, len(X_scaled))])
y = y[time_step:]

# 划分训练和测试数据集
train_size = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 构建CNN-LSTM模型
model = Sequential()
# 添加卷积层
model.add(Conv1D(filters=35, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(time_step, 35)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
# 添加LSTM层
model.add(Reshape((time_step // 2, -1)))  # 适当调整维度以匹配LSTM的输入要求
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
# 添加全连接层和输出层
model.add(Dense(25))
model.add(Dense(1))

# 编译模型
model.compile(optimizer=legacy.Adam(learning_rate=0.00001), loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 预测未来的回报率
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 输出预测结果
# ...（省略输出和绘图代码）...

# ...（继续上面的代码）...

# 反归一化预测结果（如果之前进行了归一化）
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y.fit_transform(y.reshape(-1, 1))  # 注意：使用相同的scaler对象和参数
train_predict = scaler_y.inverse_transform(train_predict)
test_predict = scaler_y.inverse_transform(test_predict)

# 计算训练集和测试集的均方根误差（RMSE）
train_rmse = np.sqrt(np.mean((train_predict - y_train.reshape(-1, 1))**2))
test_rmse = np.sqrt(np.mean((test_predict - y_test.reshape(-1, 1))**2))

# 打印性能指标
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')
print(pd.DataFrame(train_predict).describe())
print(pd.DataFrame(test_predict).describe())

# 绘制实际值和预测值
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(y_train, label='Actual Train Returns')
plt.plot(train_predict, label='Predicted Train Returns')
plt.title('Train Data')
plt.xlabel('Time Step')
plt.ylabel('Return')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_test, label='Actual Test Returns')
plt.plot(test_predict, label='Predicted Test Returns')
plt.title('Test Data')
plt.xlabel('Time Step')
plt.ylabel('Return')
plt.legend()

plt.tight_layout()
plt.show()