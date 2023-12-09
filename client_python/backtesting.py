import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV
import joblib

file_paths = []

for name in range(1, 11):
    tmp_path = os.path.join('../dataset', str(name))

    for file_name in os.listdir(tmp_path):
        file_path = os.path.join(tmp_path, file_name)

        if os.path.isfile(file_path):
            file_paths.append(file_path)

def cal_Sharpe(prices: np.ndarray):

    '''
    This function uses realized volatility to measure the Sharpe ratio of a stock
    with ultra-high frequency data and 1-minute interval
    '''

    int_price_60 = prices[::60]
    log_return = np.diff(np.log(int_price_60))
    avg_log_return = np.mean(log_return)
    realized_vol = np.sqrt(np.sum(np.square(log_return)))
    sharpe = avg_log_return / realized_vol

    return sharpe*10000

num_files = len(file_paths)
test_res = {'sharpe': [], 
            'max_return': [], 
            'max_loss': [], 
            'final_return': [], 
            'press_mean': [], 
            'press_std': []}

for j in range(29):
    print(f'testing {j} ...')
    df = pd.DataFrame([])
    for k in range(num_files):
        if k % 29 == j:
            df = pd.concat([df, pd.read_csv(file_paths[k])])

    df['MidPrice'] = (df.BidPrice1 + df.AskPrice1) / 2
    df['Spread'] = df.AskPrice1 - df.BidPrice1
    df['tempVolume'] = df.TradeVolume.diff(1).apply(lambda x: x if x > 0 else 0)
    df['tempValue'] = df.TradeValue.diff(1).apply(lambda x: x if x > 0 else 0)
    df['Return'] = np.log(df.LastPrice / df.LastPrice.shift(1))
    df['VOI'] = df.tempVolume*(df.Return.apply(lambda x: 1 if x > 0 else x).apply(lambda x: -1 if x < 0 else x))
    df['VOI'][df['VOI']==0] = df.BidVolume1.diff(1) - df.AskVolume1.diff(1)
    df['QR1'] = (df.BidVolume1 - df.AskVolume1) / (df.BidVolume1 + df.AskVolume1)
    df['QR2'] = (df.BidVolume2 - df.AskVolume2) / (df.BidVolume2 + df.AskVolume2)
    df['QR3'] = (df.BidVolume3 - df.AskVolume3) / (df.BidVolume3 + df.AskVolume3)
    df['QR4'] = (df.BidVolume4 - df.AskVolume4) / (df.BidVolume4 + df.AskVolume4)
    df['QR5'] = (df.BidVolume5 - df.AskVolume5) / (df.BidVolume5 + df.AskVolume5)
    df['HR1'] = ((df.BidPrice1 - df.BidPrice2) - (df.AskPrice2 - df.AskPrice1) / \
             (df.BidPrice1 - df.BidPrice2) + (df.AskPrice2 - df.AskPrice1))
    df['HR2'] = ((df.BidPrice2 - df.BidPrice3) - (df.AskPrice3 - df.AskPrice2) / \
             (df.BidPrice2 - df.BidPrice3) + (df.AskPrice3 - df.AskPrice2))
    df['HR3'] = ((df.BidPrice3 - df.BidPrice4) - (df.AskPrice4 - df.AskPrice3) / \
             (df.BidPrice3 - df.BidPrice4) + (df.AskPrice4 - df.AskPrice3))
    df['HR4'] = ((df.BidPrice4 - df.BidPrice5) - (df.AskPrice5 - df.AskPrice4) / \
             (df.BidPrice4 - df.BidPrice5) + (df.AskPrice5 - df.AskPrice4))
    df['HR5'] = ((df.BidPrice5 - df.BidPrice6) - (df.AskPrice6 - df.AskPrice5) / \
             (df.BidPrice5 - df.BidPrice6) + (df.AskPrice6 - df.AskPrice5))
    tmp_df = df.iloc[:, 4:45]

    tmp_df['buy_num'] = 0
    tmp_df['buy_den'] = 0
    tmp_df['sell_num'] = 0
    tmp_df['sell_den'] = 0

    for i in range(10):
        tmp_df.buy_num += tmp_df.iloc[:,20+i] * tmp_df.MidPrice / (tmp_df.MidPrice - tmp_df.iloc[:,i])
        tmp_df.buy_den += tmp_df.MidPrice / (tmp_df.MidPrice - tmp_df.iloc[:,i])
        tmp_df.sell_num += tmp_df.iloc[:,30+i] * tmp_df.MidPrice / (tmp_df.MidPrice - tmp_df.iloc[:,10+i])
        tmp_df.sell_den += tmp_df.MidPrice / (tmp_df.MidPrice - tmp_df.iloc[:,10+i])

    tmp_df['Pbuy'] = tmp_df.buy_num / tmp_df.buy_den
    tmp_df['Psell'] = tmp_df.sell_num / tmp_df.sell_den
    df['Press'] = np.log(tmp_df.Pbuy) - np.log(tmp_df.Psell)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    df.reset_index(inplace=True)
    df['Day'] = df.index // 14400 + 1

    buy_time = []
    sell_time = []
    wealth = 1000000
    stock_value = 0
    stock_holding = 0
    total_wealth_list = []
    stock_num_list = []
    tmp_trade_volume = 500
    for day in range(1, 11):

        # 【【【在这里改成基于SVM预测的买卖信号】】】
        tmp_df = df[df.Day==day]
        buy_time = buy_time + tmp_df[tmp_df.Press > tmp_df.Press.mean() + 1.5*tmp_df.Press.std()][tmp_df.QR1<tmp_df.QR1.mean()][tmp_df.QR2<tmp_df.QR2.mean()][tmp_df.QR3<tmp_df.QR3.mean()].index.to_list()
        sell_time = sell_time + tmp_df[tmp_df.Press < tmp_df.Press.mean() - 1.5*tmp_df.Press.std()].index.to_list()
    
    for t in range(144000):
        data_t = df.iloc[t]
        price_buy = data_t.AskPrice1
        price_sell = data_t.BidPrice1
        if t % 14400 > 14340:
            if stock_holding > 0:
                wealth += 0.9998*stock_holding*price_sell
                stock_holding = 0

        else:
            if t in buy_time:
                if wealth > tmp_trade_volume*price_buy:
                    stock_holding += tmp_trade_volume
                    wealth -= 1.0002*tmp_trade_volume*price_buy
            elif t in sell_time:
                if stock_holding >= tmp_trade_volume:
                    stock_holding -= tmp_trade_volume
                    wealth += 0.9998*tmp_trade_volume*price_sell

        stock_value = stock_holding*data_t.LastPrice
        stock_num_list.append(stock_holding)
        total_wealth_list.append(wealth+stock_value)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot([m for m in range(144000)], total_wealth_list)
    plt.subplot(212)
    plt.plot([m for m in range(144000)], stock_num_list)


    # 存储df
    df_name = 'Stock' + str(j+1) + '.csv'
    performance_fig_name = 'Performace' + str(j+1) + '.png'
    # 需要存数据的时候取消掉注释
    # df.to_csv(df_name)
    plt.savefig(performance_fig_name)
    print(df_name, 'saved.')

    # 存储策略表现数据
    sharpe = cal_Sharpe(np.array(total_wealth_list))
    max_return = max(total_wealth_list) / 1000000 - 1
    max_loss = 1 - min(total_wealth_list) / max(total_wealth_list)
    final_return = total_wealth_list[-1] / 1000000 - 1
    test_res['sharpe'].append(sharpe)
    test_res['max_return'].append(max_return)
    test_res['max_loss'].append(max_loss)
    test_res['final_return'].append(final_return)
    test_res['press_mean'].append(df.Press.mean())
    test_res['press_std'].append(df.Press.std())
    

    print('Stock' + str(j+1) + 'processed.')


res_df = pd.DataFrame(test_res, index=['Stock'+str(i) for i in range(1, 30)])
res_df.to_csv('backtesting_res.csv')