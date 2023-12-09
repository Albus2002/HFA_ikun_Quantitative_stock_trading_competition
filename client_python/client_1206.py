import requests
import socket
import json
import time
import logging
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from keras import losses
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

press_list = [[] for i in range(29)]
price_list = [[] for i in range(29)]
# pd.DataFrame(np.array(price_list).T, columns=['Stock'+str(i+1) for i in range(29)]).to_csv('price_online.csv')
# pd.DataFrame(np.array(press_list).T, columns=['Stock'+str(i+1) for i in range(29)]).to_csv('press_online.csv')

def cal_Press(bidprice: list, askprice: list, bidvolume: list, askvolume: list):

    buy_num = 0
    buy_den = 0
    sell_num = 0
    sell_den = 0
    mid = (bidprice[0] + askprice[0]) / 2

    for i in range(10):
        if mid == bidprice[i] or mid == askprice[i]:
            return 0
        buy_num += bidvolume[i] * (mid / (mid - bidprice[i]))
        buy_den += mid / (mid - bidprice[i])
        sell_num += askvolume[i] * (mid / (askprice[i] - mid))
        sell_den += mid / (askprice[i] - mid)

    pbuy = buy_num / buy_den
    psell = sell_num / sell_den
    press = np.log(pbuy) - np.log(psell)
    
    return press

# press_his = pd.read_csv('dataset\press_1.csv', index_col=0)
# press_his.replace([np.inf, -np.inf], 0, inplace=True)
# press_upper = press_his.Press.mean() + press_his.Press.std()
# press_lower = press_his.Press.mean() - press_his.Press.std()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def ConvertToSimTime_us(start_time, time_ratio, day, running_time):
    return (time.time() - start_time - (day - 1) * running_time) * time_ratio

class BotsClass:
    def __init__(self, username, password):
        self.username = username
        self.password = password
    def login(self):
        pass
    def init(self):
        pass
    def bod(self):
        pass
    def work(self):
        pass
    def eod(self):
        pass
    def final(self):
        pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def ConvertToSimTime_us(start_time, time_ratio, day, running_time):
    return (time.time() - start_time - (day - 1) * running_time) * time_ratio

class BotsClass:
    def __init__(self, username, password):
        self.username = username
        self.password = password
    def login(self):
        pass
    def init(self):
        pass
    def bod(self):
        pass
    def work(self):
        pass
    def eod(self):
        pass
    def final(self):
        pass

class BotsDemoClass(BotsClass):
    def __init__(self, username, password):
        super().__init__(username, password);
        self.api = InterfaceClass("https://trading.competition.ubiquant.com")
        self.sharesholding = [0 for i in range(29)]
        self.sharesvalue = [0 for i in range(29)]
        self.buying = [0 for i in range(29)]
        self.selling = [0 for i in range(29)]
        self.bought = [0 for i in range(29)]
        self.sold = [0 for i in range(29)]
        self.orders = dict([(i, {}) for i in range(29)])
        self.pnl = 0
        self.pnl_threshold = 2000
        self.value_threshold = 555555
        self.press_upper = [2 for i in range(29)]
        self.press_lower = [-1.5 for i in range(29)]
        self.trade_list = [1, 7, 8, 15, 29]

    def login(self):
        response = self.api.sendLogin(self.username, self.password)
        if response["status"] == "Success":
            self.token_ub = response["token_ub"]
            logger.info("Login Success: {}".format(self.token_ub))
        else:
            logger.info("Login Error: ", response["status"])
    def GetInstruments(self):
        response = self.api.sendGetInstrumentInfo(self.token_ub)
        if response["status"] == "Success":
            self.instruments = []
            for instrument in response["instruments"]:
                self.instruments.append(instrument["instrument_name"])
            logger.info("Get Instruments: {}".format(self.instruments))
    def init(self):
        response = self.api.sendGetGameInfo(self.token_ub)
        if response["status"] == "Success":
            self.start_time = response["next_game_start_time"]
            self.running_days = response["next_game_running_days"]
            self.running_time = response["next_game_running_time"]
            self.time_ratio = response["next_game_time_ratio"]
        self.GetInstruments()
        self.day = 0

    def bod(self):
        # 导入存储的press和price数据，并且继续收集     
        tmp_price_df = pd.read_csv('price_online.csv')
        tmp_press_df = pd.read_csv('press_online.csv')
        for ID in range(29):
            tmp_press = tmp_press_df.loc[ID]
            tmp_press.dropna(inplace=True)
            tmp_press.replace([np.inf, -np.inf], 0, inplace=True)
            self.press_upper[ID] = tmp_press.mean() + 2*tmp_press.std()
            self.press_lower[ID] = tmp_press.mean() - 1.5*tmp_press.std()
            press_list[ID] = tmp_press.to_list()

            tmp_price = tmp_price_df.loc[ID]
            tmp_price.dropna(inplace=True)
            tmp_price.replace([np.inf, -np.inf], 0, inplace=True)
            price_list[ID] = tmp_price.to_list()



    def work(self):

        stockID_list = [1, 7, 8, 15, 29]
        # 确定好数据后再加入其他股票，目标股票id为[1, 2, 10, 13, 16, 20]
        for ID in stockID_list:
            orderBook = self.api.sendGetLimitOrderBook(token_ub=self.token_ub, 
                                                       instrument=self.instruments[ID-1])
            tradeBook = self.api.sendGetTrade(self.token_ub, self.instruments[ID-1])
            tmp_trade_volume = 500
            # print(orderBook)
            press_upper = self.press_upper[ID - 1]
            press_lower = self.press_lower[ID - 1]
            if orderBook['status'] == 'Success' and tradeBook['status'] == 'Success':
                press = cal_Press(bidprice = orderBook['lob']['bidprice'], askprice = orderBook['lob']['askprice'], 
                                bidvolume = orderBook['lob']['bidvolume'], askvolume = orderBook['lob']['askvolume'])
                if press == 0:
                    press_list[ID-1].append(press)
                    price_list[ID-1].append(orderBook['lob']['last_price'])
                    continue

                press_list[ID-1].append(press)
                price_list[ID-1].append(orderBook['lob']['last_price'])
                t = ConvertToSimTime_us(self.start_time, self.time_ratio, self.day, self.running_time)

                if press > press_upper:
                    order_res = self.api.sendOrder(self.token_ub, self.instruments[ID-1], t, 'buy', orderBook['lob']['askprice'][0], tmp_trade_volume)
                    trade_res = self.api.sendGetTrade(self.token_ub, self.instruments[ID-1])
                    order_index = order_res['index']
                    order_time = order_res['localtime']                    

                    # 这个地方判断下单是否成功
                    if order_res['status'] == 'Success':
                        self.buying[ID-1] += tmp_trade_volume
                        self.orders[ID-1][order_index] = order_time
                        
                    # 这个地方判断是否成交，但还没想好如果没成交应该怎么操作，考虑直接取消买单ing
                    if trade_res['status'] == 'Success':
                        try:
                            trade_index = trade_res['trade_list'][0]['order_index']
                            if trade_index == order_index:
                                self.buying[ID-1] -= tmp_trade_volume
                                self.sharesholding[ID-1] += tmp_trade_volume
                                try:
                                    self.orders[ID-1].pop(order_index)
                                except KeyError:
                                    continue
                            else:
                                self.buying[ID-1] -= tmp_trade_volume
                                self.api.sendCancel(self.token_ub, self.instruments[ID-1], order_time, order_index)
                                try:
                                    self.orders[ID-1].pop(order_index)
                                except KeyError:
                                    continue
                        except IndexError:
                            try:
                                self.orders[ID-1].pop(order_index)
                            except KeyError:
                                continue
                            self.buying[ID-1] -= tmp_trade_volume
                    
                elif press < press_lower and self.sharesholding[ID-1] > 0:
                    order_res = self.api.sendOrder(self.token_ub, self.instruments[ID-1], t, 'sell', orderBook['lob']['bidprice'][0], tmp_trade_volume)
                    trade_res = self.api.sendGetTrade(self.token_ub, self.instruments[ID-1])
                    order_index = order_res['index']
                    order_time = order_res['localtime']                  

                    # 这个地方判断下单是否成功
                    if order_res['status'] == 'Success':
                        self.selling[ID-1] += tmp_trade_volume
                        self.orders[ID-1][order_index] = order_time

                    # 这个地方判断是否成交，但还没想好如果没成交应该怎么操作，考虑直接取消买单ing
                    if trade_res['status'] == 'Success':
                        try: 
                            trade_index = trade_res['trade_list'][0]['order_index']
                            if trade_index == order_index:
                                self.selling[ID-1] -= tmp_trade_volume
                                self.sharesholding[ID-1] += tmp_trade_volume
                                try:
                                    self.orders[ID-1].pop(order_index)
                                except KeyError:
                                    continue
                            else:
                                self.selling[ID-1] -= tmp_trade_volume
                                self.api.sendCancel(self.token_ub, self.instruments[ID-1], order_time, order_index)
                                try:
                                    self.orders[ID-1].pop(order_index)
                                except KeyError:
                                    continue
                        
                        except IndexError:
                            try:
                                self.orders[ID-1].pop(order_index)
                            except KeyError:
                                continue
                            self.buying[ID-1] -= tmp_trade_volume

                            

    def check_status(self):
        
        # 这个函数用来每隔一段时间自动读取和更新当前持仓数据和订单数据，同时做止盈/减仓
        active_orders = self.api.sendGetActiveOrder(self.token_ub)
        user_info = self.api.sendGetUserInfo(self.token_ub)
        t = ConvertToSimTime_us(self.start_time, self.time_ratio, self.day, self.running_time)

        # 更新订单数据（maybe考虑在这里撤掉所有未成交订单）
        if active_orders['status'] == 'Success':
            
            for ID in range(29):
                tmp_active_orders = active_orders['instruments'][ID]
                # self.orders[ID] = {}
                real_order_indices = []
                # for order in tmp_active_orders['active_orders']:
                    # real_order_indices.append(order['order_index'])
                
                # for key in self.orders[ID].keys():
                    # if key not in real_order_indices:
                        # self.orders[ID].pop(key)
                for order_id in real_order_indices:
                    if order_id not in self.orders[ID].keys():
                        self.orders[ID][order_id] = t

        # 更新持仓数据
        if user_info['status'] == 'Success':
            
            real_data = user_info['rows']
            tmp_pnl = 0

            for ID in range(29):
                
                orderBook = self.api.sendGetLimitOrderBook(self.token_ub, self.instruments[ID])
                real_shares = real_data[ID]['share_holding']
                self.sharesholding[ID] = real_shares
                self.sharesvalue[ID] = real_shares * orderBook['lob']['last_price']
                tmp_pnl += real_data[ID]['pnl']
            
            self.pnl = tmp_pnl
        
        # 达到目标后止盈
        if self.pnl > self.pnl_threshold:

            for ID in range(29):

                orderBook = self.api.sendGetLimitOrderBook(self.token_ub, self.instruments[ID])
                tmp_trading_volume = min((self.sharesholding[ID]*0.3 // 100 * 100), 2000)
                # 下卖单
                if orderBook['status'] == 'Success' and self.sharesholding[ID] > 1000:
                    
                    order_res = self.api.sendOrder(self.token_ub, self.instruments[ID], t, 'sell', orderBook['lob']['bidprice'][0], tmp_trading_volume)
                     
                
                # 验证是否成功
                    if order_res['status'] == 'Success':
                        order_index = order_res['index']
                        order_time = order_res['localtime'] 
                        self.selling[ID] += tmp_trading_volume
                        self.orders[ID][order_index] = order_time

            self.pnl_threshold += self.pnl_threshold

        # 仓位过多后卖掉一部分
        if sum(self.sharesvalue) > self.value_threshold:
            for ID in range(29):

                orderBook = self.api.sendGetLimitOrderBook(self.token_ub, self.instruments[ID])
                tmp_trading_volume = min((self.sharesholding[ID]*0.3 // 100 * 100), 2000)
                # 下卖单
                if orderBook['status'] == 'Success' and self.sharesholding[ID] > 2000:
                    
                    order_res = self.api.sendOrder(self.token_ub, self.instruments[ID], t, 'sell', orderBook['lob']['bidprice'][0], tmp_trading_volume)
                    
                
                # 验证是否成功
                    if order_res['status'] == 'Success':
                        order_index = order_res['index']
                        order_time = order_res['localtime']  
                        self.selling[ID] += tmp_trading_volume
                        self.orders[ID][order_index] = order_time

    
    def work_eod(self):
        # 这个函数用来在收盘前撤单并清仓
        stockID_list = [1, 7, 8, 15, 29]
        t = ConvertToSimTime_us(self.start_time, self.time_ratio, self.day, self.running_time)

        for ID in stockID_list:

            # 先撤单
            for order_id in list(self.orders[ID-1].keys()):
                cancel_res = self.api.sendCancel(self.token_ub, self.instruments[ID-1], 
                                                 self.orders[ID-1][order_id], order_id)
                if cancel_res['status'] == 'Success':
                    self.orders[ID-1].pop(order_id)
            
            # 再清仓
            if self.sharesholding[ID-1] > 0:

                orderBook = self.api.sendGetLimitOrderBook(token_ub=self.token_ub, 
                                                       instrument=self.instruments[ID-1])
                tmp_trading_volume = self.sharesholding[ID-1]
                order_res = self.api.sendOrder(self.token_ub, self.instruments[ID-1], t, 'sell', orderBook['lob']['askprice'][0], tmp_trading_volume)
                if order_res['status'] == 'Success':
                    self.sharesholding[ID-1] -= tmp_trading_volume


    def eod(self):
        '''
        max_len_press = 0
        max_len_price = 0
        for ID in range(29):
            max_len_price = max(max_len_price, len(price_list[ID]))
            max_len_press = max(max_len_press, len(press_list[ID]))
        
        for ID in range(29):
            len_price = len(price_list[ID])
            len_press = len(press_list[ID])
            tmp_price_list = price_list[ID]
            tmp_press_list = press_list[ID]
            price_list[ID] = tmp_price_list + [0 for i in range(max_len_price-len_price)]
            press_list[ID] = tmp_press_list + [0 for i in range(max_len_press-len_press)]
        df_press = pd.DataFrame(np.array(press_list).T, columns=['Stock'+str(i+1) for i in range(29)])
        df_price = pd.DataFrame(np.array(price_list).T, columns=['Stock'+str(i+1) for i in range(29)])
        '''
        df_press = pd.DataFrame(press_list)
        df_price = pd.DataFrame(price_list)
        df_press.to_csv('press_online.csv')
        print('press saved.')
        df_price.to_csv('price_online.csv')
        print('price saved.')
        print('press size:', df_press.shape)
        print('price size:', df_price.shape)

    def final(self):
        pass

class InterfaceClass:
    def __init__(self, domain_name):
        self.domain_name = domain_name
        self.session = requests.Session()
    def sendLogin(self, username, password):
        url = self.domain_name + "/api/Login"
        data = {
            "user": username,
            "password": password
        }
        response = self.session.post(url, data=json.dumps(data)).json()
        return response
    
    def sendGetGameInfo(self, token_ub):
        url = self.domain_name + "/api/TradeAPI/GetGAmeInfo"

    def sendOrder(self, token_ub, instrument, localtime, direction, price, volume):
        logger.debug("Order: Instrument: {}, Direction:{}, Price: {}, Volume:{}".format(instrument, direction, price, volume))
        url = self.domain_name + "/api/TradeAPI/Order"
        data = {
            "token_ub": token_ub,
            "user_info": "NULL",
            "instrument": instrument,
            "localtime": localtime,
            "direction": direction,
            "price": price,
            "volume": volume,
        }
        response = self.session.post(url, data=json.dumps(data)).json()
        return response

    def sendCancel(self, token_ub, instrument, localtime, index):
        logger.debug("Cancel: Instrument: {}, index:{}".format(instrument, index))
        url = self.domain_name + "/api/TradeAPI/Cancel"
        data = {
            "token_ub": token_ub,
            "user_info": "NULL",
            "instrument": instrument,
            "localtime": 0,
            "index": index
        }
        response = self.session.post(url, data=json.dumps(data)).json()
        return response

    def sendGetLimitOrderBook(self, token_ub, instrument):
        logger.debug("GetLimitOrderBOok: Instrument: {}".format(instrument))
        url = self.domain_name + "/api/TradeAPI/GetLimitOrderBook"
        data = {
            "token_ub": token_ub,
            "instrument": instrument
        }
        response = self.session.post(url, data=json.dumps(data)).json()
        return response

    def sendGetUserInfo(self, token_ub):
        logger.debug("GetUserInfo: ")
        url = self.domain_name + "/api/TradeAPI/GetUserInfo"
        data = {
            "token_ub": token_ub,
        }
        response = self.session.post(url, data=json.dumps(data)).json()
        return response

    def sendGetGameInfo(self, token_ub):
        logger.debug("GetGameInfo: ")
        url = self.domain_name + "/api/TradeAPI/GetGameInfo"
        data = {
            "token_ub": token_ub,
        }
        response = self.session.post(url, data=json.dumps(data)).json()
        return response

    def sendGetInstrumentInfo(self, token_ub):
        logger.debug("GetInstrumentInfo: ")
        url = self.domain_name + "/api/TradeAPI/GetInstrumentInfo"
        data = {
            "token_ub": token_ub,
        }
        response = self.session.post(url, data=json.dumps(data)).json()
        return response

    def sendGetTrade(self, token_ub, instrument):
        logger.debug("GetTrade: Instrment: {}".format(instrument))
        url = self.domain_name + "/api/TradeAPI/GetTrade"
        data = {
            "token_ub": token_ub,
            "instrument_name": instrument
        }
        response = self.session.post(url, data=json.dumps(data)).json()
        return response

    def sendGetActiveOrder(self, token_ub):
        logger.debug("GetActiveOrder: ")
        url = self.domain_name + "/api/TradeAPI/GetActiveOrder"
        data = {
            "token_ub": token_ub,
        }
        response = self.session.post(url, data=json.dumps(data)).json()
        return response
    

bot = BotsDemoClass("UBIQ_TEAM207", "o5IzFTxQq")
bot.login()
bot.init()

SimTimeLen = 14400
endWaitTime = 300
online_flag = 0
count = 0

while True:
    if ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) < SimTimeLen:
        break
    else:
        bot.day += 1

while bot.day <= bot.running_days:
    while True:
        if ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) > -900:
            break
    if online_flag == 1:
        bot.bod
    now = round(ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time))
    # print('Now!')
    for s in range(now, SimTimeLen + endWaitTime):
        while True:
            if ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) >= s:
                # print('Not in working time...')
                break
        t = ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time)
        logger.info("Work Time: {}".format(t))

        
        if t < SimTimeLen - 60:
            if count % 100 == 0:
                bot.check_status()
                print(press_list)
                print(price_list)
            else:
                bot.work()
            count += 1
        elif t < SimTimeLen - 30:
            bot.work_eod()
            # 撤掉所有未成交的单并清仓

    bot.eod()
    bot.day += 1
    online_flag = 1
bot.final()