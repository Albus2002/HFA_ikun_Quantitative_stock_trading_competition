import requests
import socket
import json
import time
import logging
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import joblib
from math import sqrt



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def cal_Press(bidprice: list, askprice: list, bidvolume: list, askvolume: list):

    buy_num = 0
    buy_den = 0
    sell_num = 0
    sell_den = 0
    mid = (bidprice[0] + askprice[0]) / 2

    for i in range(10):
        buy_num += bidvolume[i] * (mid / (mid - bidprice[i]))
        buy_den += mid / (mid - bidprice[i])
        sell_num += askvolume[i] * (mid / (askprice[i] - mid))
        sell_den += mid / (askprice[i] - mid)

    pbuy = buy_num / buy_den
    psell = sell_num / sell_den
    press = np.log(pbuy) - np.log(psell)
    
    return press

def cal_HR2(bidprice: list, askprice: list):

    HR = [0 for i in range(5)]

    for i in range(5):
        HR[i] = ((bidprice[i] - bidprice[i+1]) - (askprice[i+1] - askprice[i])) / ((bidprice[i] - bidprice[i+1]) + (askprice[i+1] - askprice[i]))

    return HR

def cal_QR2(bidvolume: list, askvolume: list):

    QR = [0 for i in range(5)]

    for i in range(5):
        QR[i] = (bidvolume[i] - askvolume[i]) / (bidvolume[i] + askvolume[i])

    return QR

def cal_VOI2(price: float, price_1: float, 
             bidvolume: list, bidvolume_1: list, 
             askvolume: list, askvolume_1: list, 
             trade_volume: int):

    VOI = 0

    if price == price_1:
        VOI = (bidvolume[0] - bidvolume_1[0]) - (askvolume[0] - askvolume_1[0])
    elif price > price_1:
        VOI = trade_volume
    else:
        VOI = -trade_volume

    return VOI

def cal_Spread(askprice:list,bidprice:list):
    Spread = 0
    Spread = askprice[0] - bidprice[0]
    return Spread

def cal_tempValue(trade_value:float,trade_value_1:float):
    return trade_value - trade_value_1


def cal_tempVolume(trade_volume:float,trade_volume_1:float):
    return trade_volume - trade_volume_1



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
        super().__init__(username, password)
        self.api = InterfaceClass("https://trading.competition.ubiquant.com")
        self.tmp_trade_volume = 800
        self.sharesholding = [0 for i in range(29)]
        self.sharesvalue = [0 for i in range(29)]
        self.buying = [0 for i in range(29)]
        self.selling = [0 for i in range(29)]
        self.bought = [0 for i in range(29)]
        self.sold = [0 for i in range(29)]
        self.orders = dict([(i, {}) for i in range(29)])
        self.buy_interval = [0 for i in range(29)]
        self.sell_interval = [0 for i in range(29)]
        self.ask_volume = [[] for i in range(29)]
        self.bid_volume = [[] for i in range(29)]
        self.trade_value = [0 for i in range(29)]
        self.trade_volume = [0 for i in range(29)]
        self.last_price = [0 for i in range(29)]
        self.volume_flag = [0 for i in range(29)]
        self.revenue = [0 for i in range(29)]
        self.input_features = [
            'BidPrice1', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5',
            'AskPrice1', 'AskPrice2', 'AskPrice3', 'AskPrice4', 'AskPrice5',
            'BidVolume1', 'BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5',
            'AskVolume1', 'AskVolume2', 'AskVolume3', 'AskVolume4', 'AskVolume5',
            'Spread', 'tempVolume', 'tempValue', 'VOI',
            'QR1', 'QR2', 'QR3', 'QR4', 'QR5',
            'HR1', 'HR2', 'HR3', 'HR4', 'HR5', 'Press'
        ]
        self.trade_list = [1,3,5]
        self.pred_result = [[] for i in range(29)]
        self.sharesholding = [0 for i in range(29)]
        self.pnl = 0
        self.pnl_threshold = 1000
        self.value_threshold = 600000
        self.pnl_threshold_minus = -1000
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
        return response
    def init(self):
        # print(self.token_ub)
        response = self.api.sendGetGameInfo(self.token_ub)
        if response["status"] == "Success":
            self.start_time = response["next_game_start_time"]
            self.running_days = response["next_game_running_days"]
            self.running_time = response["next_game_running_time"]
            self.time_ratio = response["next_game_time_ratio"]
        self.GetInstruments()
        self.day = 0
        print("finished.\n")
    def bod(self):
        pass        
    def work(self): 
        for i in self.trade_list:
            print("working...")
            stockID = i
            
            LOB = self.api.sendGetLimitOrderBook(self.token_ub, self.instruments[stockID - 1])
            if LOB['status'] == 'Success':
                if self.volume_flag[stockID - 1] == 0:
                    self.ask_volume[stockID - 1] = LOB['lob']['askvolume']
                    self.bid_volume[stockID - 1] = LOB['lob']['bidvolume']
                    self.trade_value[stockID - 1] = LOB['lob']['trade_value']
                    self.trade_volume[stockID - 1] = LOB['lob']['trade_volume']
                    self.last_price[stockID - 1] = LOB['lob']['last_price']
                    self.volume_flag[stockID - 1] = 1
                    return
                else:
                    TOB = self.api.sendGetTrade(self.token_ub,self.instruments[stockID - 1])
                    if LOB["status"] == "Success" and TOB['status'] == "Success":
                        t = ConvertToSimTime_us(self.start_time, self.time_ratio, self.day, self.running_time)
                        HR = cal_HR2(LOB["lob"]["bidprice"],LOB["lob"]["askprice"])
                        QR = cal_QR2(LOB['lob']['bidvolume'],LOB['lob']['askvolume'])
                        VOI = cal_VOI2(LOB['lob']['last_price'],self.last_price[stockID - 1],LOB['lob']['bidvolume'],self.bid_volume[stockID - 1],LOB['lob']['askvolume'],self.ask_volume[stockID - 1],LOB['lob']['trade_volume'])
                        Spread = cal_Spread(LOB['lob']['askprice'],LOB['lob']['bidprice'])
                        tempValue = cal_tempValue(LOB['lob']['trade_value'],self.trade_value[stockID - 1])
                        tempVolume = cal_tempVolume(LOB['lob']['trade_volume'],self.trade_volume[stockID - 1])
                        Press = cal_Press(LOB['lob']['bidprice'],LOB['lob']['askprice'],LOB['lob']['bidvolume'],LOB['lob']['askvolume'])

                        single_data = []
                        single_data.extend(LOB['lob']['bidprice'][0:5])
                        single_data.extend(LOB['lob']['askprice'][0:5])
                        single_data.extend(LOB['lob']['bidvolume'][0:5])
                        single_data.extend(LOB['lob']['askvolume'][0:5])
                        single_data.extend([Spread])
                        single_data.extend([tempValue])
                        single_data.extend([tempVolume])
                        single_data.extend([VOI])
                        single_data.extend(QR)
                        single_data.extend(HR)
                        single_data.extend([Press])
                        print(single_data)

                        single_data_d = {self.input_features[i]: single_data[i] for i in range(len(self.input_features))}
                        single_data_df = pd.DataFrame(single_data_d, index=[0])
                        svr_model = joblib.load(f'svr_model_{stockID}.joblib')
                        scaler_model = joblib.load(f'scaler_model_{stockID}.joblib')

                        single_data_scaler = scaler_model.transform(single_data_df)
                        prediction = svr_model.predict(single_data_scaler)
                        self.buy_interval[stockID - 1] += 1
                        self.sell_interval[stockID - 1] += 1
                        self.pred_result[stockID - 1].extend([prediction])
                        print(pd.DataFrame(self.pred_result[stockID - 1]).describe())
                        sorted_data = sorted(self.pred_result[stockID - 1])
                        # 找出特定数据值的索引
                        value_index = sorted_data.index(prediction)
                        # 计算数据值的位置百分比
                        # 注意：由于索引是从0开始的，我们需要加1
                        # 并且由于我们需要百分位数，我们需要除以数据总数并乘以100
                        percentile_rank = ((value_index + 1) / len(sorted_data)) * 100
                        if percentile_rank  > 0.75 and self.buy_interval[stockID - 1] > 400:  # 如果预测为正，考虑买入
                            t = ConvertToSimTime_us(self.start_time, self.time_ratio, self.day, self.running_time)
                            order_res = self.api.sendOrder(self.token_ub, self.instruments[stockID - 1], t, 'buy', LOB['lob']['askprice'][0], self.tmp_trade_volume)
                            trade_res = self.api.sendGetTrade(self.token_ub, self.instruments[stockID - 1])
                            order_index = order_res['index']
                            order_time = order_res['localtime']                    

                            # 这个地方判断下单是否成功
                            if order_res['status'] == 'Success':
                                self.buying[stockID - 1] += self.tmp_trade_volume
                                self.orders[stockID - 1][order_index] = order_time
                                
                            # 这个地方判断是否成交，但还没想好如果没成交应该怎么操作，考虑直接取消买单ing
                            if trade_res['status'] == 'Success':
                                try:
                                    trade_index = trade_res['trade_list'][0]['order_index']
                                    if trade_index == order_index:
                                        self.buying[stockID - 1] -= self.tmp_trade_volume
                                        self.sharesholding[stockID - 1] += self.tmp_trade_volume
                                        try:
                                            self.orders[stockID - 1].pop(order_index)
                                        except KeyError:
                                            continue
                                    else:
                                        self.buying[stockID - 1] -= self.tmp_trade_volume
                                        self.api.sendCancel(self.token_ub, self.instruments[stockID - 1], order_time, order_index)
                                        try:
                                            self.orders[stockID - 1].pop(order_index)
                                        except KeyError:
                                            continue
                                except IndexError:
                                    try:
                                        self.orders[stockID - 1].pop(order_index)
                                    except KeyError:
                                        continue
                                    self.buying[stockID - 1] -= self.tmp_trade_volume
                        elif percentile_rank < 0.25 and self.sharesholding[stockID-1] > 0 and self.buy_interval[stockID - 1] > 400:  # 如果预测为负，考虑卖出
                            t = ConvertToSimTime_us(self.start_time, self.time_ratio, self.day, self.running_time)
                            order_res = self.api.sendOrder(self.token_ub, self.instruments[stockID - 1], t, 'sell', LOB['lob']['askprice'][0], self.tmp_trade_volume)
                            trade_res = self.api.sendGetTrade(self.token_ub, self.instruments[stockID - 1])
                            order_index = order_res['index']
                            order_time = order_res['localtime']                    

                            # 这个地方判断下单是否成功
                            if order_res['status'] == 'Success':
                                self.buying[stockID - 1] += self.tmp_trade_volume
                                self.orders[stockID - 1][order_index] = order_time
                                
                            # 这个地方判断是否成交，但还没想好如果没成交应该怎么操作，考虑直接取消买单ing
                            if trade_res['status'] == 'Success':
                                try:
                                    trade_index = trade_res['trade_list'][0]['order_index']
                                    if trade_index == order_index:
                                        self.buying[stockID - 1] -= self.tmp_trade_volume
                                        self.sharesholding[stockID - 1] += self.tmp_trade_volume
                                        try:
                                            self.orders[stockID - 1].pop(order_index)
                                        except KeyError:
                                            continue
                                    else:
                                        self.buying[stockID - 1] -= self.tmp_trade_volume
                                        self.api.sendCancel(self.token_ub, self.instruments[stockID - 1], order_time, order_index)
                                        try:
                                            self.orders[stockID - 1].pop(order_index)
                                        except KeyError:
                                            continue
                                except IndexError:
                                    try:
                                        self.orders[stockID - 1].pop(order_index)
                                    except KeyError:
                                        continue
                                    self.buying[stockID - 1] -= self.tmp_trade_volume
                        self.ask_volume[stockID - 1] = LOB['lob']['askvolume']
                        self.bid_volume[stockID - 1] = LOB['lob']['bidvolume']
                        self.trade_value[stockID - 1] = LOB['lob']['trade_value']
                        self.trade_volume[stockID - 1] = LOB['lob']['trade_volume']
                        self.last_price[stockID - 1] = LOB['lob']['last_price']
                        self.volume_flag[stockID - 1] = 1
                        # askprice_1 = float(LOB["lob"]["askprice"][0])
                        # t = ConvertToSimTime_us(self.start_time, self.time_ratio, self.day, self.running_time)
                        # response = self.api.sendOrder(self.token_ub, self.instruments[stockID], t, "buy", askprice_1, 100)
    def work_eod(self):
        # 这个函数用来在收盘前撤单并清仓
        stockID_list = self.trade_list
        t = ConvertToSimTime_us(self.start_time, self.time_ratio, self.day, self.running_time)
        for ID in stockID_list:
            # 先撤单
            for order_id in list(self.orders[ID-1].keys()):
                cancel_res = self.api.sendCancel(self.token_ub, self.instruments[ID-1], self.orders[ID-1][order_id], order_id)
                if cancel_res['status'] == 'Success':
                    self.orders[ID-1].pop(order_id)
            # 再清仓
            if self.sharesholding[ID-1] > 0:
                while(1):
                    orderBook = self.api.sendGetLimitOrderBook(token_ub=self.token_ub, instrument=self.instruments[ID-1])

                    order_res = self.api.sendOrder(self.token_ub, self.instruments[ID-1], t, 'sell', orderBook['lob']['askprice'][0], self.tmp_trade_volume)
                    if order_res['status'] == 'Success':
                        self.sharesholding[ID - 1] -= self.tmp_trade_volume
                    if self.sharesholding[ID - 1] <= 0:
                        break

    def check_status(self):
        
        # 这个函数用来每隔一段时间自动读取和更新当前持仓数据和订单数据，同时做止盈/减仓
        active_orders = self.api.sendGetActiveOrder(self.token_ub)
        print("hfAklewhfkjebfkgfkehfklWbefkjwebfkuwhflhewfljhwefjkweHwef")
        print(active_orders['status'])
        user_info = self.api.sendGetUserInfo(self.token_ub)
        t = ConvertToSimTime_us(self.start_time, self.time_ratio, self.day, self.running_time)

        # 更新订单数据（maybe考虑在这里撤掉所有未成交订单）
        if active_orders['status'] == 'Success':
            
            for ID in self.trade_list:
                tmp_active_orders = active_orders['instruments'][ID]
                # self.orders[ID] = {}
                real_order_indices = []
                for order in tmp_active_orders['active_orders']:
                    real_order_indices.append(order['order_index'])
                
                for key in self.orders[ID].keys():
                    if key not in real_order_indices:
                        self.orders[ID].pop(key)
                for order_id in real_order_indices:
                    if order_id not in self.orders[ID].keys():
                        self.orders[ID][order_id] = t
        # 更新持仓数据
        if user_info['status'] == 'Success':
            
            real_data = user_info['rows']
            tmp_pnl = 0

            for ID in self.trade_list:
                
                orderBook = self.api.sendGetLimitOrderBook(self.token_ub, self.instruments[ID - 1])
                real_shares = real_data[ID - 1]['share_holding']
                self.sharesholding[ID - 1] = real_shares
                self.sharesvalue[ID - 1] = real_shares * orderBook['lob']['last_price']
                tmp_pnl += real_data[ID - 1]['pnl']
            
            self.pnl = tmp_pnl
        # 达到目标后止盈
        if self.pnl > self.pnl_threshold:

            for ID in self.trade_list:

                orderBook = self.api.sendGetLimitOrderBook(self.token_ub, self.instruments[ID-1])
                tmp_trading_volume = min((self.sharesholding[ID - 1]*0.3 // 100 * 100), self.tmp_trade_volume)
                # 下卖单
                if orderBook['status'] == 'Success' and self.sharesholding[ID - 1] > 1000:
                    
                    order_res = self.api.sendOrder(self.token_ub, self.instruments[ID - 1], t, 'sell', orderBook['lob']['bidprice'][0], tmp_trading_volume)
                    order_index = order_res['index']
                    order_time = order_res['localtime']  
                
                # 验证是否成功
                    if order_res['status'] == 'Success':
                        self.selling[ID-1] += tmp_trading_volume
                        self.orders[ID-1][order_index] = order_time

            self.pnl_threshold += self.pnl_threshold

        elif self.pnl < self.pnl_threshold_minus:

            for ID in self.trade_list:

                orderBook = self.api.sendGetLimitOrderBook(self.token_ub, self.instruments[ID-1])
                tmp_trading_volume = min((self.sharesholding[ID - 1]*0.3 // 100 * 100), self.tmp_trade_volume * 2)
                # 下卖单
                if orderBook['status'] == 'Success' and self.sharesholding[ID - 1] > 1000:
                    
                    order_res = self.api.sendOrder(self.token_ub, self.instruments[ID - 1], t, 'sell', orderBook['lob']['bidprice'][0], tmp_trading_volume)
                    print(order_res)
                    order_index = order_res['index']
                    order_time = order_res['localtime']  
                
                # 验证是否成功
                    if order_res['status'] == 'Success':
                        self.selling[ID-1] += tmp_trading_volume
                        self.orders[ID-1][order_index] = order_time

        # 仓位过多后卖掉一部分
        if sum(self.sharesvalue) > self.value_threshold:
            for ID in self.trade_list:

                orderBook = self.api.sendGetLimitOrderBook(self.token_ub, self.instruments[ID-1])
                tmp_trading_volume = max((self.sharesholding[ID - 1]*0.3 // 100 * 100), self.tmp_trade_volume * 2)
                # 下卖单
                if orderBook['status'] == 'Success' and self.sharesholding[ID - 1] > 0:
                    
                    order_res = self.api.sendOrder(self.token_ub, self.instruments[ID - 1], t, 'sell', orderBook['lob']['bidprice'][0], tmp_trading_volume)
                    order_index = order_res['index']
                    order_time = order_res['localtime']  
                # 验证是否成功
                    if order_res['status'] == 'Success':
                        self.selling[ID-1] += tmp_trading_volume
                        self.orders[ID-1][order_index] = order_time


    def eod(self):
        print("prediction dis:\n")
        for stockID in self.trade_list:
            print(self.pred_result[stockID - 1])
        
        print("revenue:\n")
        for stockID in self.trade_list:
            print(self.revenue[stockID - 1])
        
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
        print(url)
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
endWaitTime = 3000
tmp_cnt = 0
while True:
    if ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) < SimTimeLen:
        break
    else:
        bot.day += 1

while bot.day <= bot.running_days:
    while True:
        if ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) > -900:
            break
    bot.bod()
    now = round(ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time))
    for s in range(now, SimTimeLen + endWaitTime):
        while True:
            if ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) >= s:
                break
        t = ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time)
        logger.info("Work Time: {}".format(t))
        tmp_cnt += 1
        print(f'tmp:{tmp_cnt}')
        if tmp_cnt == 60:
            bot.check_status()
            tmp_cnt = 0
        if t < SimTimeLen - 60:
            bot.work()
        elif t < SimTimeLen - 30:
            bot.work_eod() # 清仓
    bot.eod()
    bot.day += 1
bot.final()
