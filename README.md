# HFA_ikun_Quantitative_stock_trading_competition

Albuslyu and Oliverliu 2023.12.9

## 交易模式

client和九坤官方网站https://trading.competition.ubiquant.com/main进行交互。所有股票数据为真实信息，一共29支股票。

## 交易规则

client_python中拥有你所有需要的函数。
发送交易请求需要sendOrder函数，而获取股票信息需要Getlimitorderbook函数。
bot通过每次调用work()函数进行操作，其中也是我们的核心策略。

## 测试指标
pnl:当日持仓+现金价值与初始资金之差。初始资金为100w。
sharp:收益曲线的陡峭程度。
最终的score由二者综合得出(sharp*pnl的加权)

## 具体文件
dataset里的数字标号是原始数据，rebuild_data是计算出训练指标的数据。
client_c是基于c++的客户端，另一个是基于python的。
main.py里是基于SVM进行决策的bot，client_1206是基于press指标的。值得一提的是，press指标也是SVM训练的输入之一。
SVM通过输入35维向量来预测return列，核函数为多项式（经测试后优于其他函数），poly=3,C,gamma参数需要手动调试。
从模拟效果来看，press指标的操作比SVM要更加稳健，测试平均收益远好于SVM。推测应该是加入非训练数据后，SVM模型对于新的输入鲁棒性不足，导致产生预测值的“爆炸”：可能会出现10的20次方的结果。为了修正这一偏差，对于预测决策的判断从原本的“预测为正购买，预测为负卖出”改为“预测值在已有预测结果80%分位以上购买，20%分位以下卖出”的策略，在一段时间内收益能达到单日单支股票（stock3）1000pnl的收益。但在继续加入新数据后，SVM效果依旧会变的很差，甚至无法保证平均收益不亏损。

Press指标的定义在买卖压力指标.txt中，由Oliverliu提出，事实证明，这是一个很好的衡量指标，在bot操作得当（止盈、持仓规模控制模块不出错）的情况下能达到平均单日400pnl以上的收益，峰值pnl能达到8500.
