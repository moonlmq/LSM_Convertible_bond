# import numpy as np
# from time import time
# import pandas as pd
# import math
# import datetime
# import tushare as ts
# from arch import arch_model
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from matplotlib.font_manager import FontProperties
# from urllib.request import urlretrieve
# import os
#
# # 导入转债数据，缺失值填充，导入无风险利率：1年期银行间拆借利率
# file = pd.read_excel(r'e:\bond\2018年上市转债数据.xlsx')
# file = pd.read_excel(r'e:\bond\正在交易可转债数据.xlsx')
#
#
# file = file[:-2]
# file.iloc[:, 8].fillna(70, inplace=True)
# shibor = pd.read_csv(r'e:\bond\Shibor1Y.csv', encoding='GBK')
#
# def choose(num):
#     '''
#     :param num: 76个转债序号
#     :return:
#     '''
#     global stock_no, date, date_6m
#     stock_no = file.iloc[num, 2][:6]  # 股票代码
#     stock_no = str(1 - int(stock_no[0] == '6')) + stock_no
#     date = file.iloc[num, 4].date()  # 转债日期
#     date_6m = date + datetime.timedelta(days=180)  # 转债6个月后日期
#     date = str(date);
#     date_6m = str(date_6m)
# ############################ 计算股票波动率
# def volatility(select=0):
#     '''
#     计算股票的波动率
#     :param select: 0代表garch波动率 1代表1月波动率 2代表1年波动率
#     :return: 波动率(收益标准差)，
#     '''
#     global stock
#     start_date = '20150101'
#     end_date = '20190310'
#     url = 'http://quotes.money.163.com/service/chddata.html?code=' + stock_no + '&start=' + start_date + '&end=' + end_date + '&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'
#     ############ 股票走势数据存在'e:\\bond'的文件夹下
#     filename = 'e:\\bond\\' + stock_no + '.csv'
#     if not os.path.exists(filename): urlretrieve(url, filename)
#     stock = pd.read_csv(filename, encoding='gbk')
#     stock = stock.iloc[:, [0, 3]].set_index('日期').iloc[:, 0]
#     stock = stock.sort_index()
#     # 剔除停牌的日期
#     stock = stock[stock > 0]
#     # 简单收益率
#     # stock_pct = stock.pct_change().dropna()
#     # 对数收益率
#     stock_log = stock.apply(lambda x: math.log(x)).diff().dropna()
#
#     ###########循环日期，得到波动率
#     datelist = stock_log[date:].index
#     if select == 1:
#         V_1m = stock_log.rolling(window=20).std().dropna() * np.sqrt(250)
#         return V_1m[date]
#     if select == 2:
#         V_1y = stock_log.rolling(window=250).std().dropna() * np.sqrt(250)
#         return V_1y[date]
#     if select==0:
#         garch11 = arch_model(stock_log, p=1, q=1)
#         res = garch11.fit(update_freq=10)
#         V_garch = res.conditional_volatility * np.sqrt(250)
#         return V_garch[date]
# #######################################################################
# def path(step, price):
#     '''
#     构造价格路径，20000条模拟路径，每条路径 50 个时间步数
#     :param step: 模拟路径个数
#     :param price: 初始价格
#     :return: 模拟路径
#     '''
#     S = np.zeros((step + 1, I))
#     S[0] = price
#     np.random.seed(20000)
#     for t in range(1, step + 1):
#         z = np.random.standard_normal(I)
#         S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
#     return S
# def Judge_redeem(n, S):  # 判断是否赎回，最近30交易日有15个高于1.3
#     '''
#     判断是否赎回
#     :param n: 当前时间节点 从1500到0
#     :param S: 价格路径
#     :return: 是否赎回
#     '''
#     t = S[n - 29:n + 1]
#     ### 赎回情况，最近30天、
#     t2 = np.where(np.sum(t >= RedeemTigger, axis=0) >= 15)[0]
#     return t2
# def Judge_put(n, S):  # 判断是否回售，最近30交易日有15个低于0.7
#     '''
#     判断是否回售
#     :param n: 当前时间节点 从1500到0
#     :param S: 价格路径
#     :return: 是否回售
#     '''
#     t = S[n - 29:n + 1]
#     ### 赎回情况，最近30天
#     t2 = np.where(np.sum(t <= PutTigger, axis=0) >= 15)[0]
#     return t2
# def LSM(S):
#     '''
#     根据构造的价格路径，及设定的参数，LSM模型计算转债价格
#     :param S: 价格路径
#     :return: LSM价值
#     '''
#     ### 最小现金流
#     discountVet = Nt * np.exp(-r * dt * np.array(range(M + 1)))
#     discountVet = discountVet[::-1]
#     ################################################
#     n = M
#     # 构造现金流矩阵
#     CashFlows = np.zeros((M + 1, I))
#     CashFlows[n] = [max(x * Cr, Nt) for x in S[-1]]
#     V = CashFlows[n]  # 当期价值
#
#     # 构造策略矩阵  0：无 1：到期还本 2：自愿转股 3：被迫转股 4：转债回售
#     Strategy = np.zeros((M + 1, I))
#     Strategy[n] = [int(x * Cr > Nt) + 1 for x in S[-1]]
#
#     # R2 = np.zeros((M + 1))   # 这里不需要
#     while n > 125:
#         n -= 1
#         if n % 300 == 0: print(n)
#         price = S[n]
#         ### 赎回情况，大于赎回价格，则必须转股
#         list1 = Judge_redeem(n, S)
#         ### 回售情况，当回售价值小于贴现值则不回售
#         list2 = []
#         judge_put = Judge_put(n, S)
#         if PutValue < discountVet[n]:
#             list2 = judge_put
#         ### 未触发回售，但是转股价值低于贴现值，则不转股，不回售
#         list3 = np.array(list(set(np.where(price * Cr < discountVet[n])[0]) & (set(range(I)) - set(judge_put))))
#
#         ### 需要回归的
#         flag = np.array(list((set(range(I)) - set(list1) - set(list2) - set(list3))))
#         if len(flag) == 0: continue
#         X = price[flag].reshape(-1, 1) * Cr  # 内在价值
#         Y = (V[flag] * np.exp(-r * dt)).reshape(-1, 1)  # 存续价值
#
#         t = price.copy()
#         t[judge_put] = Put  # 内在价值
#         V_in = t[flag].reshape(-1, 1) * Cr
#
#         model = LinearRegression()
#         model.fit(X, Y)
#         # R2[n] = model.score(X, Y)
#         # a = model.intercept_  # 截距
#         # b = model.coef_  # 回归系数
#         # print("最佳拟合线:截距", a, ",回归系数：", b)
#         Y_pred = model.predict(X)
#
#         ### 需要转股，内在价值大于预估存续价值，和强制赎回的
#         flag1 = np.append(flag[np.where(V_in > Y_pred)[0]], list1).astype(int)
#         if len(flag1) > 0:
#             CashFlows[n, flag1] = price[flag1] * Cr
#             CashFlows[n + 1:, flag1] = 0
#             ### 修正策略
#             Strategy[n, flag1] = 2
#             Strategy[n + 1:, flag1] = 0
#             if len(list1) > 0: Strategy[n, list1] = 3
#             t = list(set(flag1) & set(judge_put))
#             if len(t) > 0:
#                 CashFlows[n, t] = PutValue
#                 Strategy[n, t] = 4
#
#         ### 计算n时刻的现值
#         V = np.dot(np.exp(-r * dt * np.array(range(M + 1 - n))), CashFlows[n:])
#     V = np.dot(np.exp(-r * dt * (1 + np.array(range(M + 1 - n)))), CashFlows[n:])
#     V_LSM = V.mean() * np.exp(-r * dt * 125)
#     return V_LSM
#
# #########################  计算76转债的LSM价格
# V_LSM = dict()
# V_BSM = dict()
# V_BSM_pro= dict()
#
# for num in range(76):
#     choose(num)
#     sigma = volatility(2)  # 波动率(收益标准差)，0是garch波动率 1是1月波动率 2是1年波动率
#     ############## LSM 参数
#     S0 = stock[date]  # 股票或指数初始的价格;
#     r = shibor.iloc[np.where(shibor.tradeDate == date)[0], 5].values[0] / 100  # 无风险利率
#     T = 6  # 期权的到期年限(距离到期日时间间隔)
#     M = 250 * T  # 一年考虑250步，一天一步，number of time steps
#     dt = T / M  # 时间间隔
#     I = 2000  # 模拟路径个数
#     N0 = 100  # 面值
#     Nt = file.iloc[num, 9]  # 到期归还本金
#     Cp = file.iloc[num, 5]  # 转股价格
#     Cr = N0 / Cp  # 转股比例
#     RedeemTigger = file.iloc[num, 7] * Cp / 100  # 赎回触发价格
#     Redeem = Cp * 1.0113  # 赎回价格
#     PutTigger = file.iloc[num, 8] * Cp / 100  # 回售触发价格
#     Put = Cp * 1.0113  # 回售价格
#     PutValue = Put * Cr  # 触发回售时，回售价值
#
#     S = path(M,S0)
#
#     V_LSM[num]=LSM(S)
#
#     ###########################  BSM model
#     # 构造现金流矩阵
#     CashFlows = np.zeros((M + 1, I))
#     CashFlows[-1] = [max(x * Cr, Nt) for x in S[-1]]
#     V = CashFlows[-1]
#     V_BSM[num] = V.mean() * np.exp(-r * dt * 1500)
#
#     ######################### 考虑赎回回售  BSM_pro model
#     state = (S > RedeemTigger) + 0     # 若超过赎回，state标记为1
#     state = state - (S < PutTigger)     # 若超过回售，state标记为-1
#     state1 = state.cumsum(axis=0)
#     state2 = np.roll(state1, 30, axis=0)
#     state2[:30] = 0
#     state = state1 - state2               # 当前30天内，满足赎回（回售）的天数
#
#     a, b = np.where((state == 15) | (state == -15))  # 刚好可以赎回（回售）的位置
#     position = pd.DataFrame({'a':a,'b':b}).sort_values(['b', 'a'])
#     position = position.groupby('b').first()        # 找到各个路径第一次满足赎回（回售）的位置
#
#     for y in position.index:
#         x = position.loc[y][0]
#         if state[x, y] < 0:
#             CashFlows[x, y] = PutValue
#         else:
#             CashFlows[x, y] = S[x, y] * Cr
#         CashFlows[-1, y] = 0
#     V = np.dot(np.exp(-r * dt * np.array(range(M + 1))), CashFlows)
#     V_BSM_pro[num] = V.mean()
#
#
# ##### 将模型得到的 LSM  BSM   BSM_pro模型计算的 上市价格 ，写入file文件，并保存
# file.iloc[:,-3]= V_LSM.values()
# file.iloc[:,-2]=V_BSM.values()
# file.iloc[:,-1]=V_BSM_pro.values()
# file.to_excel('结果.xlsx','Sheet1')

##############################################################################################
##############################################################################################
##############################################################################################


import numpy as np
from time import time
import pandas as pd
import math
import datetime
import tushare as ts
from arch import arch_model
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.font_manager import FontProperties
from urllib.request import urlretrieve
import os
import myfunc

# 导入转债数据，缺失值填充，导入无风险利率：1年期银行间拆借利率
file = pd.read_excel(r'e:\bond\正在交易可转债数据.xlsx')
file.iloc[:, 8].fillna(70, inplace=True)
stk_list=file.iloc[:,0].str[:6]

bond = pd.read_csv(r'e:\bond\bond_kzz.csv',encoding='gbk')
bond =bond.set_index('tradeDate')

shibor = pd.read_excel(r'e:\bond\利率数据.xls')        # 这里有6类无风险收益，选取国债30年期
shibor = shibor[1:-2]
shibor.index = shibor.iloc[:,0].astype(str).str[:10]
shibor = shibor.iloc[:,4]
shibor = shibor[(shibor.index>'2015-01-01') & (shibor.index<'2019-01-01')]

datelist = pd.Series(shibor.index)           # 20150101到20180101的交易日

def choose(num):
    '''
    :param num: 76个转债序号
    :return:
    '''
    global stock_no, date, day1,day2,datelist_stk
    stock_no = file.iloc[num, 0][:6]  # 股票代码
    stock_no = str(1 - int(stock_no[0] == '6')) + stock_no
    date = file.iloc[num, 4].date()  # 转债交易日期日期，剩下1250天来算
    date = str(date)

    day1 = datelist[(datelist<'2018-01-01') & (datelist>=date)].__len__()  # 上市日到20180101的天数
    day2 = datelist[datelist >= date].__len__()                           # 上市日到当前20190315的天数

    datelist_stk = datelist[datelist>=max('2018-01-01',date)].tolist()                   # 需要计算的日期

num=1
choose(1)
############################ 计算股票波动率
def volatility(select=0):
    '''
    计算股票的波动率
    :param select: 0代表garch波动率 1代表1月波动率 2代表1年波动率
    :return: 波动率(收益标准差)，
    '''
    global stock
    start_date = '20150101'
    end_date = '20190325'
    url = 'http://quotes.money.163.com/service/chddata.html?code=' + stock_no + '&start=' + start_date + '&end=' + end_date + '&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'
    ############ 股票走势数据存在'e:\\bond'的文件夹下
    filename = 'e:\\bond\\' + stock_no + '.csv'
    if not os.path.exists(filename): urlretrieve(url, filename)
    stock = pd.read_csv(filename, encoding='gbk')
    stock = stock.iloc[:, [0, 3]].set_index('日期').iloc[:, 0]
    stock = stock.sort_index()
    # 剔除停牌的日期
    stock = stock[stock > 0]
    # 简单收益率
    # stock_pct = stock.pct_change().dropna()
    # 对数收益率
    stock_log = stock.apply(lambda x: math.log(x)).diff().dropna()

    ###########循环日期，得到波动率
    datelist = stock_log[date:].index
    if select == 1:
        V_1m = stock_log.rolling(window=20).std().dropna() * np.sqrt(250)
        return V_1m
    if select == 2:
        V_1y = stock_log.rolling(window=250).std().dropna() * np.sqrt(250)
        return V_1y
    if select==0:
        garch11 = arch_model(stock_log, p=1, q=1)
        res = garch11.fit(update_freq=10)
        V_garch = res.conditional_volatility * np.sqrt(250)
        return V_garch
#######################################################################
def path(step, price):
    '''
    构造价格路径，20000条模拟路径，每条路径 50 个时间步数
    :param step: 模拟路径个数
    :param price: 初始价格
    :return: 模拟路径
    '''
    S = np.zeros((step + 1, I))
    S[0] = price
    np.random.seed(2000)
    for t in range(1, step + 1):
        z = np.random.standard_normal(I)
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    return S
def Judge_redeem(n, S):  # 判断是否赎回，最近30交易日有15个高于1.3
    '''
    判断是否赎回
    :param n: 当前时间节点 从1500到0
    :param S: 价格路径
    :return: 是否赎回
    '''
    t = S[n - 29:n + 1]
    ### 赎回情况，最近30天、
    t2 = np.where(np.sum(t >= RedeemTigger, axis=0) >= 15)[0]
    return t2
def Judge_put(n, S):  # 判断是否回售，最近30交易日有15个低于0.7
    '''
    判断是否回售
    :param n: 当前时间节点 从1500到0
    :param S: 价格路径
    :return: 是否回售
    '''
    t = S[n - 29:n + 1]
    ### 赎回情况，最近30天
    t2 = np.where(np.sum(t <= PutTigger, axis=0) >= 15)[0]
    return t2
def LSM_(S):
    '''
    根据构造的价格路径，及设定的参数，LSM模型计算转债价格
    :param S: 价格路径
    :return: LSM价值
    '''
    ### 最小现金流
    discountVet = Nt * np.exp(-r * dt * np.array(range(M + 1)))
    discountVet = discountVet[::-1]
    ################################################
    n = M
    # 构造现金流矩阵
    CashFlows = np.zeros((M + 1, I))
    CashFlows[n] = [max(x * Cr, Nt) for x in S[-1]]
    V = CashFlows[n]  # 当期价值

    # 构造策略矩阵  0：无 1：到期还本 2：自愿转股 3：被迫转股 4：转债回售
    Strategy = np.zeros((M + 1, I))
    Strategy[n] = [int(x * Cr > Nt) + 1 for x in S[-1]]

    # R2 = np.zeros((M + 1))   # 这里不需要
    while n > 0:
        n -= 1
        if n % 300 == 0: print(n)
        price = S[n]
        ### 赎回情况，大于赎回价格，则必须转股
        list1 = Judge_redeem(n, S)
        ### 回售情况，当回售价值小于贴现值则不回售
        list2 = []
        judge_put = Judge_put(n, S)
        if PutValue < discountVet[n]:
            list2 = judge_put
        ### 未触发回售，但是转股价值低于贴现值，则不转股，不回售
        list3 = np.array(list(set(np.where(price * Cr < discountVet[n])[0]) & (set(range(I)) - set(judge_put))))

        ### 需要回归的
        flag = np.array(list((set(range(I)) - set(list1) - set(list2) - set(list3))))
        if len(flag) == 0: continue
        X = price[flag].reshape(-1, 1) * Cr  # 内在价值
        Y = (V[flag] * np.exp(-r * dt)).reshape(-1, 1)  # 存续价值

        t = price.copy()
        t[judge_put] = Put  # 内在价值
        V_in = t[flag].reshape(-1, 1) * Cr

        model = LinearRegression()
        model.fit(X, Y)
        # R2[n] = model.score(X, Y)
        # a = model.intercept_  # 截距
        # b = model.coef_  # 回归系数
        # print("最佳拟合线:截距", a, ",回归系数：", b)
        Y_pred = model.predict(X)

        ### 需要转股，内在价值大于预估存续价值，和强制赎回的
        flag1 = np.append(flag[np.where(V_in > Y_pred)[0]], list1).astype(int)
        if len(flag1) > 0:
            CashFlows[n, flag1] = price[flag1] * Cr
            CashFlows[n + 1:, flag1] = 0
            ### 修正策略
            Strategy[n, flag1] = 2
            Strategy[n + 1:, flag1] = 0
            if len(list1) > 0: Strategy[n, list1] = 3
            t = list(set(flag1) & set(judge_put))
            if len(t) > 0:
                CashFlows[n, t] = PutValue
                Strategy[n, t] = 4

        ### 计算n时刻的现值
        V = np.dot(np.exp(-r * dt * np.array(range(M + 1 - n))), CashFlows[n:])
    V_LSM = np.dot(np.exp(-r * dt * (1 + np.array(range(M + 1 - n)))), CashFlows[n:])
    # V_LSM = V_LSM.mean() * np.exp(-r * dt * 125)
    return V_LSM

#########################  计算76转债的LSM价格
V_LSM = pd.DataFrame(index=datelist[datelist>'2018-01-01'],columns=stk_list)
V_BSM = pd.DataFrame(index=datelist[datelist>'2018-01-01'],columns=stk_list)
V_BSM_pro= pd.DataFrame(index=datelist[datelist>'2018-01-01'],columns=stk_list)

num=0
for num in range(22):  #len(file)
    choose(num)
    sigma_ = volatility(0)  # 波动率(收益标准差)，0是garch波动率 1是1月波动率 2是1年波动率
    # riqi = datelist_stk[0];i=0
    for i,riqi in enumerate(datelist_stk):
        print(stock_no[1:],riqi)
        try:
            ############## LSM 参数
            sigma = sigma_[riqi]
            S0 = stock[riqi]  # 股票或指数初始的价格;
            r = shibor[riqi] / 100  # 无风险利率
            T = 6  # 期权的到期年限(距离到期日时间间隔)
            M = 250 * T - 250//2 - day1 - i# 一年考虑250步，一天一步，number of time steps
            dt = 1 / 250  # 时间间隔
            I = 2000  # 模拟路径个数
            N0 = 100  # 面值
            Nt = file.iloc[num, 9]  # 到期归还本金
            Cp = file.iloc[num, 5]  # 转股价格
            Cr = N0 / Cp  # 转股比例
            RedeemTigger = file.iloc[num, 7] * Cp / 100  # 赎回触发价格
            Redeem = Cp * 1.0113  # 赎回价格
            PutTigger = file.iloc[num, 8] * Cp / 100  # 回售触发价格
            Put = Cp * 1.0113  # 回售价格
            PutValue = Put * Cr  # 触发回售时，回售价值

            S = path(M,S0)
            # V_LSM.ix[riqi,num]=LSM_(S)

            ######################################################  BSM model
            # 构造现金流矩阵
            CashFlows = np.zeros((M + 1, I))
            CashFlows[-1] = [max(x * Cr, Nt) for x in S[-1]]
            V = CashFlows[-1]
            t = V.mean() * np.exp(-r * dt * M)
            V_BSM.ix[riqi, num] = t

            ################################################# 考虑赎回回售  BSM_pro model
            state = (S > RedeemTigger) + 0  # 若超过赎回，state标记为1
            state = state - (S < PutTigger)  # 若超过回售，state标记为-1
            state1 = state.cumsum(axis=0)
            state2 = np.roll(state1, 30, axis=0)
            state2[:30] = 0
            state = state1 - state2  # 当前30天内，满足赎回（回售）的天数

            a, b = np.where((state == 15) | (state == -15))  # 刚好可以赎回（回售）的位置
            position = pd.DataFrame({'a': a, 'b': b}).sort_values(['b', 'a'])
            position = position.groupby('b').first()  # 找到各个路径第一次满足赎回（回售）的位置

            for y in position.index:
                x = position.loc[y][0]
                if state[x, y] < 0:
                    CashFlows[x, y] = PutValue
                else:
                    CashFlows[x, y] = S[x, y] * Cr
                CashFlows[-1, y] = 0
            V = np.dot(np.exp(-r * dt * np.array(range(M + 1))), CashFlows)
            V_BSM_pro.ix[riqi, num] = V.mean()

        except:
            print('未找到日期')
V_LSM.to_csv('V_LSM.csv')
V_BSM.to_csv('V_BSM.csv')
V_BSM_pro.to_csv('V_BSM_pro.csv')


# V_LSM=pd.read_csv('e:\\bond\\LSM.csv',encoding='gbk')
# V_LSM = V_LSM.iloc[:,:23].set_index('指标名称')
# r = shibor.loc[V_LSM.index]
# V_LSM =V_LSM.apply(lambda x:x/np.exp(-r * dt * 125),axis=1)
# V_LSM.to_csv('e:\\bond\\V_LSM.csv')

V_LSM=pd.read_csv('e:\\bond\\V_LSM.csv').iloc[:,:23].set_index('指标名称')
V_BSM=pd.read_csv('e:\\bond\\V_BSM.csv').iloc[:,:23].set_index('指标名称')
V_BSM_pro=pd.read_csv('e:\\bond\\V_BSM_pro.csv').iloc[:,:23].set_index('指标名称')


def pic(model = 'LSM'):
    from matplotlib import ticker as mticker
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=12)
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    lns1 = ax1.plot(V_LSM_mean_.index, V_LSM_mean_.iloc[:,0], '-r')
    lns2 = ax1.plot(V_LSM_mean_.index, V_LSM_mean_.iloc[:,1], '-g')
    lns3 = ax2.bar(V_LSM_mean_.index, V_LSM_mean_.iloc[:,2], align='center', color='b')
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax1.legend(lns1+lns2, [model+' Price','True Price'], loc=2, fontsize=12)
    ax2.legend(lns3, ['error rate'], fontsize=12)
    ax2.set_ylim(-0.05,0.2)
    ax1.set_xlim(-10,250)
    ax1.set_ylabel(u'价格走势', fontproperties=font, fontsize=16)
    ax2.set_ylabel(u'误差率', fontproperties=font, fontsize=16)
    ax1.set_xlabel(u'日期', fontproperties=font, fontsize=16)
    ax2.set_yticklabels([str(x*100//1)+'0%' for x in ax2.get_yticks()], fontproperties=font, fontsize=14)
    ax1.set_title(u"2018年度可转债"+model+"价格和真实价格均值", fontproperties=font, fontsize=16)
    ax1.grid()
    plt.savefig('e:\\bond\\'+ model +'.png')
    plt.show()

bond_code = file.iloc[:,2].str[:6].tolist()[:22]
bond = bond[bond_code]
bond = bond[bond.index<'2019-01-01']
bond_mean = bond.mean(axis=1)
bond_mean.name='bond'

V_LSM.columns = bond_code
V_LSM_error = (V_LSM-bond)/bond
V_LSM_error=V_LSM_error.dropna(how='all')
V_LSM_mean = V_LSM.mean(axis=1).dropna(how='all')
V_LSM_error_mean=V_LSM_error.mean(axis=1).dropna(how='all')
V_LSM_mean_ = V_LSM_mean.to_frame(name='V_LSM').join(bond_mean).join(V_LSM_error_mean.rename('error'))
pic('LSM')
myfunc.add_sheet(V_LSM_mean_,'Sheet1','e:\\bond\\price')


V_BSM.columns = bond_code
V_BSM_error = (V_BSM-bond)/bond
V_BSM_error=V_BSM_error.dropna(how='all')
V_LSM_mean = V_BSM.mean(axis=1).dropna(how='all')
V_LSM_error_mean=V_BSM_error.mean(axis=1).dropna(how='all')
V_LSM_mean_ = V_LSM_mean.to_frame(name='V_LSM').join(bond_mean).join(V_LSM_error_mean.rename('error'))
pic('BSM')
myfunc.add_sheet(V_LSM_mean_,'Sheet1','e:\\bond\\price')


V_BSM_pro.columns = bond_code
V_BSM_pro_error = (V_BSM_pro-bond)/bond
V_BSM_pro_error=V_BSM_pro_error.dropna(how='all')
V_LSM_mean = V_BSM_pro.mean(axis=1).dropna(how='all')
V_LSM_error_mean=V_BSM_pro_error.mean(axis=1).dropna(how='all')
V_LSM_mean_ = V_LSM_mean.to_frame(name='V_LSM').join(bond_mean).join(V_LSM_error_mean.rename('error'))
pic('BSM-change')
myfunc.add_sheet(V_LSM_mean_,'Sheet1','e:\\bond\\price')

##### 将模型得到的 LSM  BSM   BSM_pro模型计算的 上市价格 ，写入file文件，并保存
file.iloc[:,-3]= V_LSM.values()
file.iloc[:,-2]=V_BSM.values()
file.iloc[:,-1]=V_BSM_pro.values()
file.to_excel('结果.xlsx','Sheet1')
