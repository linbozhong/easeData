# coding:utf-8

import math
import requests
import csv
import numpy as np
import pandas as pd
import os
import pymongo
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
import seaborn as sns
import tushare as ts

from copy import copy
from collections import OrderedDict
from datetime import datetime, timedelta, time
from dateutil.relativedelta import relativedelta, FR
from scipy import stats
from jqdatasdk import opt, query
from pyecharts import Line, Kline, Bar
from pyecharts import Page, Overlap

from database import MongoDbConnector
from base import LoggerWrapper
from collector import JQDataCollector
from functions import (strToDate, dateToStr, getTestPath, getDataDir, roundFloat, getLastFriday, saveCsv)
from const import *
from text import *

sns.set()
QVIX_URL = 'http://1.optbbs.com/d/csv/d/vixk.csv'


class MonteCarlo(LoggerWrapper):
    def __init__(self, underlyingAnalyzer):
        super(MonteCarlo, self).__init__()
        self.underlyingAnalyzer = underlyingAnalyzer
        self.price = None
        self.simCount = None
        self.expiredDays = None
        self.simCache = None

    def getPrice(self, start=None):
        """获取价格序列并生成对数收益率数据"""
        if self.price is None:
            price = self.underlyingAnalyzer.getPrice()
            if start:
                start = strToDate(start)
                price = price[start:]
                price = copy(price)
            price['returns'] = price.close.pct_change()
            price['returns'] = np.log(price['returns'] + 1)
            self.price = price
        return self.price

    def monteCarloSim(self, simCount, expiredDays):
        """
        进行蒙特卡洛模拟，并保存计算结果
        :param simCount: 模拟次数
        :param expiredDays: 剩余到期日
        :return:
        """
        if self.expiredDays == expiredDays and self.simCount >= simCount and self.simCache:
            print('Data from cache.')
            return self.simCache

        price = self.getPrice()
        lastPrice = price['close'][-1]
        dailyVol = price['returns'].std()
        print(dailyVol * 15.87)

        randDailyReturn = np.random.normal(0, dailyVol, size=(simCount, expiredDays)) + 1
        randDailyReturn = np.insert(randDailyReturn, 0, lastPrice, axis=1)
        randPrice = np.multiply.accumulate(randDailyReturn, axis=1)
        df = pd.DataFrame(randPrice)

        self.simCount = simCount
        self.expiredDays = expiredDays
        self.simCache = df
        return df

    def getExpiredDays(self, expiredDate):
        """输入最后交易日，获取剩余交易天数"""
        d1 = datetime.now().date()
        d2 = strToDate(expiredDate).date()
        cal = self.underlyingAnalyzer.collector.getTradeDayCal()
        days = cal[(cal > d1) & (cal <= d2)]
        return len(days)

    def getProbability(self, up, down, expiredDate, simCount=100000):
        """计算合约存续期间维持上下行权价的概率"""
        expiredDays = self.getExpiredDays(expiredDate)
        df = self.monteCarloSim(simCount, expiredDays)
        selDf = df[(df > up) | (df < down)]
        selDf.dropna(how='all', inplace=True)
        return 1 - (len(selDf) / float(simCount))


class CorrelationAnalyzer(LoggerWrapper):
    """
    期货各品种相关系数分析
    """
    DB_NAME_MAP = {
        DAILY: VNPY_DAILY_DB_NAME,
        BAR: VNPY_MINUTE_DB_NAME,
    }

    def __init__(self):
        super(CorrelationAnalyzer, self).__init__()
        self.freq = None
        self.num = None
        self.period = 1
        self.threshold = 0.5
        self.endDay = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

        self.tradeCal = None
        self.targetCollection = None
        self.exclude = None
        self.cffex = ['IF', 'IC', 'IH', 'T', 'TF']
        self.dateRange = None

        self.dbConnector = MongoDbConnector()
        self.db = None

    @staticmethod
    def getCorrelation(df):
        """
        从价格序列计算相关系数。
        ----------------------------------------------------------------------------------------------------------------
        :param df: DataFrame
                价格序列
        :return:
        """
        df = df.pct_change()
        # df.dropna(inplace=True) # 波动小的品种在1分钟数据上会出现很多0的数据，从而导致整个表被删除。
        df = df.corr()
        # df.to_csv(getTestPath('dailyCorr.csv'))
        return df

    def setDb(self):
        dbName = self.DB_NAME_MAP[self.freq]
        self.db = self.dbConnector.connect()[dbName]

    def setFreq(self, freq):
        """
        设定要分析的时间周期。
        ‘daily’表示使用日线数据，分析月度的相关性；‘bar’表示使用分钟线数据，分析日度的相关性。
        ----------------------------------------------------------------------------------------------------------------
        :param freq: ‘daily’或‘bar’
        :return:
        """
        self.freq = freq
        self.setDb()
        self.setTargetCollection()

    def setNum(self, n):
        """
        设定期数。
        ----------------------------------------------------------------------------------------------------------------
        :param n: int
                期数，从结束日期往前推算
        :return:
        """
        self.num = n

    def setPeriod(self, n):
        """
        设定每期时间间隔，默认1。
        ----------------------------------------------------------------------------------------------------------------
        :param n: int
                时间间隔
        :return:
        """
        self.period = n

    def setThreshold(self, threshold):
        """
        设定相关系数阈值，默认0.5。
        ----------------------------------------------------------------------------------------------------------------
        :param threshold: int
                阈值
        :return:
        """
        self.threshold = threshold

    def setEndDay(self, date):
        """
        设定结束日期，默认今日0时0分。
        ----------------------------------------------------------------------------------------------------------------
        :param date: string
                往前倒推获取数据的基准日
        :return:
        """
        self.endDay = strToDate(date)

    def setTargetCollection(self, cols=None):
        """
        设定要做相关性分析的期货品种集合。
        ----------------------------------------------------------------------------------------------------------------
        :param cols:
                期货品种列表
        :return:
        """
        if cols is None:
            cols = [item for item in self.db.collection_names() if '9999' in item]
        self.targetCollection = cols

    def setExclude(self, varieties=None):
        """
        设定要不做分析的期货品种。如新上市的、数据量不够的。
        ----------------------------------------------------------------------------------------------------------------
        :param varieties: iterable
                期货品种列表
        :return:
        """
        self.exclude = varieties

    def isTradeDay(self, date):
        """
        通过tushare模块交易日功能判断日期是否为交易日。
        ----------------------------------------------------------------------------------------------------------------
        :param date: string
                日期
        :return: bool
        """
        if self.tradeCal is None:
            self.tradeCal = ts.trade_cal()
            self.tradeCal.set_index('calendarDate', inplace=True)
            self.tradeCal = self.tradeCal['isOpen']
        return bool(self.tradeCal[date])

    def getDateRange(self):
        """
        通过期数和间隔（月），来获取一段日期区间序列。
        ----------------------------------------------------------------------------------------------------------------
        :return: list[tuple(dt)]
                数据范例：
                [
                    (datetime.datetime(2017, 12, 1, 0, 0), datetime.datetime(2018, 1, 1, 0, 0)),
                    ...
                ]
        """
        if self.freq == BAR:
            end = self.endDay.replace(hour=15, minute=1)
            timeDelta = relativedelta(days=1)  # period在bar模式下无效
            start = end.replace(hour=9)
        else:
            end = self.endDay.replace(day=1)
            timeDelta = relativedelta(months=self.period)
            start = end - timeDelta

        dRange = []
        for i in range(self.num):
            # bar模式要判断日期是否为交易日，只有交易日才添加到列表。
            if self.freq == BAR:
                while not self.isTradeDay(dateToStr(end)):
                    end = end - timeDelta
                start = end.replace(hour=9)

            dRange.append((start, end))
            start = start - timeDelta
            end = end - timeDelta
        dRange.reverse()
        return dRange

    def getPrice(self, start, end, collections=None, exclude=None):
        """
        获取价格序列。
        ----------------------------------------------------------------------------------------------------------------
        :param start: datetime
                开始日期
        :param end: datetime
                结束日期
        :param collections: iterable
                要获取的期货品种列表
        :param exclude: iterable
                要排除的期货品种列表
        :return: DataFrame
        """
        filename = '{}-{}.csv'.format(start.strftime('%Y%m%d %H%M%S'), end.strftime('%Y%m%d %H%M%S'))
        self.info(u'{} Getting price: {}'.format(self.freq, filename))

        if collections is None:
            collections = self.targetCollection
        if exclude is None:
            exclude = self.exclude

        allPrices = {}
        for colName in collections:
            variety = colName.replace('9999', '')
            if (exclude is not None) and (variety in exclude):
                continue
            col = self.db[colName]
            flt = {'datetime': {'$gte': start, '$lt': end}}
            projection = {'_id': False, 'datetime': True, 'close': True}
            cursor = col.find(flt, projection).sort('datetime', pymongo.ASCENDING)
            closeDict = {rec['datetime']: rec['close'] for rec in cursor}
            closeSeries = pd.Series(closeDict)
            allPrices[variety] = closeSeries
        df = pd.DataFrame(allPrices)
        if self.freq == DAILY:
            df.dropna(axis=1, inplace=True)  # 剔除日线数据不完整的合约，如新上市合约等
        # df.to_csv(os.path.join(getTestPath(filename)))
        return df

    def getCorrelationArray(self):
        """
        获取多个时间段的相关系数表格合集。
        ----------------------------------------------------------------------------------------------------------------
        :return: DataFrame
        """
        self.dateRange = self.getDateRange()
        dfList = []
        print(self.dateRange)
        for start, end in self.dateRange:
            priceDf = self.getPrice(start, end)
            corrDf = self.getCorrelation(priceDf)
            dfList.append(corrDf)
        dfs = pd.concat(dfList)
        dfs.to_csv(getTestPath('{}_corrArrayDf.csv'.format(self.freq)))
        return dfs

    def analyzeCorrelation(self, df):
        """
        从相关系数表格找出存在相关系数的品种。
        ----------------------------------------------------------------------------------------------------------------
        :param df: DataFrame
                相关系数表格
        :return: dict
            key: string, 期货品种代码
            value: list[tuple(string, float)]
            数据范例：
            {
                'rb':[('hc': 0.687), ('j', 0.654)],
                ...
            }
        """
        d = dict()
        for colName, series in df.iteritems():
            l = []
            for idxName, corrValue in series.iteritems():
                if (corrValue >= self.threshold or corrValue <= -self.threshold) and corrValue != 1:
                    t = idxName, corrValue
                    l.append(t)
            d[colName] = l
        return d

    def plotCorrelationArray(self, df):
        """
        基于相关系数合集表格绘制相关系数走势图。
        ----------------------------------------------------------------------------------------------------------------
        :param df: DataFrame
                多时间的相关系数合集
        :return:
        """
        dateRange = [dateToStr(item[0]) for item in self.dateRange]  # 仅保留开始日期，用作x轴标签

        grouped = df.groupby(df.index)
        groupedDict = dict(list(grouped))

        # 通过相关系数均值来筛选出有相关的品种，避免计算无用数据。
        dfMean = grouped.mean()
        corrDict = self.analyzeCorrelation(dfMean)
        for key, corrList in corrDict.iteritems():
            if not corrList:  # 排除没有与其他任何品种有相关关系的品种
                continue
            selList = [item[0] for item in corrList]  # 得到有相关关系品种的名称列表
            varietyDf = groupedDict[key]
            varietyDf['date'] = dateRange
            varietyDf.set_index('date', inplace=True)
            selDf = varietyDf[selList]  # 根据名称筛选df

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(key)
            ax.set_ylabel('correlation')
            ax.plot([0.5 for i in range(len(selDf))], linestyle='dashed', color='k')  # 绘制0.5的参考线
            selDf.plot(ax=ax, ylim=[-0.5, 1])  # 0.5以上的负相关基本不存在。
            filename = '{}_{}.png'.format(self.freq, key)
            fig.savefig(getTestPath(filename))


class PositionDiffPlotter(LoggerWrapper):
    """
    期权品种持仓量走势图
    """

    def __init__(self):
        super(PositionDiffPlotter, self).__init__()
        self.jqsdk = JQDataCollector()
        self.tradingContractInfo = None
        self.mainContract = None
        self.comContract = None
        self.contractName = None
        self.dailyPriceDict = OrderedDict()

        self.isExcludeAdjusted = True
        self.underlyingSymbol = '510050.XSHG'
        self.exchange = 'XSHG'
        self.date = datetime(2018, 12, 26)
        self.queryMonth = '1901'
        self.queryExercisePrice = [
            2.1,
            2.15,
            2.2,
            2.25,
            2.3,
            2.35,
            2.4,
            2.45,
            2.5
        ]
        self.compExercisePrice = [
            2.7,
            2.75,
            2.8,
        ]

        self.red = '#c91818'
        self.blue = '#34a9ad'

    @staticmethod
    def calcPositionDiff(dailyDf):
        """
        从日行情DataFrame计算仓差。
        :param dailyDf:
        :return:
        """
        # df = dailyDf.set_index('date')
        df = dailyDf
        df['prePosition'] = df['position'].shift(1)
        df['positionDiff'] = df['position'] - df['prePosition']
        df['positionDiff'].fillna(0, inplace=True)
        return df['positionDiff'].map(int)

    def setQueryMonth(self, month):
        """
        设置月份。
        :param month:
        :return:
        """
        self.queryMonth = month

    def cleanCache(self):
        """
        清空已经缓存的数据。
        :return:
        """
        self.tradingContractInfo = None
        self.mainContract = None
        self.comContract = None
        self.contractName = None
        self.dailyPriceDict.clear()

    def getTradingContract(self):
        """
        获取某个月份的交易合约。
        :return: pd.DataFrame
        """
        if self.tradingContractInfo is None:
            filename = u'{}_{}.csv'.format(OPTION, BASIC)
            path = os.path.join(self.jqsdk.getBasicPath(OPTION), filename)
            if not os.path.exists(path):
                db = opt.OPT_CONTRACT_INFO
                q = query(db).filter(db.exchange_code == self.exchange,
                                     db.underlying_symbol == self.underlyingSymbol).order_by(
                    db.id.desc())  # api有数量限制，降序优先获取最新数据
                df = self.jqsdk.run_query(q)
            else:
                df = pd.read_csv(path, index_col=0)
                mask = (df.exchange_code == self.exchange) & (df.underlying_symbol == self.underlyingSymbol)
                df = df[mask]

            # 筛选符合月份的数据
            postfix = 'M' if self.isExcludeAdjusted else ''
            filterWord = self.queryMonth + postfix
            filterFunc = lambda x: filterWord in x
            monthMask = df.trading_code.map(filterFunc)
            df = df[monthMask]

            self.tradingContractInfo = df
            # print(df.name.values[0], type(df.name.values[0]))
        return self.tradingContractInfo

    def getContractName(self, contractCode):
        """
        获取期权合约的中文名。
        :param contractCode: string
                期权代码
        :return: pd.Series
        """
        if self.contractName is None:
            df = self.getTradingContract()
            df = df.set_index('code')
            self.contractName = df['name']
        return self.contractName[contractCode]

    def getDailyPrice(self, contract):
        """
        获取期权合约的日行情数据。
        :param contract: string
                期权代码。
        :return: pd.DataFrame
        """
        db = opt.OPT_DAILY_PRICE
        q = query(db).filter(db.code == contract)
        df = self.jqsdk.run_query(q)
        df = df.set_index('date')
        return df

    def getMainContract(self):
        """
        仅显示设定的部分合约。
        :return: pd.DataFrame
        """
        if self.mainContract is None:
            df = self.getTradingContract()
            mask = df.exercise_price.isin(self.queryExercisePrice)
            df = df[mask]
            df = df.sort_values(by='name')
            self.mainContract = df
        return self.mainContract

    def getCompContract(self):
        """
        选出要对比的合约。
        :return: pd.DataFrame
        """
        if self.comContract is None:
            df = self.getTradingContract()
            mask = df.exercise_price.isin(self.compExercisePrice)
            self.comContract = df[mask]
        return self.comContract

    def getAllContractDailyPrice(self):
        """
        获取该月份的所有合约的日线数据，并保存为字典。
        :return: OrderDict()
                key: string
                    期权合约编码
                value: pd.DataFrame
                    期权日线价格序列

        """
        if not self.dailyPriceDict:
            contracts = self.getTradingContract().code
            for contract in contracts:
                self.dailyPriceDict[contract] = self.getDailyPrice(contract)
        return self.dailyPriceDict

    def getGoupedCode(self):
        """
        获取按行权价分组的期权代码, 用于顺序排版。
        :return: OrderedDict
                key: float
                    期权行权价
                value: list
                    期权合约编码列表

        """
        df = self.getTradingContract()
        df = df.sort_values(by='exercise_price')  # 按行权价排序
        grouped = df.groupby('exercise_price')
        groupCode = OrderedDict()
        for exercisePrice, df in grouped:
            df = df.sort_values(by='trading_code')  # 按沽购名称排序
            groupCode[exercisePrice] = df['code'].tolist()
        return groupCode

    def getPosition(self):
        """
        获取持仓量数据。
        :return: pd.DataFrame
                index: date 日期
                columns: string 期权合约编码-持仓量series
        """
        price = self.getAllContractDailyPrice()
        posDict = {code: priceDf['position'] for code, priceDf in price.items()}
        # posDict = dict()
        # for code, priceDf in price.items():
        #     posDict[code] = priceDf['position']
        df = pd.DataFrame(posDict)
        df.fillna(0, inplace=True)
        return df

    def getPositionDiff(self):
        """
        获取所有的仓差数据。
        :return: pd.DataFrame
                index: date 日期
                columns: string 期权合约编码-仓差series
        """
        price = self.getAllContractDailyPrice()
        posDiffDict = dict()
        for code, priceDf in price.items():
            posDiffDict[code] = self.calcPositionDiff(priceDf)
        df = pd.DataFrame(posDiffDict)
        df.fillna(0, inplace=True)
        return df

    def nameToColor(self, name):
        """
        通过期权名称获取要使用的颜色。
        :return:
        """
        buy = u'购'

        if not isinstance(name, unicode):
            name = name.decode('utf-8')
        if buy in name:
            color = self.red
        else:
            color = self.blue
        return color

    def plotData(self, method):
        """
        绘制并输出数据走势图。
        :type method: bound method of Class
                获取某个数据的方法
        :return:
        """
        width = 1500

        titleZh = {
            'Position': u'持仓量',
            'PositionDiff': u'仓差'
        }

        lineDisplaySetting = {
            'is_datazoom_show': True,
            'datazoom_type': 'both',
            'datazoom_range': [0, 100],
            'line_width': 2
        }

        divideList = ['  ', 'margin:50px 0']

        funcName = method.im_func.func_name
        displayName = funcName.replace('get', '')

        df = method()
        groupCode = self.getGoupedCode()
        # df = pd.read_csv(getTestPath('posDiff.csv'), index_col=0)
        xtickLabels = df.index.tolist()

        page = Page()

        # 对比组图
        multiLine = Line(u'期权{}对比走势图'.format(titleZh.get(displayName)), width=width)
        for price in self.compExercisePrice:
            codeList = groupCode[price]
            for code in codeList:
                multiLine.add(self.getContractName(code), xtickLabels, df[code].values.tolist(), **lineDisplaySetting)
        page.add(multiLine)

        # 行权价组图
        lineList = []
        for exercisePrice, codeList in groupCode.items():
            line = Line(u'行权价{}{}走势'.format(str(int(exercisePrice * 1000)), titleZh.get(displayName)), width=width,
                        extra_html_text_label=divideList)
            for code in codeList:
                name = self.getContractName(code)
                line.add(name, xtickLabels, df[code].values.tolist(), **lineDisplaySetting)
            lineList.append(line)
        for line in lineList:
            page.add(line)

        htmlName = '{}{}.html'.format(displayName.lower(), self.queryMonth)
        outputDir = self.jqsdk.getResearchPath(OPTION, 'dailytask')
        page.render(os.path.join(outputDir, htmlName))


class SellBuyRatioPlotter(LoggerWrapper):
    """
    期权沽购比走势图
    """

    def __init__(self):
        super(SellBuyRatioPlotter, self).__init__()
        self.jqsdk = JQDataCollector()

        self.contractInfo = None
        self.contractTypeDict = None
        self.tradingCodeDict = None
        self.lastTradeDateDict = None
        self.strikePriceGroupedContractDict = None

        self.underlyingSymbol = '510050.XSHG'

        self.isOnlyNearby = True

        # 绘制平值期权价和图相关属性
        self.atmCombinePrice = None
        self.atmStart = None
        self.atmEnd = None

        # 绘制50etf成交量占比相关属性
        self.idxStart = '2012-01-01'
        self.etfIdx = '000016.XSHG'
        self.shIdx = '000001.XSHG'
        self.szIdx = '399106.XSHE'

    def getContractInfo(self):
        """
        获取期权合约的基础信息，并做缓存。
        :return: pd.DataFrame
        """
        if self.contractInfo is None:
            filename = u'{}_{}.csv'.format(OPTION, BASIC)
            path = os.path.join(self.jqsdk.getBasicPath(OPTION), filename)
            df = pd.read_csv(path, index_col=0)
            mask = (df.underlying_symbol == self.underlyingSymbol)
            df = df[mask]
            df = df.set_index('code')
            self.contractInfo = df
        return self.contractInfo

    def getTradingCode(self, code):
        """
        获取合约编码-合约交易代码的映射，并缓存。
        :param code: string
                合约编码
        :return: string
                合约交易代码
        """
        if self.tradingCodeDict is None:
            df = self.getContractInfo()
            self.tradingCodeDict = df['trading_code'].to_dict()
        return self.tradingCodeDict.get(code, None)

    def getLastTradeDate(self, code):
        """
        获取合约编码-合约最后交易日的映射，并缓存。
        :param code: string
                合约编码
        :return: string
                合约最后交易日
        """
        if self.lastTradeDateDict is None:
            df = self.getContractInfo()
            self.lastTradeDateDict = df['last_trade_date'].to_dict()
        return self.lastTradeDateDict.get(code, None)

    def getContractType(self, code):
        """
        获取合约编码-合约看涨看跌类型的映射，并缓存
        :param code: string
                合约编码
        :return: string
                看涨看跌类型简码
        """
        if self.contractTypeDict is None:
            df = self.getContractInfo()
            # df = df.set_index('code')
            self.contractTypeDict = df['contract_type'].to_dict()
        return self.contractTypeDict.get(code, None)

    def getNearbyContract(self, fp, keepCurrent=False):
        """
        从每日期权价格表中筛选出当月合约的数据，如果距离到期日不足7日，则选出下月的数据。
        :param fp: filePath
                某个交易日的所有期权合约日行情数据
        :param keepCurrent: Bool
                是否使用当月合约（不切换到下月）
        :return: pd.DataFrame
        """
        df = pd.read_csv(fp, index_col=0)

        df['last_trade_date'] = df['code'].map(self.getLastTradeDate)
        df['trading_code'] = df['code'].map(self.getTradingCode)
        df = df[df.trading_code.notnull()]  # 排除非标的的数据

        getMonth = lambda tradingCode: int(tradingCode[7: 11])  # 50etf期权, 8-12位表示合约月份
        df['month'] = df['trading_code'].map(getMonth)

        monthList = df['month'].drop_duplicates()
        monthList = monthList.tolist()
        monthList.sort()

        df_current = df[df.month == monthList[0]]
        date = strToDate(df_current['date'].iloc[0])
        lastDate = strToDate(df_current['last_trade_date'].iloc[0])

        if lastDate - date > timedelta(days=7) or keepCurrent is True:
            resDf = df_current
        else:
            df_next = df[df.month == monthList[1]]
            resDf = df_next

        return resDf

    def getEtfMarketRatio(self):
        """
        获取50etf指数、上证指数、深证综指的成交量和成交额，并计算50etf占的比例。
        :return: pd.DataFrame
        """
        start = self.idxStart
        end = self.jqsdk.today
        toPercent = lambda x: round(x * 100, 2)

        # 收集数据
        d = {}
        volList = []
        turnoverList = []
        for idx in [self.etfIdx, self.shIdx, self.szIdx]:
            df = self.jqsdk.get_price(idx, start_date=start, end_date=end)
            vol = df['volume']
            vol.name = idx
            volList.append(vol)
            turnover = df['money']
            turnover.name = idx
            turnoverList.append(turnover)
        d['volumeRatio'] = volList
        d['turnoverRatio'] = turnoverList

        # 处理数据
        postDataLst = []
        for name, lst in d.iteritems():
            df = pd.concat(lst, axis=1)
            df[name] = df[self.etfIdx] / (df[self.shIdx] + df[self.szIdx])
            df[name] = df[name].map(toPercent)
            postDataLst.append(df[name])

        return pd.concat(postDataLst, axis=1)

    def plotEtfMarketRatio(self):
        """
        绘制50etf占两市成交量比例走势图。
        :return:
        """
        nameDict = {'turnoverRatio': u'成交额占比',
                    'volumeRatio': u'成交量占比',
                    }

        df = self.getEtfMarketRatio()
        df.index = df.index.map(dateToStr)
        # df = pd.read_csv(getTestPath('etfRatio.csv'), index_col=0)
        idx = df.index.tolist()
        overlap = Overlap(width=1500, height=600)
        line = Line(u'50ETF成交占两市比例', is_animation=False)
        for colName, series in df.iteritems():
            value = series.tolist()
            line.add(nameDict[colName], idx, value,
                     is_datazoom_show=True, datazoom_type='both', datazoom_range=[80, 100],
                     tooltip_trigger='axis', tooltip_axispointer_type='cross',
                     is_toolbox_show=False,
                     is_symbol_show=False,
                     yaxis_formatter='%'
                     )
            overlap.add(line)
        htmlName = 'etfMarketRatio.html'
        outputDir = self.jqsdk.getResearchPath(OPTION, 'dailytask')
        overlap.render(os.path.join(outputDir, htmlName))

    def getRecentDays(self, nDays=7):
        """
        取最近n个交易日的起止日期。
        :param nDays:
        :return: tuple(string, string)
        """
        # 获取交易日
        tradeDays = self.jqsdk.getTradeDayCal()

        # 获取截至日期
        today = self.jqsdk.today
        if today.date() in tradeDays:
            end = self.jqsdk.getPreTradeDay(today)
        else:
            latestTradeDay = self.jqsdk.getPreTradeDay(today)
            latestTradeDayDt = datetime(latestTradeDay.year, latestTradeDay.month, latestTradeDay.day)
            end = self.jqsdk.getPreTradeDay(latestTradeDayDt)

        preTradeDays = tradeDays[tradeDays <= end]
        dateRange = preTradeDays[-nDays:]
        return dateToStr(dateRange[0]), dateToStr(dateRange[-1])

    def getFpsByDateRange(self, start, end):
        """
        获取某个日期区间的期权日行情文件路径列表
        :param start: string
        :param end: string
        :return: list(fp)
        """
        tradeDays = self.jqsdk.getTradeDayCal()
        start = strToDate(start).date()
        end = strToDate(end).date()
        dateRange = tradeDays[(tradeDays >= start) & (tradeDays <= end)]

        fnames = ['option_daily_{}.csv'.format(dateToStr(date)) for date in dateRange]
        fps = [os.path.join(self.jqsdk.getPricePath('option', 'daily'), filename) for filename in fnames]
        return fps

    def setAtmStart(self, date):
        self.atmStart = date

    def setAtmEnd(self, date):
        self.atmEnd = date

    def getAtmAlphaByDate(self, fp):
        """
        获取当日平值期权沽、购的alpha（theta/gamma)
        :param fp:
        :return: tuple
            eg. ('2019-01-03', {'C': 0.2311, 'P': 0.2145})
        """
        self.info(u'获取单日平值期权Alpha:{}'.format(os.path.basename(fp)))

        df = self.getAtmContract(fp)
        df['contract_type'] = df['trading_code'].map(lambda x: 'C' if 'C' in x else 'P')
        date = strToDate(df['date'].iloc[0])

        alphaDict = {}
        for idx, series in df.iterrows():
            code = series['code']
            contractType = series['contract_type']
            db = opt.OPT_RISK_INDICATOR
            q = query(db).filter(db.code == code, db.date == date)
            df = self.jqsdk.run_query(q)
            print(df)
            alphaDict[contractType] = roundFloat(abs(df.iloc[0]['theta'] / df.iloc[0]['gamma']), 4)

        return date, alphaDict

    def getAtmAlphaByRange(self, start, end):
        """
        取某个日期区间的平值期权组的alpha数据。
        :param start: string
        :param end: string
        :return: pd.DataFrame
        """
        fps = self.getFpsByDateRange(start, end)
        resList = [self.getAtmAlphaByDate(fp) for fp in fps]

        data = {date: [alphaDict['C'], alphaDict['P']] for (date, alphaDict) in resList}
        df = pd.DataFrame.from_dict(data=data, orient='index', columns=['Call', 'Put'])
        df.sort_index(inplace=True)
        return df

    def getAtmAlpha(self):
        """
        获取从2015-02-09以来的平值期权alpha数据。
        :return: pd.DataFrame
        """
        start = '2015-02-09'
        filename = 'atm_alpha.csv'
        path = os.path.join(self.jqsdk.getPricePath(OPTION, STUDY_DAILY), filename)

        today = self.jqsdk.today
        if not os.path.exists(path):
            df = self.getAtmAlphaByRange(start=start, end=dateToStr(today))
            df.to_csv(path, encoding='utf-8-sig')
        else:
            df = pd.read_csv(path, index_col=0)
            lastDay = strToDate(df.index.values[-1])
            newStart = lastDay + timedelta(days=1)
            if newStart > today - timedelta(days=1):
                self.info(u'{}'.format(FILE_IS_NEWEST))
            else:
                df_new = self.getAtmAlphaByRange(start=dateToStr(newStart), end=dateToStr(today - timedelta(days=1)))
                df_new.index = df_new.index.map(dateToStr)
                df = pd.concat([df, df_new])
                df.to_csv(path, encoding='utf-8-sig')

        # end = df.index[-1]
        # underlyingPrice = self.jqsdk.get_price(self.underlyingSymbol, start_date=start, end_date=end)
        # underlyingPrice.index = underlyingPrice.index.map(dateToStr)
        # close = underlyingPrice['close']
        # df[self.underlyingSymbol] = close

        qvixPlotter = QvixPlotter()
        qvixDf = qvixPlotter.getCsvData()
        df['qvix'] = qvixDf['close']
        return df

    def plotAtmAlpha(self):
        """
        绘制平值期权alpha走势图。
        :return:
        """
        width = 1500
        height = 600
        displayItem = ['Call', 'Put']

        nameDict = {
            'Call': u'平值认购',
            'Put': u'平值认沽',
            'qvix': u'Qvix波动率'
        }

        zoomDict = {
            'is_datazoom_show': True,
            'datazoom_type': 'both',
            'line_width': 2,
            'yaxis_name_size': 14,
            'yaxis_name_gap': 35,
            'tooltip_trigger': 'axis',
            'tooltip_axispointer_type': 'cross'
        }

        df = self.getAtmAlpha()
        xtickLabels = df.index.tolist()

        page = Page()
        overlap = Overlap(width=width, height=height)

        # 对比组图
        multiLine = Line(u'平值期权alpha走势图')
        etfLine = Line(u'波动率指数与平值期权Alpha')
        for name, series in df.iteritems():
            if name == 'qvix':
                etfLine.add(nameDict[name], xtickLabels, series.values.tolist(), yaxis_min=10, yaxis_max=50,
                            yaxis_force_interval=5, yaxis_name=u'Qvix指数', **zoomDict)
            elif name in displayItem:
                multiLine.add(nameDict[name], xtickLabels, series.values.tolist(),
                              yaxis_name=u'沽购Alpha', is_splitline_show=False, yaxis_min=0, yaxis_max=0.8,
                              yaxis_force_interval=0.08, mark_point=["max", "min"], mark_point_symbolsize=70,
                              **zoomDict)

        overlap.add(etfLine)
        overlap.add(multiLine, yaxis_index=1, is_add_yaxis=True)
        page.add(overlap)

        outputDir = self.jqsdk.getResearchPath(OPTION, 'dailytask')
        page.render(os.path.join(outputDir, 'atm_alpha.html'))

    def strategyAtmReturnNextTradedayOpen(self, start, end):
        """
        保存平值期权收盘最后一分钟到次日开盘5分钟的数据汇总。
        :param start:
        :param end:
        :return:
        """
        tradeDays = self.jqsdk.get_trade_days(start, end)

        l = []
        for tradeDay in tradeDays:
            self.info(u'获取{}'.format(dateToStr(tradeDay)))
            startDt = datetime.combine(tradeDay, time(15, 0, 0))
            nextTradeDay = self.jqsdk.getNextTradeDay(startDt)
            endDt = datetime.combine(nextTradeDay, time(9, 35, 0))

            fn = 'option_daily_{}.csv'.format(dateToStr(tradeDay))
            fp = os.path.join(self.jqsdk.getPricePath('option', 'daily'), fn)
            atmDf = self.getAtmContract(fp)

            dfList = []
            for idx, series in atmDf.iterrows():
                df = self.jqsdk.get_price(series['code'], start_date=startDt, end_date=endDt, frequency='1m')
                dfList.append(df)
            df = dfList[0] + dfList[1]
            df['tradeDay'] = dateToStr(startDt)
            l.append(df)
        df = pd.concat(l)
        return df

    def getNeutralNextTradeDayBar(self, start, end, group='atm', *arg, **kwargs):
        """
        获取delta中性组合（平值或宽跨式组合）次交易日的连续分钟线数据汇总。
        :param start: str
        :param end: str
        :param group: str. 'atm' or 'straddle'
        :return:
        """
        if group == 'atm':
            func = self.getAtmContract
            save_fn = 'atm_continuous_bar.csv'
        elif group == 'straddle':
            func = self.getStraddleContract
            if 'level' in kwargs:
                level = kwargs['level']
            else:
                level = 2
            save_fn = 'straddle_continuous_bar_{}.csv'.format(level)
        else:
            self.error(u'错误的组合类型')
            return

        print(save_fn)

        tradeDays = self.jqsdk.get_trade_days(start, end)

        l = []
        for tradeDay in tradeDays:
            self.info(u'获取delta中性组合分钟数据：{}'.format(dateToStr(tradeDay)))
            startDt = datetime.combine(tradeDay, time(16, 0, 0))
            nextTradeDay = self.jqsdk.getNextTradeDay(startDt)
            endDt = datetime.combine(nextTradeDay, time(16, 0, 0))
            if endDt >= self.jqsdk.today:
                self.info(u'该交易日尚未结束，数据尚未更新！')
                break

            fn = 'option_daily_{}.csv'.format(dateToStr(tradeDay))
            fp = os.path.join(self.jqsdk.getPricePath('option', 'daily'), fn)
            groupDf = func(fp, *arg, **kwargs)
            if groupDf is None:
                continue

            dfList = []
            for idx, series in groupDf.iterrows():
                df = self.jqsdk.get_price(series['code'], start_date=startDt, end_date=endDt, frequency='1m')
                dfList.append(df)
            df = dfList[0] + dfList[1]
            df['tradeDay'] = dateToStr(endDt)
            l.append(df)
        df = pd.concat(l)

        save_fp = os.path.join(self.jqsdk.getResearchPath(OPTION, 'dailytask'), save_fn)
        df.to_csv(save_fp)
        return df

    def updateNeutralNextTradeDayBar(self, end=None, group='atm', *args, **kwargs):
        """
        更新delta中性组合（平值或宽跨式组合）次交易日的连续分钟线数据。
        :param end: str
        :param group: str, 'atm' or 'straddle'
        :return:
        """
        if group == 'atm':
            fn = 'atm_continuous_bar.csv'
        elif group == 'straddle':
            if 'level' in kwargs:
                level = kwargs['level']
            else:
                level = 2
            fn = 'straddle_continuous_bar_{}.csv'.format(level)
        else:
            self.error(u'错误的组合类型')
            return

        if end is None:
            today = self.jqsdk.today
            end = self.jqsdk.getPreTradeDay(today)
            end = dateToStr(end)

        fp = os.path.join(self.jqsdk.getResearchPath(OPTION, 'dailytask'), fn)
        if not os.path.exists(fp):
            self.getNeutralNextTradeDayBar('2017-01-01', end, group=group, *args, **kwargs)
        else:
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            lastDay = df.index.tolist()[-1].strftime('%Y-%m-%d')
            print(lastDay)
            print(end)
            if strToDate(lastDay) >= strToDate(end):
                self.info(u'中性组合的分钟线数据是最新的！')
            else:
                df_new = self.getNeutralNextTradeDayBar(lastDay, end, group=group, *args, **kwargs)
                df = df.append(df_new)
                df.to_csv(fp)

    def getAtmReturnByRange(self, start, end):
        """
        获取某个平值期权跨式组合的区间收益。
        :param start: string. '%Y-%m-%d'
                开始交易日,
        :param end: string. '%Y-%m-%d'
                结束交易日
        :return:
        """
        startFn = 'option_daily_{}.csv'.format(start)
        startFp = os.path.join(self.jqsdk.getPricePath('option', 'daily'), startFn)
        print(startFp)
        atmDf = self.getAtmContract(startFp, keepCurrent=True)

        entry = exit_ = 0
        for idx, series in atmDf.iterrows():
            print(series['code'])
            print(series['trading_code'])
            # 今日确定平值的期权，从下一个交易日开始入场
            nextTradeDay = self.jqsdk.getNextTradeDay(strToDate(start))
            df = self.jqsdk.get_price(series['code'], nextTradeDay, strToDate(end))
            # print(df)
            entry += df.close[0]
            exit_ += df.close[-1]
        return entry, exit_

    def strategyAtmLastTradeDays(self, n=5, startFrom=1501):
        """
        持有跨式平值期权最后n个交易日的收益率
        :param n: int
        :param startFrom: int
                开始计算的月份, 1501表示2015年1月份
        :return:
        """

        def getPreTradeDate(date):
            preDate = strToDate(date) - timedelta(days=n)
            return dateToStr(self.jqsdk.getPreTradeDay(preDate))

        contractDf = copy(self.getContractInfo())  # 避免修改操作影响到原缓存文件
        contractDf.drop_duplicates(subset='last_trade_date', inplace=True)
        contractDf['month'] = contractDf['trading_code'].map(lambda tradingCode: int(tradingCode[7: 11]))
        contractDf.sort_values(by='month', inplace=True)
        contractDf['begin_date'] = contractDf['last_trade_date'].map(getPreTradeDate)
        contractDf = contractDf[contractDf.month >= startFrom]
        # contractDf.set_index('month', inplace=True)

        recList = []
        for idx, series in contractDf.iterrows():
            begin = series['begin_date']
            end = series['last_trade_date']
            if strToDate(end) >= self.jqsdk.today:
                continue

            d = OrderedDict()
            d['month'] = series['month']
            d['begin_date'] = begin
            d['last_trade_date'] = end
            entry, exit_ = self.getAtmReturnByRange(begin, end)
            d['entry'], d['exit'] = roundFloat(entry * 10000, 1), roundFloat(exit_ * 10000, 1)
            recList.append(d)
        df = pd.DataFrame(recList)
        df['return'] = (df['entry'] - df['exit']).map(lambda x: round(x, 1))
        df['returnRate'] = (df['return'] / df['entry']).map(lambda x: round(x, 3))
        return df

    def getStraddleContract(self, fp, level=2, **kwargs):
        """
        获取跨式合约的dataFrame。
        :param fp: 初始日的期权价格数据文件。
        :param level: int
                虚值多少档
        :return: pd.DataFrame
        """
        df = self.getAtmContract(fp, **kwargs)
        atmStrikePrice = df.strikePrice.values[0]

        strikePriceList = self.strikePriceGroupedContractDict.keys()
        strikePriceList.sort()
        print(atmStrikePrice, strikePriceList)
        atmIndex = strikePriceList.index(atmStrikePrice)

        try:
            callStrikePrice = strikePriceList[atmIndex + level]
            putStrikePrice = strikePriceList[atmIndex - level]
            print(callStrikePrice, putStrikePrice)
        except IndexError:
            # 部分时间波动较大，导致平值上下的档位不足，则不采集当日的数据。
            self.info(u'档位不足！')
            return None

        callDf = self.strikePriceGroupedContractDict[callStrikePrice]
        onlyCallDf = callDf[callDf.trading_code.map(lambda code: 'C' in code)]
        putDf = self.strikePriceGroupedContractDict[putStrikePrice]
        onlyPutDf = putDf[putDf.trading_code.map(lambda code: 'P' in code)]

        resDf = pd.concat([onlyCallDf, onlyPutDf])
        # print resDf
        return resDf

    def getAtmContract(self, fp, keepCurrent=False):
        """
        获取某个交易日的平值期权合约数据，并按行权价分组保存当日的合约基础数据。
        :param keepCurrent: Bool
                是否使用当月合约（不切换到下月）
        :param fp: 日行情数据csv文件路径
        :return: pd.DataFrame
        """
        self.info(u'获取当日平值期权:{}'.format(os.path.basename(fp)))

        df = self.getNearbyContract(fp, keepCurrent)
        df['strikePrice'] = df['trading_code'].map(lambda x: int(x[-4:]))  # 提取行权价
        df['isHasM'] = df['trading_code'].map(lambda x: x[11] == 'M')
        if df['isHasM'].any():
            mask = df['trading_code'].map(lambda x: x[11] == 'M')  # 如果有存在M的合约，就过滤掉带A的合约
            df = df[mask]

        # 按行权价分组，并计算认购和认购的权利金价差，存入列表，通过排序得出平值期权
        grouped = df.groupby('strikePrice')
        groupedDict = dict(list(grouped))
        spreadList = []
        for strikePrice, df in groupedDict.items():
            close = df.close.tolist()
            spread = abs(close[0] - close[1])
            spreadList.append((strikePrice, spread))
        spreadList.sort(key=lambda x: (x[1], x[0]))  # 以元祖的第二项(即价差)优先排序

        # 获取平值卖购和卖沽的价和数据
        atmStrike = spreadList[0][0]  # 取平值期权的行权价
        atmDf = groupedDict.get(atmStrike)  # 获取保存平值期权两个合约的dataframe
        self.strikePriceGroupedContractDict = groupedDict  # 保存分组数据供其他函数调用
        return atmDf

    def getAssignedContract(self, fp, callStrikePrice, putStrikePrice, **kwargs):
        """
        获取某个交易日指定行权价的跨式合约数据。
        :param fp:
        :param callStrikePrice:
        :param putStrikePrice:
        :return:
        """
        self.getAtmContract(fp, **kwargs)

        if callStrikePrice == putStrikePrice:
            df = self.strikePriceGroupedContractDict[callStrikePrice]
        else:
            callDf = self.strikePriceGroupedContractDict[callStrikePrice]
            onlyCallDf = callDf[callDf.trading_code.map(lambda code: 'C' in code)]
            putDf = self.strikePriceGroupedContractDict[putStrikePrice]
            onlyPutDf = putDf[putDf.trading_code.map(lambda code: 'P' in code)]
            df = pd.concat([onlyCallDf, onlyPutDf])
        return df

    def getMergedPrice(self, start, end, func, dataType='close', **kwargs):
        """
        获取期权组合指定日期区间的价和数据。
        :param start: datetime.date 或 datetime.datetime
        :param end: datetime.datetime
        :param func: methods
        :param dataType: str
        :param kwargs: func运行的输入函数
        :return: pandas.Series
        """
        startFn = 'option_daily_{}.csv'.format(dateToStr(start))
        startFp = os.path.join(self.jqsdk.getPricePath('option', 'daily'), startFn)
        groupDf = func(startFp, **kwargs)
        # print(groupDf)

        # 获取价和图label
        month = groupDf['month'].iloc[0]
        leg1 = groupDf['strikePrice'].iloc[0]
        leg2 = groupDf['strikePrice'].iloc[1]
        label = '{}-{}-{}'.format(month, leg1, leg2)

        # 获取价和数据

        closeList = []
        klineList = []
        for idx, series in groupDf.iterrows():
            df = self.jqsdk.get_price(series['code'], start, end.replace(hour=18), '1m')
            klineList.append(df)
            closeList.append(df.close)

        if dataType == 'close':
            mergeClose = closeList[0] + closeList[1]
            mergeClose = mergeClose.map(lambda x: round(x, 4))
            mergeClose.name = label
            # print(mergeClose)
            return mergeClose
        elif dataType == 'kline':
            mergeKline = klineList[0] + klineList[1]
            mergeKline.index.name = label
            return mergeKline
        else:
            self.error('Wrong Data type')

    def getOneWeekMergedPrice(self, func, dataType='close', **kwargs):
        """
        获取上周五（包含）到今日的期权跨式组合价和数据。
        :param: func: methods
                 获取价和数据的函数
        :param: **kwargs: func运行的输入参数
        :return: pandas.Series
        """
        lastFri = getLastFriday()
        if not self.jqsdk.isTradeDay(lastFri):
            lastFri = self.jqsdk.getPreTradeDay(lastFri)
        if dataType == 'close':
            mergeClose = self.getMergedPrice(lastFri, self.jqsdk.today, func, dataType=dataType, **kwargs)
            return mergeClose
        else:
            mergeDf = self.getMergedPrice(lastFri, self.jqsdk.today, func, dataType=dataType, **kwargs)
            return mergeDf

    def plotMergePriceKline(self, func, filename, **kwargs):
        df = self.getOneWeekMergedPrice(func, dataType='kline', **kwargs)
        title = df.index.name
        plotter = KlinePlotter(df)
        plotter.plotAll(title, filename)

    def plotMergedPrice(self, filename, mergeClose):
        """
        绘制期权跨式组合的价和走势图。
        :param: filename: string
                    保存的html文件名
        :param: mergeClose: pandas.series
                    期权组合价和数据
        """

        close = mergeClose
        # df = pd.read_csv(getTestPath('atm.csv'), index_col=0, parse_dates=True)
        yMin = close.min() * 0.95
        yMax = close.max() * 1.05

        marklineList = [{'xAxis': i} for i in range(0, len(close), 240)]

        overlap = Overlap(width=1500, height=600)
        line = Line(u'期权组合购沽价和图', is_animation=False)
        line.add(close.name, close.index.tolist(), close.tolist(),
                 is_datazoom_show=True, datazoom_type='both', datazoom_range=[0, 100],
                 is_datazoom_extra_show=True, datazoom_extra_type='both', datazoom_extra_range=[0, 100],
                 xaxis_interval=239, yaxis_min=yMin, yaxis_max=yMax,
                 tooltip_trigger='axis', tooltip_formatter='{b}', tooltip_axispointer_type='cross',
                 mark_line_raw=marklineList, mark_line_symbolsize=0,
                 is_toolbox_show=False)
        overlap.add(line)

        htmlName = '{}.html'.format(filename)
        outputDir = self.jqsdk.getResearchPath(OPTION, 'dailytask')
        overlap.render(os.path.join(outputDir, htmlName))

    def getAtmPriceCombineByDate(self, fp):
        """
        获取某个交易日的平值期权价和分钟数据。
        :param fp: 日行情数据csv文件路径
        :return: tuple(string, pd.Series)
        """
        self.info(u'获取单日价和数据:{}'.format(os.path.basename(fp)))

        df = self.getAtmContract(fp)
        date = strToDate(df['date'].iloc[0])
        month = df['month'].iloc[0]
        atmStrike = df['strikePrice'].iloc[0]
        label = '{}-{}'.format(month, atmStrike)

        closeList = []
        for code in df.code.tolist():
            # 今日确定平值的期权，取下一个交易日的数据
            nextTradeDay = self.jqsdk.getNextTradeDay(date)
            minuteDf = self.jqsdk.get_price(code, nextTradeDay, nextTradeDay + timedelta(days=1), '1m')
            close = minuteDf.close
            closeList.append(close)
        mergeClose = closeList[0] + closeList[1]
        mergeClose = mergeClose.map(lambda x: round(x, 4))
        mergeClose.name = label

        return label, mergeClose

    def getAtmPriceCombineByRange(self, start, end):
        """
        取某个日期区间的平值期权组的价和数据。
        :param start: string
        :param end: string
        :return: list[tuple(string, pd.Series)]
        """
        tradeDays = self.jqsdk.getTradeDayCal()
        start = strToDate(start).date()
        end = strToDate(end).date()
        dateRange = tradeDays[(tradeDays >= start) & (tradeDays <= end)]

        fnames = ['option_daily_{}.csv'.format(dateToStr(date)) for date in dateRange]
        fps = [os.path.join(self.jqsdk.getPricePath('option', 'daily'), filename) for filename in fnames]

        resultList = [self.getAtmPriceCombineByDate(fp) for fp in fps]
        return resultList

    def mergeAtmPriceCombine(self):
        """
        拼接每日的平值期权价合数据。以匹配绘图模块。
        同一合约的纵向拼接，不同合约的横向拼接。
        :return: pd.DataFrame
        """
        # 如果没有指定日期，默认获取最近7个交易日
        if not self.atmStart or not self.atmEnd:
            self.atmStart, self.atmEnd = self.getRecentDays()
        resultList = self.getAtmPriceCombineByRange(self.atmStart, self.atmEnd)

        # 把按交易日组成的数据列表改成按行权价分组
        merger = dict()
        for label, series in resultList:
            merger.setdefault(label, []).append(series)

        # 拼合所有数据
        data = []
        for label, seriesList in merger.items():
            if len(seriesList) > 1:
                df = pd.concat(seriesList)  # 同一行权价在多个交易日成为平值的，先纵向连接
                data.append(df)
            else:
                data.append(seriesList[0])  # 某个行权价仅平值一天，只有一个数据
        self.atmCombinePrice = pd.concat(data, axis=1, sort=True)
        return self.atmCombinePrice

    def plotAtmCombinePrice(self):
        """
        绘制平值期权价和走势图。
        :return:
        """
        if self.atmCombinePrice is None:
            self.mergeAtmPriceCombine()
        df = self.atmCombinePrice
        # df = pd.read_csv(getTestPath('atm.csv'), index_col=0, parse_dates=True)
        yMin = df.min().sort_values().values[0] * 0.95
        yMax = df.max().sort_values().values[-1] * 1.05

        idx = df.index.tolist()
        marklineList = [{'xAxis': i} for i in range(0, len(idx), 240)]

        overlap = Overlap(width=1500, height=600)
        line = Line(u'平值购沽价和图', is_animation=False)
        for colName, series in df.iteritems():
            value = series.tolist()
            # print(value, len(value))
            line.add(colName, idx, value,
                     is_datazoom_show=True, datazoom_type='both', datazoom_range=[0, 100],
                     is_datazoom_extra_show=True, datazoom_extra_type='both', datazoom_extra_range=[0, 100],
                     xaxis_interval=239, yaxis_min=yMin, yaxis_max=yMax,
                     tooltip_trigger='axis', tooltip_formatter='{b}', tooltip_axispointer_type='cross',
                     mark_line_raw=marklineList, mark_line_symbolsize=0,
                     is_toolbox_show=False)
            overlap.add(line)

        htmlName = 'atmCombine.html'
        outputDir = self.jqsdk.getResearchPath(OPTION, 'dailytask')
        overlap.render(os.path.join(outputDir, htmlName))

    def plotMinute(self):
        """
        测试方法。
        :return:
        """
        m1_a = self.jqsdk.get_price('10001562.XSHG', '2018-12-25', '2018-12-25 16:00:00', '1m')
        m1_b = self.jqsdk.get_price('10001562.XSHG', '2018-12-27', '2018-12-27 16:00:00', '1m')
        m2 = self.jqsdk.get_price('10001571.XSHG', '2018-12-26', '2018-12-26 16:00:00', '1m')
        m1 = pd.concat([m1_a, m1_b])
        m1_close = m1.close
        m2_close = m2.close
        m1_close.name = 'm1c'
        m2_close.name = 'm2c'
        l = [m1_close, m2_close]
        df = pd.concat(l, axis=1)
        idx = df.index.tolist()

        overlap = Overlap(width=1400, height=600)

        line = Line(u'测试断层数据')
        for colName, series in df.iteritems():
            value = series.tolist()
            line.add(colName, idx, value, is_datazoom_show=True, datazoom_type='both')
            overlap.add(line)
        overlap.render(getTestPath('duancengMinute.html'))

    def calcRatioByDate(self, fp):
        """
        计算单日的沽购比值
        :param fp: filepath
                单日价格文件
        :return: dict
        """
        if self.isOnlyNearby:
            df = self.getNearbyContract(fp)
        else:
            df = pd.read_csv(fp, index_col=0)

        df['contract_type'] = df.code.map(self.getContractType)
        # df.to_csv(getTestPath('ratiosource.csv'))

        grouped = df.groupby(by='contract_type')
        resultDf = grouped.sum()
        resultDf.to_csv(getTestPath('ratiosum.csv'))

        volume = resultDf['volume']
        money = resultDf['money']
        position = resultDf['position']

        resultDict = dict()
        resultDict['date'] = df['date'].iloc[0]
        resultDict['volumeRatio'] = round(float(volume['PO']) / volume['CO'], 2)
        resultDict['moneyRatio'] = round(float(money['PO']) / money['CO'], 2)
        resultDict['positionRatio'] = round(float(position['PO']) / position['CO'], 2)
        return resultDict

    def load_files(self, fpList):
        """
        从给定的文件名列表依次计算每日的沽购数据。
        :param fpList:
        :return:
        """
        data = []
        for fp in fpList:
            try:
                self.info(u'计算{}'.format(fp))
                data.append(self.calcRatioByDate(fp))
            except IOError:
                self.error(u'{}:{}'.format(FILE_IS_NOT_EXISTED, fp))
        df = pd.DataFrame(data)
        df = df.set_index('date')
        return df

    def getRatio(self):
        """
        从数据文件目录中读取每天的数据，生成每日的沽购比数据，并加入50etf的标的价格走势。
        :return:
        """
        if self.isOnlyNearby:
            filename = 'pc_ratio_data_nearby.csv'
        else:
            filename = 'pc_ratio_data.csv'
        path = os.path.join(self.jqsdk.getPricePath(OPTION, STUDY_DAILY), filename)
        root = self.jqsdk.getPricePath(OPTION, DAILY)

        if not os.path.exists(path):
            fps = [os.path.join(root, fn) for fn in os.listdir(root)]
            df = self.load_files(fps)
            df.to_csv(path, encoding='utf-8-sig')
        else:
            df = pd.read_csv(path, index_col=0)
            lastday = strToDate(df.index.values[-1])

            missionDays = self.jqsdk.get_trade_days(start_date=lastday + timedelta(days=1))
            if len(missionDays) != 0:
                fps = []
                for day in missionDays:
                    fn = '{}_{}_{}.csv'.format(OPTION, DAILY, dateToStr(day))
                    fp = os.path.join(root, fn)
                    fps.append(fp)
                df_new = self.load_files(fps)
                df = pd.concat([df, df_new])
                df.to_csv(path, encoding='utf-8-sig')

        # 添加50etf数据
        # df = df.set_index('date')
        start = '2015-02-09'
        end = df.index[-1]
        underlyingPrice = self.jqsdk.get_price(self.underlyingSymbol, start_date=start, end_date=end)
        underlyingPrice.index = underlyingPrice.index.map(dateToStr)
        close = underlyingPrice['close']
        df[self.underlyingSymbol] = close
        return df

    def getRatioIndicator(self):
        """
        获取持仓量比值的衍生计算数据
        :return:
        """
        f = lambda x: round(x, 2)
        df = self.getRatio()

        # df['positionRatio_ma5'] = df['positionRatio'].rolling(5).mean()
        # df['positionRatioToMa5'] = df['positionRatio'] / df['positionRatio_ma5']
        # df['positionRatioToMa5'] = df['positionRatioToMa5'].map(f)
        df['moneyRatio_ma5'] = df['moneyRatio'].rolling(5).mean()
        df['moneyRatio_ma10'] = df['moneyRatio'].rolling(10).mean()
        # df['moneyRatioToMa5'] = df['moneyRatio'] / df['moneyRatio_ma5']
        # df['moneyRatioToMa5'] = df['moneyRatioToMa5'].map(f)
        return df

    def plotRatio(self):
        """
        绘制沽购各项指标走势图。
        :return:
        """
        width = 1500
        height = 600
        displayItem = ['volumeRatio', self.underlyingSymbol, 'moneyRatio', 'moneyRatio_ma5', 'moneyRatio_ma10']

        if self.isOnlyNearby:
            title = u'50et期权近月沽购比'
            htmlName = 'ratio_nearby.html'
        else:
            title = u'50etf期权全月份沽购比'
            htmlName = 'ratio_all_month.html'

        nameDict = {'moneyRatio': u'沽购成交金额比',
                    'volumeRatio': u'沽购成交量比',
                    'positionRatio': u'沽购持仓量比',
                    'moneyRatio_ma5': u'沽购成交金额比ma5',
                    'moneyRatio_ma10': u'沽购成交金额比ma10',
                    'positionRatioToMa5': u'P/C持仓量比5天平均',
                    'moneyRatioToMa5': u'P/C成交金额比5天平均',
                    self.underlyingSymbol: u'50ETF收盘价'
                    }

        zoomDict = {
            'is_datazoom_show': True,
            'datazoom_type': 'both',
            'line_width': 2,
            'yaxis_name_size': 14,
            'yaxis_name_gap': 35,
            'tooltip_trigger': 'axis',
            'tooltip_axispointer_type': 'cross'
        }

        df = self.getRatioIndicator()
        xtickLabels = df.index.tolist()

        page = Page()
        overlap = Overlap(width=width, height=height)

        # 对比组图
        multiLine = Line(u'沽购比走势图')
        etfLine = Line(title)
        areaLine = Line(u'持仓量沽购比')
        for name, series in df.iteritems():
            if name == self.underlyingSymbol:
                etfLine.add(nameDict[name], xtickLabels, series.values.tolist(), yaxis_min=1.5, yaxis_max=3.5,
                            yaxis_force_interval=0.2, yaxis_name=u'50etf收盘价', **zoomDict)
            elif name == 'positionRatio':
                areaLine.add(nameDict[name], xtickLabels, series.values.tolist(), is_fill=True, area_opacity=0.4,
                             yaxis_name=u'沽购比', is_splitline_show=False, yaxis_min=0, yaxis_max=5,
                             yaxis_force_interval=0.5, mark_point=["max", "min"], mark_point_symbolsize=45,
                             **zoomDict)
            elif name in displayItem:
                multiLine.add(nameDict[name], xtickLabels, series.values.tolist(), mark_point=["max", "min"],
                              mark_point_symbolsize=45, **zoomDict)

        overlap.add(etfLine)
        overlap.add(areaLine, yaxis_index=1, is_add_yaxis=True)
        overlap.add(multiLine, yaxis_index=1)
        page.add(overlap)

        outputDir = self.jqsdk.getResearchPath(OPTION, 'dailytask')
        page.render(os.path.join(outputDir, htmlName))

    def get50etfAdjustDayList(self, start, end):
        """
        获取50etf样本股调整生效日期(6,12月第二个周五的下个交易日)的列表
        :param start: datetime.datetime
        :param end: datetime.datetime
        :return: datetime.date
        """
        days = [start.date()]
        for year in range(start.year, end.year + 1, 1):
            for month in [6, 12]:
                date = datetime(year, month, 1) + relativedelta(weekday=FR(2))
                date = self.jqsdk.getNextTradeDay(date)
                if start.date() <= date <= end.date():
                    days.append(date)
        days.append(end.date())
        return days

    def getMoneyFlowOf50EtfInPeriod(self, start, end):
        """
        获取50etf某个调整区间（在此区间，成分股不变）所有成分股的资金流向数据。
        :param start: datetime.datetime
        :param end: datetime.datetime
        :return:
        """
        self.info(u'获取50etf资金流：{} to {}'.format(dateToStr(start), dateToStr(end)))
        end = self.jqsdk.getPreTradeDay(end)
        stocks = self.jqsdk.get_index_stocks(self.etfIdx, date=start)
        df = self.jqsdk.get_money_flow(stocks, start_date=start, end_date=end)
        return df

    def getMoneyFlowOf50Etf(self, start, end):
        """
        获取某个时间段50etf所有成分股的资金流向数据。
        :param start:
        :param end:
        :return:
        """
        days = [datetime(d.year, d.month, d.day) for d in self.get50etfAdjustDayList(start, end)]
        periodList = []
        for idx in range(0, len(days) - 1):
            dStart = days[idx]
            if idx + 1 < len(days) - 1:
                # 如果不是整个大区间的最后一天，小区间的最后一天需要往前推一个交易日，避免获取重复一天的数据
                dEnd = self.jqsdk.getPreTradeDay(days[idx + 1])
                dEnd = datetime(dEnd.year, dEnd.month, dEnd.day)
            else:
                dEnd = days[idx + 1]
            periodList.append((dStart, dEnd))
        # print(periodList)

        dfs = [self.getMoneyFlowOf50EtfInPeriod(*dateTuple) for dateTuple in periodList]
        df_merge = pd.concat(dfs)
        df_merge.date = df_merge.date.map(dateToStr)
        path = os.path.join(self.jqsdk.getPricePath(OPTION, STUDY_DAILY), '50etf_money_flow.csv')
        df_merge.to_csv(path)
        return df_merge

    def updateMoneyFlowOf50Etf(self):
        """
        更新50etf资金流量数据，要确保更新的起止日期是在成分股的一个调整区间之内，不能跨区间。
        :return:
        """
        path = os.path.join(self.jqsdk.getPricePath(OPTION, STUDY_DAILY), '50etf_money_flow.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            lastDate = df.date.iloc[-1]
            # print(lastDate, type(lastDate))
            newStart = self.jqsdk.getNextTradeDay(strToDate(lastDate))
            newStart = datetime(newStart.year, newStart.month, newStart.day)
            if newStart > self.jqsdk.today:  # 如果新的开始日期大于今天
                self.info(u'{}'.format(FILE_IS_NEWEST))
            else:
                dfNew = self.getMoneyFlowOf50EtfInPeriod(newStart, self.jqsdk.today)
                dfNew.date = dfNew.date.map(dateToStr)
                df = pd.concat([df, dfNew])
                df.to_csv(path)
        else:
            self.info(u'找不到数据文件，请先获取。')

    def plotMoneyFlowOf50Etf(self):
        """
        绘制50etf资金流向图表。
        :return:
        """
        width = 1500
        height = 600
        displayItem = ['net_amount_main', 'net_amount_xl', 'net_amount_l']

        nameDict = {'net_amount_main': u'主力净额',
                    'net_amount_xl': u'超大单净额',
                    'net_amount_l': u'大单净额',
                    self.underlyingSymbol: u'50ETF收盘价'
                    }

        zoomDict = {
            'is_datazoom_show': True,
            'datazoom_type': 'both',
            'line_width': 2,
            'yaxis_name_size': 14,
            'yaxis_name_gap': 35,
            'tooltip_trigger': 'axis',
            'tooltip_axispointer_type': 'cross'
        }

        path = os.path.join(self.jqsdk.getPricePath(OPTION, STUDY_DAILY), '50etf_money_flow.csv')
        df = pd.read_csv(path, index_col=0)
        df.set_index('date', inplace=True)
        df = df[displayItem]
        df = df.groupby(df.index).sum()
        underlyingPrice = self.jqsdk.get_price(self.underlyingSymbol, start_date=df.index.values[0],
                                               end_date=df.index.values[-1])
        underlyingPrice.index = underlyingPrice.index.map(dateToStr)
        df[self.underlyingSymbol] = underlyingPrice['close']
        xtickLabels = df.index.tolist()
        # print(df)

        page = Page()
        overlap = Overlap(width=width, height=height)

        # 对比组图
        bars = Bar(u'50etf资金流向图')
        etfLine = Line(u'50etf资金流向图')
        for name, series in df.iteritems():
            if name == self.underlyingSymbol:
                etfLine.add(nameDict[name], xtickLabels, series.values.tolist(),
                            yaxis_min=1.8, yaxis_max=3.4, yaxis_force_interval=0.2,
                            yaxis_name=u'50etf收盘价', is_splitline_show=False, **zoomDict)
            elif name in displayItem:
                bars.add(nameDict[name], xtickLabels,
                         map(roundFloat, np.divide(series.values, 10000).tolist()),
                         bar_category_gap='50%',
                         yaxis_name=u'50etf资金流向（亿元）',
                         mark_point_symbolsize=45, **zoomDict)

        overlap.add(bars)
        overlap.add(etfLine, yaxis_index=1, is_add_yaxis=True)
        page.add(overlap)

        outputDir = self.jqsdk.getResearchPath(OPTION, 'dailytask')
        page.render(os.path.join(outputDir, '50etf_money_flow.html'))


class VixAnalyzer(LoggerWrapper):
    def __init__(self, collector, symbol):
        super(VixAnalyzer, self).__init__()
        self.symbol = symbol
        self.collector = collector
        self.underlyingAnalyzer = OptionUnderlyingAnalyzer(self.collector, self.symbol)

        self.dataFilePath = self.getFilePath()
        self.vix = None
        self.plotter = Plotter(self.collector)

    @staticmethod
    def getPercentile(seq):
        latest = seq[-1]
        return stats.percentileofscore(seq, latest)

    def getFilePath(self):
        fn = 'vixBar.csv'
        fp = os.path.join(self.collector.getResearchPath(OPTION, 'qvix'), fn)
        return fp

    def getOutputPath(self, fn):
        dirName = os.path.dirname(self.dataFilePath)
        return os.path.join(dirName, fn)

    def updateVixData(self):
        """
        获取期权论坛波动率指数日线数据。
        :return:
        """
        try:
            resp = requests.get(QVIX_URL)
            csv_reader = csv.DictReader(resp.iter_lines())
            csv_list = list(csv_reader)
        except:
            print('error occur when get qvix.')
        else:
            name_dict = {
                '1': 'datetime',
                '1 ': 'datetime',
                '2': 'open',
                '3': 'high',
                '4': 'low',
                '5': 'close'
            }
            df = pd.DataFrame(csv_list)
            df.rename(columns=name_dict, inplace=True)

            while True:
                if df.iloc[-1]['open'] == '':
                    df = df[:-1]
                else:
                    break

            df['datetime'] = df['datetime'].map(
                lambda t_stamp: dateToStr(datetime.fromtimestamp(float(t_stamp) / 1000)))
            for item in ['open', 'high', 'low', 'close']:
                df[item] = df[item].map(lambda str_num: float(str_num))

            df.set_index('datetime', inplace=True)
            df.to_csv(self.dataFilePath)
            print('update completely!')
            return df

    def getVix(self):
        """
        获取隐含波动率数据
        :return:
        """
        if self.vix is None:
            self.vix = pd.read_csv(self.dataFilePath, index_col=0, parse_dates=True)
        return self.vix

    def getVixPercentileDist(self, n=0):
        if n == 0:
            name = 'History'
        else:
            name = 'Recent_{}_days'.format(n)
        df = self.getVix()
        df = df.iloc[-n:]
        seq = df['close'].values
        percentSeq = [np.percentile(seq, i) for i in range(0, 105, 5)]
        s = pd.Series(percentSeq, index=range(0, 105, 5))
        s.name = name
        return s

    def getAllVixPercentileDist(self, nList):
        sList = list()
        for n in nList:
            s = self.getVixPercentileDist(n)
            sList.append(s)
        df = pd.concat(sList, axis=1)
        df.index = df.index.map(lambda x: '{}%'.format(x))
        saveCsv(df, self.getOutputPath('vix_percentile_dist.csv'))
        return df

    def getVixPercentile(self, n=0):
        """
        计算vix所处的百分位
        :param n: int. 计算近n个交易日的数据，取0时表示使用所有历史数据。
        :return: pd.Series
        """
        df = self.getVix()
        if n == 0:
            name = 'percentile'
            seq = df['close'].values
            df[name] = df['close'].map(lambda number: stats.percentileofscore(seq, number))
        else:
            name = 'percentile_{}'.format(n)
            df[name] = df['close'].rolling(n).apply(self.getPercentile, raw=True)
        return df[name]

    def getHVPercentile(self, n=0):
        """
        计算HV所处的百分位
        :param n: int. 近n个交易日的hv20数据，取0表示使用自有50etf期权的历史数据。
        :return:
        """
        df = self.getVix()
        self.getHisVolatility()
        sourceName = 'HV_20'
        if n == 0:
            name = 'percentile_HV20'
            seq = df[sourceName].values
            df[name] = df[sourceName].map(lambda number: stats.percentileofscore(seq, number))
        else:
            name = 'percentile_HV20_{}'.format(n)
            df[name] = df[sourceName].rolling(n).apply(self.getPercentile, raw=True)
        return df[name]

    def getHisVolatility(self, n=20):
        """
        获取标的的历史波动率
        :param n: 最近n个交易日的历史波动率
        :return:
        """
        df = self.getVix()
        name = 'HV_{}'.format(n)
        df[name] = self.underlyingAnalyzer.getHistVolatility(n)
        return df[name]

    def analyzeVixAndUnderlying(self, start=None, end=None):
        df = self.getVix()
        underlying = self.underlyingAnalyzer.getPrice()

        underlying['pre_close'] = underlying['close'].shift(1)
        df['open_return'] = (underlying['open'] - underlying['pre_close']) / underlying['pre_close']
        df['close_return'] = (underlying['close'] - underlying['pre_close']) / underlying['pre_close']
        df['amplitude'] = (underlying['high'] - underlying['low']) / underlying['pre_close']

        df['vix_pre_close'] = df['close'].shift(1)
        df['vix_open_return_abs'] = df['open'] - df['vix_pre_close']
        df['vix_open_return_pct'] = df['vix_open_return_abs'] / df['vix_pre_close']
        df['vix_close_return_abs'] = df['close'] - df['vix_pre_close']
        df['vix_close_return_pct'] = df['vix_close_return_abs'] / df['vix_pre_close']
        df['vix_drawback_abs'] = df['high'] - df['vix_pre_close']
        df['vix_drawback_pct'] = df['vix_drawback_abs'] / df['vix_pre_close']

        start = datetime(2016, 6, 13) if start is None else strToDate(start)
        if end is None:
            df = df[start:]
        else:
            end = strToDate(end)
            df = df[start: end]
        df = copy(df)

        csv_fn = 'vix_underlying_corr_{}_{}.csv'.format(dateToStr(pd.to_datetime(df.index.values[0])),
                                                        dateToStr(pd.to_datetime(df.index.values[-1])))
        saveCsv(df, self.getOutputPath(csv_fn))

        fig, axes = plt.subplots(3, 3, figsize=(32, 32))
        axes = axes.flatten()

        sns.regplot('open_return', 'vix_open_return_pct', data=df, ax=axes[0])
        sns.regplot('open_return', 'vix_drawback_pct', data=df, ax=axes[1])
        sns.regplot('open_return', 'vix_close_return_pct', data=df, ax=axes[2])
        sns.regplot('amplitude', 'vix_drawback_pct', data=df, ax=axes[3])
        sns.regplot('amplitude', 'vix_close_return_pct', data=df, ax=axes[4])
        sns.regplot('close_return', 'vix_close_return_pct', data=df, ax=axes[5])
        sns.regplot('vix_open_return_pct', 'vix_drawback_pct', data=df, ax=axes[6])
        sns.regplot('vix_open_return_pct', 'vix_close_return_pct', data=df, ax=axes[7])

        plt.subplots_adjust(wspace=0.3, hspace=0.2)
        fn = 'vix_underlying_corr_{}_{}'.format(dateToStr(pd.to_datetime(df.index.values[0])),
                                                dateToStr(pd.to_datetime(df.index.values[-1])))
        fig.savefig(self.getOutputPath(fn))

    def plotPercentile(self, nList):
        """
        输出vix百分位的折线图
        :return:
        """
        title = u'波动率指数百分位图'
        colNameList = []
        for n in nList:
            if n == 0:
                colNameList.append('percentile')
            else:
                colNameList.append('percentile_{}'.format(n))
            self.getVixPercentile(n)

        self.plotter.setCsvData(self.vix)
        line = self.plotter.plotLine(title, colNameList)
        self.plotter.addRenderItem(line)

    def plotHvPercentile(self, nList):
        """
        输出HV百分位的折线图
        :return:
        """
        title = u'历史波动率百分位图'
        colNameList = []
        for n in nList:
            if n == 0:
                colNameList.append('percentile_HV20')
            else:
                colNameList.append('percentile_HV20_{}'.format(n))
            self.getHVPercentile(n)

        self.plotter.setCsvData(self.vix)
        line = self.plotter.plotLine(title, colNameList)
        self.plotter.addRenderItem(line)

    def plotPercentileDist(self, nList):
        """
        绘制IV分位百分位分布
        :param nList:
        :return:
        """
        title = u'波动率指数分位分布值'
        colNameList = []
        if 0 in nList:
            colNameList.append('History')
        recentList = ['Recent_{}_days'.format(i) for i in nList if i != 0]
        colNameList.extend(recentList)

        df = self.getAllVixPercentileDist(nList)
        self.plotter.setCsvData(df)
        line = self.plotter.plotLine(title, colNameList)
        self.plotter.addRenderItem(line)

    def plotVolatilityDiff(self, nList):
        """
        绘制隐含波动率、历史波动率
        :param n:
        :return:
        """
        title = u'历史波动率与隐含波动率对比'
        df = self.getVix()
        colNameList = []
        colNameList.append('IV')
        for n in nList:
            colNameList.append('HV_{}'.format(n))
            self.getHisVolatility(n)
        df['IV'] = df['close']

        self.plotter.setCsvData(self.vix)
        line = self.plotter.plotLine(title, colNameList)
        self.plotter.addRenderItem(line)

    def render(self, path, filename):
        """
        输出html
        :param path:
        :param filename:
        :return:
        """
        self.plotter.renderHtml(path, filename)


class OptionUnderlyingAnalyzer(LoggerWrapper):
    """
    期权标的分析
    """

    def __init__(self, collector, symbol):
        super(OptionUnderlyingAnalyzer, self).__init__()
        self.collector = collector
        self.symbol = symbol
        self.today = self.collector.today
        self.dataFilePath = self.getFilePath()

        self.price = None
        self.vix = None

    def getFilePath(self):
        fn = '{}_daily_prefq.csv'.format(self.symbol.split('.')[0])
        fp = os.path.join(self.collector.getResearchPath(OPTION, 'underlying'), fn)
        return fp

    def getOutputPath(self, fn):
        dirName = os.path.dirname(self.dataFilePath)
        return os.path.join(dirName, fn)

    def updatePrice(self, start):
        end = self.today
        fp = self.dataFilePath

        if not os.path.exists(fp):
            df = self.collector.get_price(self.symbol, start_date=start, end_date=end, fq='pre')
            df = df[df.volume != 0]
            saveCsv(df, fp)
        else:
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            lastDay = df.index.tolist()[-1] + timedelta(days=1)
            if lastDay <= end:
                df_new = self.collector.get_price(self.symbol, lastDay, end, fq='pre')
                df_new = df_new[df_new.volume != 0]
                df = df.append(df_new)
                saveCsv(df, fp)

    def getPrice(self):
        if self.price is None:
            self.price = pd.read_csv(self.dataFilePath, index_col=0, parse_dates=True)
        return self.price

    def getVix(self):
        if self.vix is None:
            vixAnalyzer = VixAnalyzer(self.collector, self.symbol)
            self.vix = vixAnalyzer.getVix()
        return self.vix

    @staticmethod
    def getDuration(df, start, end):
        start = datetime(2005, 2, 23) if start is None else strToDate(start)
        if end is None:
            df = df[start:]
        else:
            end = strToDate(end)
            df = df[start: end]
        df = copy(df)
        return df

    def preProcessChange(self):
        df = self.getPrice()
        df['pre_close'] = df['close'].shift(1)
        df['open_return'] = (df['open'] - df['pre_close']) / df['pre_close']
        df['close_return'] = (df['close'] - df['pre_close']) / df['pre_close']
        df['up_max'] = (df['high'] - df['pre_close']) / df['pre_close']
        df['down_max'] = (df['low'] - df['pre_close']) / df['pre_close']
        df['amplitude'] = (df['high'] - df['low']) / df['pre_close']
        df.dropna(inplace=True)
        return df

    def analyzeChange(self):
        d = OrderedDict()
        d['All'] = 0
        d['5-year'] = 250 * 5
        d['3-year'] = 250 * 3
        d['2-year'] = 250 * 2
        d['1-year'] = 250
        d['6-month'] = 125

        df = self.preProcessChange()

        items = ['open_return', 'close_return', 'up_max', 'down_max', 'amplitude']

        for item in items:
            statsLst = []
            for k, v in d.items():
                statsItem = df[item][-v:].describe()
                statsItem.name = k
                statsLst.append(statsItem)
            stats_df = pd.concat(statsLst, axis=1)
            stats_df = stats_df.T
            saveCsv(stats_df, self.getOutputPath('stats_{}.csv'.format(item)))

    def plotChange(self, start=None, end=None):
        df = self.preProcessChange()
        df = self.getDuration(df, start, end)

        fig, axes = plt.subplots(3, 5, figsize=(80, 48))
        axes = axes.transpose()
        items = ['open_return', 'close_return', 'up_max', 'down_max', 'amplitude']
        for idx, item in enumerate(items):
            data = df[item].values
            sns.distplot(data, bins=100, color='g', ax=axes[idx][0]).set_title(item)
            sns.kdeplot(data, color='r', cumulative=True, ax=axes[idx][1]).set_title(item)
            sns.boxplot(data, ax=axes[idx][2]).set_title(item)

            for i in range(3):
                axes[idx][i].xaxis.set_major_locator(ticker.MultipleLocator(0.01))
                if i == 1:
                    axes[idx][i].yaxis.set_major_locator(ticker.MultipleLocator(0.02))

        plt.subplots_adjust(wspace=0.3, hspace=0.2)
        fn = 'vix_underlying_corr_{}_{}'.format(dateToStr(pd.to_datetime(df.index.values[0])),
                                                dateToStr(pd.to_datetime(df.index.values[-1])))
        fig.savefig(self.getOutputPath(fn))

    def getImpliedVolatility(self):
        """获取期权论坛隐含波动率"""
        try:
            return self.price['IV']
        except KeyError:
            df = self.getPrice()
            df['IV'] = self.getVix()['close']
            return df['IV']

    def getHistVolatility(self, n, isLogReturn=True):
        """计算历史波动率"""
        name = 'HV_{}'.format(n)
        try:
            df = self.getPrice()
            return df[name]
        except KeyError:
            df = self.getPrice()
            df['returns'] = df.close.pct_change()
            if isLogReturn:
                df['returns'] = np.log(df['returns'] + 1)
            df[name] = df['returns'].rolling(n).std() * np.sqrt(252) * 100
            return df[name]

    def plotVolatilityBox(self):
        """绘制历史波动率盒须图"""
        statsLst = []
        volLst = []
        df = self.getPrice()
        self.getImpliedVolatility()

        statsLst.append(df['IV'].describe())
        volLst.append('IV')
        for n in [10, 20, 30, 60, 120]:
            name = 'HV_{}'.format(n)
            self.getHistVolatility(n)
            statsLst.append(df[name].describe())
            volLst.append(name)

        statsDf = pd.concat(statsLst, axis=1)
        saveCsv(statsDf, self.getOutputPath('boxplotData.csv'))

        dataDf = df[volLst]
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(1, 1, 1)
        sns.boxplot(data=dataDf, ax=ax)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        fig.savefig(self.getOutputPath('volBoxPlot.png'))


class NeutralContractAnalyzer(LoggerWrapper):
    """
    Delta中性组合分析
    """

    def __init__(self, collector, underlyingSymbol):
        super(NeutralContractAnalyzer, self).__init__()

        self.jqsdk = collector
        self.underlyingSymbol = underlyingSymbol

        self.start = '2017-01-01'
        self.today = self.jqsdk.today

        # 数据缓存
        self.contractInfo = None
        self.underlyingPrice = None

        self.contractTypeDict = None
        self.tradingCodeDict = None
        self.lastTradeDateDict = None
        self.underlyingDict = None

        self.strikePriceGroupedContractDict = None
        self.calls = None
        self.puts = None
        self.underlyingClose = None

        self.commission = 2
        self.slippage = 2
        self.interval = 3

    def setCommission(self, commission):
        self.commission = commission

    def setSlippage(self, slippage):
        self.slippage = slippage

    def setInterval(self, interval):
        self.interval = interval

    def getOutputPath(self, fn):
        return os.path.join(self.jqsdk.getResearchPath(OPTION, 'atm'), fn)

    def getContractInfo(self):
        """
        获取期权合约的基础信息，并做缓存。
        :return: pd.DataFrame
        """
        if self.contractInfo is None:
            filename = u'{}_{}.csv'.format(OPTION, BASIC)
            path = os.path.join(self.jqsdk.getBasicPath(OPTION), filename)
            df = pd.read_csv(path, index_col=0)
            mask = (df.underlying_symbol == self.underlyingSymbol)
            df = df[mask]
            df = df.set_index('code')
            self.contractInfo = df
        return self.contractInfo

    def getUnderlyingPrice(self):
        """
        获取期权标的的价格数据
        :return:
        """
        if self.underlyingPrice is None:
            end = self.today
            fn = '{}_daily.csv'.format(self.underlyingSymbol.split('.')[0])
            fp = os.path.join(self.jqsdk.getResearchPath(OPTION, 'atm'), fn)

            if not os.path.exists(fp):
                df = self.jqsdk.get_price(self.underlyingSymbol, start_date='2005-02-23', end_date=end, fq=None)
                df = df[df.volume != 0]
                saveCsv(df, fp)
            else:
                df = pd.read_csv(fp, index_col=0, parse_dates=True)
                lastDay = df.index.tolist()[-1] + timedelta(days=1)
                if lastDay < end:
                    df_new = self.jqsdk.get_price(self.underlyingSymbol, lastDay, end, fq=None)
                    df_new = df_new[df_new.volume != 0]
                    df = df.append(df_new)
                    saveCsv(df, fp)
            self.underlyingPrice = df
        return self.underlyingPrice

    def codeToTradingCode(self, code):
        """
        获取交易代码（下单用）- 合约编码（期权详情的简码）的映射，并缓存。
        :param code: string
                合约编码
        :return: string
                合约交易代码
        """
        if self.tradingCodeDict is None:
            df = self.getContractInfo()
            self.tradingCodeDict = df['trading_code'].to_dict()
        return self.tradingCodeDict.get(code, None)

    def codeToLastTradeDate(self, code):
        """
        获取交易代码（下单用）- 合约最后交易日的映射，并缓存。
        :param code: string
                合约编码
        :return: string
                合约最后交易日
        """
        if self.lastTradeDateDict is None:
            df = self.getContractInfo()
            self.lastTradeDateDict = df['last_trade_date'].to_dict()
        return self.lastTradeDateDict.get(code, None)

    def codeToContractType(self, code):
        """
        获取交易代码（下单用）- 看涨看跌类型简码的映射，并缓存
        :param code: string
                合约编码
        :return: string
                看涨看跌类型简码
        """
        if self.contractTypeDict is None:
            df = self.getContractInfo()
            self.contractTypeDict = df['contract_type'].to_dict()
        return self.contractTypeDict.get(code, None)

    def codeToUnderlying(self, code):
        """
        获取交易代码（下单用）- 标的编码的映射，并缓存
        :param code: str
        :return:
        """

        if self.underlyingDict is None:
            df = self.getContractInfo()
            self.underlyingDict = df['underlying_symbol'].to_dict()
        return self.underlyingDict.get(code, None)

    def getNearbyContract(self, fp, keepCurrent=False):
        """
        从每日价格表中选出当月合约，如果距离到期日不足7日，则选下月合约，每日分析，不需要缓存。
        :param fp: filePath
                某个交易日的所有期权合约日行情数据
        :param keepCurrent: Bool
                是否使用当月合约（不切换到下月）
        :return: pd.DataFrame
        """
        df = pd.read_csv(fp, index_col=0)

        # 筛选出匹配标的合约
        df['underlying_symbol'] = df['code'].map(self.codeToUnderlying)
        df = df[df['underlying_symbol'] == self.underlyingSymbol]
        df = copy(df)

        df['last_trade_date'] = df['code'].map(self.codeToLastTradeDate)
        df['trading_code'] = df['code'].map(self.codeToTradingCode)
        df['contract_type'] = df['code'].map(self.codeToContractType)
        df['month'] = df['trading_code'].map(lambda tradingCode: int(tradingCode[7: 11]))
        df['strikePrice'] = df['trading_code'].map(lambda x: int(x[-4:]))
        df['isHasM'] = df['trading_code'].map(lambda x: x[11] == 'M')

        monthList = df['month'].drop_duplicates().tolist()
        monthList.sort()
        # print(monthList)
        thisMonth, nextMonth = monthList[0], monthList[1]

        df_current = df[df.month == thisMonth]
        date = strToDate(df_current['date'].iloc[0])
        lastDate = strToDate(df_current['last_trade_date'].iloc[0])
        if lastDate - date > timedelta(days=7) or keepCurrent is True:
            resDf = df_current
        else:
            df_next = df[df.month == nextMonth]
            resDf = df_next
        return resDf

    def getAtmContract(self, fp, method, keepCurrent=False):
        """
        获取某个交易日的平值期权合约。
        :param fp: filepath
                日行情数据csv文件路径
        :param method: str, 'simple' or 'match'.
                simple-选取同一行权价价差最小的组合；match选择标的收盘价附近的价差最小的沽购组合（可能会不同行权价）
        :param keepCurrent: Bool
                是否使用当月合约（不切换到下月）
        :return: pd.DataFrame
        """
        self.info(u'分析当日平值期权:{}'.format(os.path.basename(fp)))
        df = self.getNearbyContract(fp, keepCurrent)
        # 筛选M合约放在平值期权的函数，防止有的月份某日都是A的，导致没有选到当月数据
        if df['isHasM'].any():
            df = df[df.isHasM]

        dt = strToDate(df['date'].iloc[0])
        underlyingClose = self.getUnderlyingPrice().close[dt]
        self.underlyingClose = underlyingClose

        if method == 'simple':
            # 按行权价分组，并计算认购和认购的权利金价差，存入列表，通过排序得出平值期权
            grouped = df.groupby('strikePrice')
            groupedDict = dict(list(grouped))
            spreadList = []
            for strikePrice, df in groupedDict.items():
                close = df.close.tolist()
                spread = abs(close[0] - close[1])
                spreadList.append((strikePrice, spread))
            spreadList.sort(key=lambda x: (x[1], x[0]))  # 以元祖的第二项(即价差)优先排序

            # 获取平值卖购和卖沽的价和数据
            atmStrike = spreadList[0][0]  # 取平值期权的行权价
            atmDf = groupedDict.get(atmStrike)  # 获取保存平值期权两个合约的dataframe
            self.strikePriceGroupedContractDict = groupedDict
        elif method == 'match':
            # dt = strToDate(df['date'].iloc[0])
            # underlyingClose = self.getUnderlyingPrice().close[dt]

            df.sort_values(by='strikePrice', inplace=True)
            callDf = df[df.contract_type == 'CO']
            putDf = df[df.contract_type == 'PO']

            strikePriceArray = callDf.strikePrice.values
            strikeUnderlyingSpread = abs(strikePriceArray - underlyingClose * 1000)
            closestIdx = np.argmin(strikeUnderlyingSpread)
            closestStrike = strikePriceArray[closestIdx]
            if abs(closestStrike - underlyingClose * 1000) / float(closestStrike) < 0.005:
                callLeg = callDf[callDf.strikePrice == closestStrike].iloc[0]
                putLeg = putDf[putDf.strikePrice == closestStrike].iloc[0]
            else:
                try:
                    callLeg = callDf[callDf.strikePrice > underlyingClose * 1000].iloc[0]
                except IndexError:
                    # 若日内波动太大，标的价格超出最虚档，只能选用最虚档
                    callLeg = callDf.iloc[-1]
                spreadClose = abs(putDf.close - callLeg.close)
                spreadClose.sort_values(inplace=True)
                putLegIdx = spreadClose.index[0]
                putLeg = putDf.loc[putLegIdx]

            atmDf = pd.concat([callLeg, putLeg], axis=1)
            atmDf = atmDf.T

            # if callLeg.strikePrice - putLeg.strikePrice < 0:
            #     print(u'Xu to Shi', callLeg.strikePrice, putLeg.strikePrice)

            self.calls = callDf
            self.puts = putDf
            self.underlyingClose = underlyingClose
        else:
            self.info(u'错误的分析方法！')
            return
        return atmDf

    def getStraddleContract(self, fp, method, level=1, **kwargs):
        """
        获取某个交易日的平值期权合约。
        :param fp: filepath
                日行情数据csv文件路径
        :param method: str, 'simple' or 'match'.
                参考getAtmContract的method
        :param level: int
                虚值多少档
        :return: pd.DataFrame
        """

        if method == 'simple':
            df = self.getAtmContract(fp, method, **kwargs)
            atmStrikePrice = df.strikePrice.values[0]

            strikePriceList = self.strikePriceGroupedContractDict.keys()
            strikePriceList.sort()
            print(atmStrikePrice, strikePriceList)
            atmIndex = strikePriceList.index(atmStrikePrice)

            try:
                callStrikePrice = strikePriceList[atmIndex + level]
                if atmIndex - level >= 0:
                    putStrikePrice = strikePriceList[atmIndex - level]
                else:
                    self.error(u'沽档位不足')
                    return
            except IndexError:
                # 部分时间波动较大，导致平值上下的档位不足，则不采集当日的数据。
                self.info(u'档位超出所有行权价范围！')
                return

            callDf = self.strikePriceGroupedContractDict[callStrikePrice]
            onlyCallDf = callDf[callDf.trading_code.map(lambda code: 'C' in code)]
            putDf = self.strikePriceGroupedContractDict[putStrikePrice]
            onlyPutDf = putDf[putDf.trading_code.map(lambda code: 'P' in code)]

            resDf = pd.concat([onlyCallDf, onlyPutDf])
            return resDf
        elif method == 'match':
            df = self.getAtmContract(fp, method, **kwargs)
            callStrikePrice = df[df.contract_type == 'CO'].iloc[0].strikePrice
            putStrikePrice = df[df.contract_type == 'PO'].iloc[0].strikePrice

            # print(callStrikePrice, putStrikePrice)
            strikePriceList = self.calls.strikePrice.to_list()
            # print(strikePriceList)
            try:
                callIdx = strikePriceList.index(callStrikePrice) + level
                putIdx = strikePriceList.index(putStrikePrice) - level
                if putIdx < 0:
                    self.error(u'沽档位不足！')
                    return
                callLeg = self.calls[self.calls.strikePrice == strikePriceList[callIdx]]
                putLeg = self.puts[self.puts.strikePrice == strikePriceList[putIdx]]
                resDf = pd.concat([callLeg, putLeg])
                # print(resDf)
                return resDf
            except IndexError:
                self.error(u'档位超出所有行权价范围！')
                return

    def getNeutralGroupInfo(self, start, end, group='atm', method='match', *arg, **kwargs):
        """
        获取delta中性组合（平值或宽跨式组合）的基本信息。
        :param start: str
        :param end: str
        :param group: str. 'atm' or 'straddle'
        :param method: str. 'simple' or 'match'
        :return:
        """
        if group == 'atm':
            func = self.getAtmContract
            save_fn = 'atm_info_{}.csv'.format(method)
        elif group == 'straddle':
            func = self.getStraddleContract
            if 'level' in kwargs:
                level = kwargs['level']
            else:
                level = 1
            save_fn = 'straddle_{}_info_{}.csv'.format(level, method)
        else:
            self.error(u'错误的组合类型')
            return

        infoList = []
        tradeDays = self.jqsdk.get_trade_days(start, end)
        for tradeDay in tradeDays:
            self.info(u'获取delta中性组合分钟数据：{}'.format(dateToStr(tradeDay)))
            fn = 'option_daily_{}.csv'.format(dateToStr(tradeDay))
            fp = os.path.join(self.jqsdk.getPricePath('option', 'daily'), fn)

            fnGreece = 'option_greece_{}.csv'.format(dateToStr(tradeDay))
            fpGreece = os.path.join(self.jqsdk.getPricePath('option', 'greece'), fnGreece)

            groupDf = func(fp, method=method, *arg, **kwargs)
            if groupDf is None:
                continue

            greeceDf = pd.read_csv(fpGreece)
            callCode = groupDf.iloc[0]['code']
            putCode = groupDf.iloc[1]['code']
            print(callCode, putCode)
            print(fnGreece)
            callGreece = greeceDf[greeceDf.code == callCode].iloc[0]
            putGreece = greeceDf[greeceDf.code == putCode].iloc[0]

            month = groupDf.month.values[0]
            callLabel = str(groupDf.iloc[0]['strikePrice']) + groupDf.iloc[0]['contract_type'][0]
            putLabel = str(groupDf.iloc[1]['strikePrice']) + groupDf.iloc[1]['contract_type'][0]
            groupLabel = '-'.join([callLabel, putLabel])

            d = dict()
            d['tradeDay'] = dateToStr(tradeDay)
            d['month'] = month
            d['groupLabel'] = groupLabel
            d['underlyingClose'] = self.underlyingClose
            d['callDelta'] = callGreece['delta']
            d['callGamma'] = callGreece['gamma']
            d['callVega'] = callGreece['vega']
            d['callTheta'] = callGreece['theta']
            d['putDelta'] = putGreece['delta']
            d['putGamma'] = putGreece['gamma']
            d['putVega'] = putGreece['vega']
            d['putTheta'] = putGreece['theta']
            infoList.append(d)

        outputCol = ['tradeDay', 'groupLabel', 'underlyingClose', 'month']
        outputGreece = [j + i.capitalize() for i in ['delta', 'gamma', 'vega', 'theta'] for j in ['call', 'put']]
        outputCol.extend(outputGreece)

        df = pd.DataFrame(infoList)
        df = df[outputCol]
        save_fp = os.path.join(self.jqsdk.getResearchPath(OPTION, 'atm'), save_fn)
        saveCsv(df, save_fp)
        return df

    def getNeutralNextTradeDayBar(self, start, end, group='atm', method='match', *arg, **kwargs):
        """
        获取delta中性组合（平值或宽跨式组合）今交易日14：50-次交易日的连续分钟线数据汇总。
        :param start: str
        :param end: str
        :param group: str. 'atm' or 'straddle'
        :param method: str. 'simple' or 'match'
        :return:
        """
        if group == 'atm':
            func = self.getAtmContract
            save_fn = 'atm_continuous_bar_{}.csv'.format(method)
        elif group == 'straddle':
            func = self.getStraddleContract
            if 'level' in kwargs:
                level = kwargs['level']
            else:
                level = 1
            save_fn = 'straddle_{}_continuous_bar_{}.csv'.format(level, method)
        else:
            self.error(u'错误的组合类型')
            return

        tradeDays = self.jqsdk.get_trade_days(start, end)

        allList = []
        for tradeDay in tradeDays:
            self.info(u'获取delta中性组合分钟数据：{}'.format(dateToStr(tradeDay)))
            startDt = datetime.combine(tradeDay, time(14, 50, 0))
            nextTradeDay = self.jqsdk.getNextTradeDay(startDt)
            endDt = datetime.combine(nextTradeDay, time(16, 0, 0))
            if endDt >= datetime.now():
                self.info(u'该交易日尚未结束，数据尚未更新！')
                break

            fn = 'option_daily_{}.csv'.format(dateToStr(tradeDay))
            fp = os.path.join(self.jqsdk.getPricePath('option', 'daily'), fn)

            groupDf = func(fp, method=method, *arg, **kwargs)
            if groupDf is None:
                continue
            month = groupDf.month.values[0]
            callLabel = str(groupDf.iloc[0]['strikePrice']) + groupDf.iloc[0]['contract_type'][0]
            putLabel = str(groupDf.iloc[1]['strikePrice']) + groupDf.iloc[1]['contract_type'][0]
            groupLabel = '-'.join([callLabel, putLabel])

            legList = []
            for idx, series in groupDf.iterrows():
                df = self.jqsdk.get_price(series['code'], start_date=startDt, end_date=endDt, frequency='1m')
                legList.append(df)
            df = legList[0] + legList[1]
            df['tradeDay'] = dateToStr(endDt)
            df['month'] = month
            df['groupLabel'] = groupLabel
            df['underlyingClose'] = self.underlyingClose

            allList.append(df)
        df = pd.concat(allList)

        save_fp = os.path.join(self.jqsdk.getResearchPath(OPTION, 'atm'), save_fn)
        saveCsv(df, save_fp)
        return df

    def updateNeutralNextTradeDayBar(self, end=None, group='atm', method='match', *args, **kwargs):
        """
        更新delta中性组合连续分钟线数据。
        :param end: str
        :param group: str. 'atm' or 'straddle'
        :param method: str. 'match' or 'simple'
        :return:
        """
        if group == 'atm':
            fn = 'atm_continuous_bar_{}.csv'.format(method)
        elif group == 'straddle':
            if 'level' in kwargs:
                level = kwargs['level']
            else:
                level = 1
            fn = 'straddle_{}_continuous_bar_{}.csv'.format(level, method)
        else:
            self.error(u'错误的组合类型')
            return

        if end is None:
            today = self.jqsdk.today
            end = self.jqsdk.getPreTradeDay(today)
            end = dateToStr(end)

        fp = os.path.join(self.jqsdk.getResearchPath(OPTION, 'atm'), fn)
        if not os.path.exists(fp):
            self.getNeutralNextTradeDayBar('2017-01-01', end, group=group, method=method, *args, **kwargs)
        else:
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            lastDay = df.index.tolist()[-1].strftime('%Y-%m-%d')
            if strToDate(lastDay) >= strToDate(end):
                self.info(u'中性组合的分钟线数据是最新的！')
            else:
                df_new = self.getNeutralNextTradeDayBar(lastDay, end, group=group, method=method, *args, **kwargs)
                df = df.append(df_new)
                saveCsv(df, fp)

    def loadNeutralContinuousBar(self, group='atm', method='match', level=1):
        """
        获取中性组合的分钟bar
        :param group: str. 'atm' or 'straddle'
        :param method: str. 'simple' or 'match'
        :param level: int
        :return:
        """
        if group == 'atm':
            fn = 'atm_continuous_bar_{}.csv'.format(method)
        elif group == 'straddle':
            fn = 'straddle_{}_continuous_bar_{}.csv'.format(level, method)
        else:
            return
        fp = os.path.join(self.jqsdk.getResearchPath(OPTION, 'atm'), fn)
        print(fp)
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        return df

    @staticmethod
    def barToDaily(df):
        s = pd.Series()
        for i in ['tradeDay', 'month', 'groupLabel', 'underlyingClose']:
            s[i] = df.iloc[0][i]
        s['pre_close'] = df.iloc[10]['close']
        s['open'] = df.iloc[11]['open']
        # s['open-1'] = df.iloc[12]['open']
        # s['open-2'] = df.iloc[13]['open']
        # s['open-3'] = df.iloc[14]['open']
        s['close'] = df.iloc[-1]['close']
        s['high'] = df.high.values.max()
        s['low'] = df.low.values.min()

        df2 = pd.DataFrame(s)
        df2 = df2.T
        return df2

    def getOHLCdaily(self, group='atm', method='match', level=1):
        """
        分钟线合成日线
        :param group: str. 'atm' or 'straddle'
        :param method: str. 'simple' or 'match'
        :param level: int
        :return:
        """
        if group == 'atm':
            fn = 'atm_ohlc_{}.csv'.format(method)
        elif group == 'straddle':
            fn = 'straddle_{}_ohlc_{}.csv'.format(level, method)
        else:
            return

        data = self.loadNeutralContinuousBar(group=group, method=method, level=level)

        df = data.groupby('tradeDay').apply(self.barToDaily)
        df.set_index('tradeDay', inplace=True)
        df.index = df.index.map(strToDate)
        saveCsv(df, self.getOutputPath(fn))
        return df

    def getLast5Days(self, method='match'):
        atmFn = 'atm_ohlc_{}.csv'.format(method)
        atmFp = self.getOutputPath(atmFn)
        dfAtm = pd.read_csv(atmFp, index_col=0, parse_dates=True)
        dfAtm = dfAtm.groupby('month').apply(lambda df: df.iloc[-5:])
        saveCsv(dfAtm, self.getOutputPath('last5_' + atmFn))

        for i in range(1, 4):
            strangleFn = 'straddle_{}_ohlc_{}.csv'.format(i, method)
            strangleFp = self.getOutputPath(strangleFn)
            dfStrangle = pd.read_csv(strangleFp, index_col=0, parse_dates=True)
            dfStrangle = dfStrangle.groupby('month').apply(lambda df: df.iloc[-5:])
            saveCsv(dfStrangle, self.getOutputPath('last5_' + strangleFn))

    def dailyBackTest(self, group='atm', method='match', level=1, start='pre_close', isLast5=False):
        if group == 'atm':
            fn = 'atm_ohlc_{}.csv'.format(method)
        elif group == 'straddle':
            fn = 'straddle_{}_ohlc_{}.csv'.format(level, method)
        else:
            return

        if isLast5:
            fn = 'last5_' + fn

        df = pd.read_csv(self.getOutputPath(fn), index_col=0, parse_dates=True)
        df['daily_trade_return'] = (df[start] - df['close']) * 10000
        df['daily_max_drawback'] = (df[start] - df['high']) * 10000
        df['trade_net'] = df['daily_trade_return'].cumsum()

        if start == 'pre_close':
            df['commission'] = 0
            df['slippage'] = 0

            n = df['commission'].values
            m = df['slippage'].values
            for idx, value in enumerate(n[:]):
                if idx % self.interval == 0:
                    n[idx] += self.commission * 2
            for idx, value in enumerate(m[:]):
                if idx % self.interval == 0:
                    m[idx] += self.slippage * 2 * 2
        else:
            df['commission'] = 4
            df['slippage'] = 8

        df['total_return'] = df['daily_trade_return'] - df['commission'] - df['slippage']
        df['net'] = df['total_return'].cumsum()
        win = df[df['total_return'] > 0]
        lose = df[df['total_return'] <= 0]

        btFn = 'backtesting_{}_{}_{}_{}.csv'.format(group, method, level, start)
        if isLast5:
            btFn = 'last5_' + btFn
        saveCsv(df, self.getOutputPath(btFn))

        allCount = len(df)
        winCount = len(win)
        loseCount = len(lose)
        top10Lose = lose['total_return'].sort_values()[0: int(loseCount * 0.1)].sum()
        top20Lose = lose['total_return'].sort_values()[0: int(loseCount * 0.2)].sum()

        profit = df['total_return'].sum()
        average_profit = profit / float(allCount)
        average_win = win['total_return'].sum() / float(winCount)
        average_lose = lose['total_return'].sum() / float(loseCount)

        maxWin = win['total_return'].max()
        maxLose = lose['total_return'].min()

        print('=' * 30)
        print(u'组合：{}, Delta中性方法：{}，调仓时间：{}'.format(group, method, start))
        print(u'总交易日：%d' % allCount)
        print(u'盈利交易日：%d' % winCount)
        print(u'亏损交易日：%d' % loseCount)
        print(u'总利润：%d' % profit)
        print(u'平均利润：%.2f' % average_profit)
        print(u'手续费：%d' % df['commission'].sum())
        print(u'交易滑点: %d' % df['slippage'].sum())
        print(u'胜率：%.2f %%' % (float(winCount) / float(allCount) * 100))
        print(u'盈利总额：%d' % win['total_return'].sum())
        print(u'亏损总额：%d' % lose['total_return'].sum())
        print(u'亏损排名前10分位的亏损总额: %d' % top10Lose)
        print(u'亏损排名前10分位的亏损占比: %.2f %%' % (top10Lose / lose['total_return'].sum() * 100))
        print(u'亏损排名前20分位的亏损总额: %d' % top20Lose)
        print(u'亏损排名前20分位的亏损占比: %.2f %%' % (top20Lose / lose['total_return'].sum() * 100))
        print(u'平均盈利金额：%.2f' % average_win)
        print(u'平均亏损金额：%.2f' % average_lose)
        print(u'最大盈利金额：%d' % maxWin)
        print(u'最大亏损金额：%d' % maxLose)
        print(u'盈亏比：%.2f' % (abs(average_win / average_lose)))

    def backTestingCompare(self, method='match', start='pre_close'):
        dfs = []
        atmFn = 'backtesting_atm_{}_1_{}.csv'.format(method, start)
        dfAtm = pd.read_csv(self.getOutputPath(atmFn), index_col=0, parse_dates=True)
        month = dfAtm['month']
        atmReturn = dfAtm['daily_trade_return']
        atmReturn.name = 'atm_r'
        atmDrawback = dfAtm['daily_max_drawback']
        atmDrawback.name = 'atm_d'
        dfs.append(month)
        dfs.append(atmReturn)
        dfs.append(atmDrawback)

        for i in range(1, 4):
            fn = 'backtesting_straddle_{}_{}_{}.csv'.format(method, i, start)
            df = pd.read_csv(self.getOutputPath(fn), index_col=0, parse_dates=True)
            tradeReturn = df['daily_trade_return']
            tradeReturn.name = 'strangle_{}_r'.format(i)
            drawback = df['daily_max_drawback']
            drawback.name = 'strangle_{}_d'.format(i)
            dfs.append(tradeReturn)
            dfs.append(drawback)

        resDf = pd.concat(dfs, axis=1)
        saveCsv(resDf, self.getOutputPath('backtesting_compare.csv'))

    def backTestingPosition(self):
        margin_dict = OrderedDict()
        margin_dict['atm'] = 7788
        margin_dict['strangle_1'] = 6400
        margin_dict['strangle_2'] = 5130
        margin_dict['strangle_3'] = 4075

        margin_level_dict = OrderedDict()
        margin_level_dict['20%'] = 200000
        margin_level_dict['30%'] = 300000
        margin_level_dict['40%'] = 400000
        margin_level_dict['50%'] = 500000
        margin_level_dict['60%'] = 600000
        margin_level_dict['70%'] = 700000

        output_item = ['month']
        # for item in margin_dict.keys():
        #     output_item.append(item)

        fp = self.getOutputPath('backtesting_compare.csv')
        df = pd.read_csv(fp, index_col=0, parse_dates=True)

        df['has_s1'] = df['strangle_1_r'].map(lambda x: 0 if np.isnan(x) else 1)
        df['has_s2'] = df['strangle_2_r'].map(lambda x: 0 if np.isnan(x) else 1)
        df['has_s3'] = df['strangle_3_r'].map(lambda x: 0 if np.isnan(x) else 1)
        df['group_count'] = df['has_s1'] + df['has_s2'] + df['has_s3'] + 1
        df.fillna(value=0, inplace=True)

        for key, value in margin_level_dict.items():
            group_margin = 'group_margin_{}'.format(key)
            atm_count = 'atm_count_{}'.format(key)
            s1_count = 's1_count_{}'.format(key)
            s2_count = 's2_count_{}'.format(key)
            s3_count = 's3_count_{}'.format(key)
            atm_return = 'atm_return_{}'.format(key)
            s1_return = 's1_return_{}'.format(key)
            s2_return = 's2_return_{}'.format(key)
            s3_return = 's3_return_{}'.format(key)
            all_return = 'returns_{}'.format(key)
            atm_drawback = 'atm_drawback_{}'.format(key)
            s1_drawback = 's1_drawback_{}'.format(key)
            s2_drawback = 's2_drawback_{}'.format(key)
            s3_drawback = 's3_drawback_{}'.format(key)
            all_drawback = 'drawbacks_{}'.format(key)

            output_item.append(all_return)
            output_item.append(all_drawback)

            df[group_margin] = value / df['group_count'] / 1.2
            df[atm_count] = np.floor(df[group_margin] / margin_dict['atm'])
            df[s1_count] = np.floor(df[group_margin] / margin_dict['strangle_1']) * df['has_s1']
            df[s2_count] = np.floor(df[group_margin] / margin_dict['strangle_2']) * df['has_s2']
            df[s3_count] = np.floor(df[group_margin] / margin_dict['strangle_3']) * df['has_s3']

            df[atm_return] = df['atm_r'] * df[atm_count]
            df[s1_return] = df['strangle_1_r'] * df[s1_count]
            df[s2_return] = df['strangle_2_r'] * df[s2_count]
            df[s3_return] = df['strangle_3_r'] * df[s3_count]
            df[all_return] = df[atm_return] + df[s1_return] + df[s2_return] + df[s3_return]

            df[atm_drawback] = df['atm_d'] * df[atm_count]
            df[s1_drawback] = df['strangle_1_d'] * df[s1_count]
            df[s2_drawback] = df['strangle_2_d'] * df[s2_count]
            df[s3_drawback] = df['strangle_3_d'] * df[s3_count]
            df[all_drawback] = df[atm_drawback] + df[s1_drawback] + df[s2_drawback] + df[s3_drawback]

        df = df[output_item]

        vixAnalyzer = VixAnalyzer(self.jqsdk, self.underlyingSymbol)
        percentileList = []
        for n in [0, 120, 250]:
            percentile = vixAnalyzer.getVixPercentile(n=n)
            percentileList.append(percentile)
            # percentile.index.name = 'tradeDay'

        mergeList = [df]
        mergeList.extend(percentileList)
        df = pd.concat(mergeList, axis=1)
        df = df.dropna()

        saveCsv(df, self.getOutputPath('backtesting_position.csv'))

    def analyzeStop(self):
        fp = self.getOutputPath('backtesting_position.csv')
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        df.sort_values(by='drawbacks_40%', inplace=True)

        resList = []
        allDays = len(df)
        stopList = range(1000, 20000, 1000)
        for stop in stopList:
            stopDict = OrderedDict()
            dfStoped = df[df['drawbacks_40%'] <= (-stop)]
            dfNonStoped = df[df['drawbacks_40%'] > (-stop)]
            stopDays = len(dfStoped)
            stopNum = -stop * stopDays
            nonStopNum = dfNonStoped['returns_40%'].sum()

            stopDict['stopMoney'] = -stop
            stopDict['allDays'] = allDays
            stopDict['stopDays'] = stopDays
            stopDict['allStopMoney'] = stopNum
            stopDict['nonStopReturn'] = nonStopNum
            stopDict['finalReturn'] = stopNum + nonStopNum

            resList.append(stopDict)

        resDf = pd.DataFrame(resList)
        saveCsv(resDf, self.getOutputPath('analyze_stop.csv'))

    def removeOHLCgap(self, df=None, *args, **kwargs):
        """
        消除期权组合（每日合约可能不同）非交易因素造成的跳空。
        :return:
        """
        name_dict = {'open_new': 'open', 'high_new': 'high', 'low_new': 'low', 'close_new': 'close'}

        if df is None:
            df = self.getOHLCdaily(*args, **kwargs)

        # 计算当日相对开盘价的涨跌幅
        df['open_rate'] = 0
        df['low_rate'] = (df['low'] - df['open']) / df['open']
        df['high_rate'] = (df['high'] - df['open']) / df['open']
        df['close_rate'] = (df['close'] - df['open']) / df['open']

        # 通过收盘价的累计值计算新起点，并在新起点基础上叠加今日的波动
        df['close_rate_cul'] = df['close_rate'].cumsum()
        df['open_new'] = df['close_rate_cul'].shift(1)
        df.fillna(value=0, inplace=True)
        df['high_new'] = df['open_new'] + df['high_rate']
        df['low_new'] = df['open_new'] + df['low_rate']
        df['close_new'] = df['open_new'] + df['close_rate']

        df = df[['open_new', 'high_new', 'low_new', 'close_new']]
        df = df.rename(columns=name_dict)
        return df

    def removeGapByMonth(self, *args, **kwargs):
        """
        按期权合约月份分组消除跳空
        :param args:
        :param kwargs:
        :return:
        """
        df = self.getOHLCdaily(*args, **kwargs)
        df = df.groupby(by='month').apply(self.removeOHLCgap)
        return df

    def plotAtmOHLC(self, method='match', isGap=True, divideByMonth=False, isIncludePre=False):
        """
        输出平值期权组合连续日k线图
        :param isIncludePre:
        :param divideByMonth:
        :param isGap:
        :param method:
        :return:
        """
        preFlag = 'includePre' if isIncludePre else 'excludePre'
        byMonthFlag = 'byMonth' if divideByMonth else 'noByMonth'
        if isGap:
            name_flag = 'gap'
            df = self.getOHLCdaily(group='atm', method=method, isIncludePre=isIncludePre)
        else:
            name_flag = 'nogap'
            if divideByMonth:
                df = self.removeGapByMonth(group='atm', method=method, isIncludePre=isIncludePre)
            else:
                df = self.removeOHLCgap(group='atm', method=method, isIncludePre=isIncludePre)

        fn = 'atm_{}_ohlc_daily_{}_{}_{}.html'.format(method, name_flag, byMonthFlag, preFlag)

        plotter = KlinePlotter(df)
        plotter.plotAll('ATM-{}-{} method'.format(method, name_flag), 'atm', fn, item=['ma'])
        self.info(u'平值期权{}-{}日K绘制完成'.format(method, name_flag))

    def plotStraddleOHLC(self, method='match', isGap=True, divideByMonth=False, isIncludePre=False, level=1):
        """
        输出宽跨式期权组合日k线图
        :param isIncludePre:
        :param divideByMonth:
        :param isGap:
        :param method:
        :param level:
        :return:
        """
        preFlag = 'includePre' if isIncludePre else 'excludePre'
        byMonthFlag = 'byMonth' if divideByMonth else 'noByMonth'
        if isGap:
            name_flag = 'gap'
            df = self.getOHLCdaily(group='straddle', method=method, isIncludePre=isIncludePre, level=level)
        else:
            name_flag = 'nogap'
            if divideByMonth:
                df = self.removeGapByMonth(group='straddle', method=method, isIncludePre=isIncludePre)
            else:
                df = self.removeOHLCgap(group='straddle', method=method, isIncludePre=isIncludePre)

        fn = 'straddle_{}_{}_ohlc_daily_{}_{}_{}.html'.format(level, method, name_flag, byMonthFlag, preFlag)

        plotter = KlinePlotter(df)
        plotter.plotAll('Straddle-{}-{}-{}'.format(level, method, name_flag), 'atm', fn, item=['ma'])
        self.info(u'宽跨式{}档-{}-{}日K绘制完成'.format(level, method, name_flag))


class QvixPlotter(LoggerWrapper):
    """
    Qvix日线行情绘图器。
    """

    def __init__(self):
        super(QvixPlotter, self).__init__()
        self.jqsdk = JQDataCollector()

        self.dailyData = None

        self.underlyingSymbol = '510050.XSHG'
        self.width = 1500
        self.height = 600

        self.genStyle = {
            'is_datazoom_show': True,
            'datazoom_type': 'both'
        }

    def getCsvData(self):
        """
        从文件中读取波动率数据
        :return: pd.DataFrame
        """
        if self.dailyData is None:
            # 从网络获取最新数据
            # resp = requests.get(QVIX_URL)
            # csv_reader = csv.DictReader(resp.iter_lines())
            # csv_list = list(csv_reader)
            # latest = csv_list[-1]
            # self.info(u'波动率数据更新完成')
            #
            # if latest['2'] == '':
            #     latest = csv_list[-2]
            #
            # print(latest)
            # print(dateToStr(datetime.fromtimestamp(float(latest['1']) / 1000)))
            # d = dict()
            # d['open'] = float(latest['2'])
            # d['high'] = float(latest['3'])
            # d['low'] = float(latest['4'])
            # d['close'] = float(latest['5'])
            # s = pd.Series(d)
            # s.name = dateToStr(datetime.fromtimestamp(float(latest['1']) / 1000))
            #
            # filename = 'qvix_daily.csv'
            # fp = os.path.join(getDataDir(), RESEARCH, OPTION, 'qvix', filename)
            # df = pd.read_csv(fp, index_col=0, parse_dates=True)
            # df.index = df.index.map(dateToStr)
            # if df.iloc[-1].name != s.name:
            #     df = df.append(s)
            # df.to_csv(fp, encoding='utf-8')
            # self.dailyData = df

            filename = 'vixBar.csv'
            fp = os.path.join(getDataDir(), RESEARCH, OPTION, 'qvix', filename)
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            df.index = df.index.map(dateToStr)
            self.dailyData = df
        return self.dailyData

    def get50etfPrice(self):
        """
        获取与qvix数据文件日期区间相同的50etf日线信息
        :return: pd.DataFrame
        """
        df = self.getCsvData()
        start = df.index[0]
        end = strToDate(df.index[-1]) + timedelta(days=1)
        end = dateToStr(end)
        # print(start, end)
        underlyingPrice = self.jqsdk.get_price(self.underlyingSymbol, start_date=start, end_date=end)
        underlyingPrice.index = underlyingPrice.index.map(dateToStr)
        return underlyingPrice

    def addCloseMa(self, n):
        """
        计算均线，并保存到数据df中
        :param n: int
        :return: pd.Series
        """
        f = lambda x: round(x, 2)

        df = self.getCsvData()
        colName = 'close_ma{}'.format(n)
        df[colName] = df['close'].rolling(n).mean()
        df[colName] = df[colName].map(f)
        return df[colName]

    def addCloseStd(self, n):
        """
        计算标准差，并保存到数据df中
        :param n: int
        :return: pd.Series
        """
        df = self.getCsvData()
        colName = 'close_std{}'.format(n)
        df[colName] = df['close'].rolling(n).std()
        return df[colName]

    def add50etf(self):
        """
        获取同日期的50etf标的数据，并保存到数据df中
        :return: pd.Series
        """
        df = self.getCsvData()
        # start = df.index[0]
        # end = strToDate(df.index[-1]) + timedelta(days=1)
        # end = dateToStr(end)
        # print(start, end)
        # underlyingPrice = self.jqsdk.get_price(self.underlyingSymbol, start_date=start, end_date=end)
        # underlyingPrice.index = underlyingPrice.index.map(dateToStr)
        underlyingPrice = self.get50etfPrice()
        close = underlyingPrice['close']
        df[self.underlyingSymbol] = close
        return df[self.underlyingSymbol]

    def add50etfParkinsonNumber(self, n):
        """
        计算parkinson波动率
        :param n: int
        :return: pd.Series
        """

        def parkinson(values):
            return np.sqrt(sum(values ** 2 / (4 * np.log(2))) / len(values))

        etfDf = self.get50etfPrice()
        hlReturns = np.log(etfDf.high / etfDf.low)
        parkinsonNumber = hlReturns.rolling(n).apply(parkinson, raw=False)
        annualizedPk = parkinsonNumber * np.sqrt(252) * 100
        annualizedPk = annualizedPk.map(roundFloat)

        df = self.getCsvData()
        colName = 'Parkinson{}'.format(n)
        df[colName] = annualizedPk
        return df[colName]

    def addHistVolotility(self, n, isLogReturn=False):
        """
        计算50etf历史波动率
        :param n: int
        :return: pd.Series
        """
        f = lambda x: round(x, 2)

        close = self.add50etf()
        returns = close.pct_change()
        if isLogReturn:
            returns = np.log(returns + 1)
        dailyVolStds = returns.rolling(n).std()
        annualizedVols = dailyVolStds * np.sqrt(252) * 100  # 单位变成百分比
        annualizedVols = annualizedVols.map(f)

        colName = 'HV{}'.format(n)
        self.dailyData[colName] = annualizedVols
        return self.dailyData[colName]

    def plotKline(self, title=u'k线', addStyle=None):
        """
        返回波动率k线图对象
        :param title:
        :param addStyle: dict
        :return:
        """
        kStyle = self.genStyle.copy()
        if addStyle is not None:
            kStyle.update(addStyle)

        df = self.getCsvData()
        dateList = df.index.tolist()

        gen = df.iterrows()
        ohlc = [[data['open'], data['close'], data['low'], data['high']] for idx_, data in gen]

        kline = Kline(title)
        kline.add(u'Qvix日线', dateList, ohlc, tooltip_formatter=kline_tooltip_formatter, **kStyle)
        # kline.render(getTestPath('qvixkline.html'))

        return kline

    def getOverlapWithKline(self, title, **kwargs):
        """
        返回一个已经叠加了波动率指数k线图的叠加图。
        :param title: string
                叠加图标题
        :return:
        """
        overlap = Overlap(width=self.width, height=self.height)
        kline = self.plotKline(title, **kwargs)
        overlap.add(kline)
        return overlap

    def plotVolDiff(self, n=20):
        """
        绘制隐含波动率、历史波动率及其差额、parkinson波动率
        :param n:
        :return:
        """
        lineStyle = self.genStyle.copy()
        lineStyle['line_width'] = 2

        title = u'50etf历史波动率、隐含波动率对比、parkinson波动率'
        overlap = Overlap(width=self.width, height=self.height)

        data = {}
        hv = 'HV{}'.format(n)
        iv = 'IV'
        parkinson = 'Parkinson{}'.format(n)
        diff = '{}-{}'.format(iv, hv)
        dateList = self.getCsvData().index.tolist()
        data[hv] = self.addHistVolotility(n)
        data[iv] = self.getCsvData().close
        data[parkinson] = self.add50etfParkinsonNumber(n)

        volDiff = data[iv] - data[hv]
        volDiff = volDiff.map(lambda x: round(x, 2))

        for lineName in [hv, iv, parkinson]:
            line = Line(title)
            line.add(lineName, dateList, data[lineName].tolist(),
                     tooltip_trigger='axis', tooltip_axispointer_type='cross',
                     **lineStyle)
            overlap.add(line)

        # 需要颜色区分时使用
        # volp = volDiff.map(lambda x: np.nan if x < 0 else round(x, 2))
        # volm = volDiff.map(lambda x: np.nan if x >= 0 else round(x, 2))
        # barp = Bar()
        # barp.add(diff, dateList, volp.tolist())
        # overlap.add(barp)
        # barm = Bar()
        # barm.add(diff, dateList, volm.tolist(), is_yaxis_inverse=True)
        # overlap.add(barm)

        bar = Bar()
        bar.add(diff, dateList, volDiff.tolist())
        overlap.add(bar)

        htmlName = 'vol_diff.html'
        outputDir = self.jqsdk.getResearchPath(OPTION, 'dailytask')
        overlap.render(os.path.join(outputDir, htmlName))

    def plotICIHIF(self):
        data = OrderedDict()
        today = self.jqsdk.today
        data['ic'] = self.jqsdk.get_price('IC8888.CCFX', start_date='2019-01-01', end_date=dateToStr(today))
        data['ih'] = self.jqsdk.get_price('IH8888.CCFX', start_date='2019-01-01', end_date=dateToStr(today))
        data['if'] = self.jqsdk.get_price('IF8888.CCFX', start_date='2019-01-01', end_date=dateToStr(today))
        data['index'] = data['ic'].index.tolist()

        icihVol = 'IC-IH'
        icifVol = 'IC-IF'
        ifihVol = 'IF-IH'
        data[icihVol] = data['ic'].close - data['ih'].close
        data[icifVol] = data['ic'].close - data['if'].close
        data[ifihVol] = data['if'].close - data['ih'].close

        lineStyle = self.genStyle.copy()
        lineStyle['line_width'] = 2
        page = Page()

        overlapICIH = Overlap(width=self.width, height=self.height)
        for linename in ['ic', 'ih']:
            line = Line(u'IC-IH收盘价差')
            line.add(linename, data[linename].index.tolist(), data[linename].close.tolist(),
                     tooltip_trigger='axis', tooltip_axispointer_type='cross', **lineStyle)
            overlapICIH.add(line)
        barICIH = Line()
        barICIH.add(icihVol, data['index'], data[icihVol])
        overlapICIH.add(barICIH)

        overlapICIF = Overlap(width=self.width, height=self.height)
        for linename in ['ic', 'if']:
            line = Line(u'IC-IF收盘价差')
            line.add(linename, data[linename].index.tolist(), data[linename].close.tolist(),
                     tooltip_trigger='axis', tooltip_axispointer_type='cross', **lineStyle)
            overlapICIF.add(line)
        barICIF = Line()
        barICIF.add(icifVol, data['index'], data[icifVol])
        overlapICIF.add(barICIF)

        overlapIFIH = Overlap(width=self.width, height=self.height)
        for linename in ['if', 'ih']:
            line = Line(u'IF-IH收盘价差')
            line.add(linename, data[linename].index.tolist(), data[linename].close.tolist(),
                     tooltip_trigger='axis', tooltip_axispointer_type='cross', **lineStyle)
            overlapIFIH.add(line)
        barIFIH = Line()
        barIFIH.add(ifihVol, data['index'], data[ifihVol])
        overlapIFIH.add(barIFIH)

        page.add(overlapICIH)
        page.add(overlapICIF)
        page.add(overlapIFIH)
        htmlName = 'ic_ih_if_volDiff.html'
        outputDir = self.jqsdk.getResearchPath(OPTION, 'dailytask')
        page.render(os.path.join(outputDir, htmlName))

    def plotMa(self, n1=5, n2=20, n3=60):
        """
        绘制均线系统叠加图
        :param n1: int
        :param n2: int
        :param n3: int
        :return:
        """
        lineStyle = self.genStyle.copy()
        lineStyle['line_width'] = 2

        title = u'波动率指数移动平均线'
        overlap = Overlap(width=self.width, height=self.height)
        kline = self.plotKline(title)
        overlap.add(kline)

        df = self.getCsvData()
        dateList = df.index.tolist()

        for ma in [n1, n2, n3]:
            label = u'MA{}'.format(ma)
            line = Line()
            line.add(label, dateList, self.addCloseMa(ma).tolist(), **lineStyle)
            overlap.add(line)

        # overlap.render(getTestPath('qvixkline.html'))
        return overlap

    def plotEtf(self):
        """
        绘制etf价格和波动率指数叠加图
        :return:
        """
        lineStyle = self.genStyle.copy()
        lineStyle['line_width'] = 2

        addStyle = {
            'yaxis_min': 10,
            'yaxis_max': 50,
            'yaxis_force_interval': 5,
            'yaxis_name': u'波动率',
            'yaxis_name_size': 14,
            'yaxis_name_gap': 35,
        }
        title = u'波动率指数叠加50etf价格'
        overlap = self.getOverlapWithKline(title, addStyle=addStyle)  # 特殊坐标

        dateList = self.getCsvData().index.tolist()
        line = Line()
        line.add(u'50ETF', dateList, self.add50etf().tolist(),
                 yaxis_min=2, yaxis_max=3.2, yaxis_force_interval=0.12, yaxis_name=u'50etf收盘价',
                 is_splitline_show=False, yaxis_name_size=14, yaxis_name_gap=35,
                 **lineStyle)
        overlap.add(line, yaxis_index=1, is_add_yaxis=True)

        return overlap

    def plotBollChanel(self, n=20):
        """
        绘制boll通道叠加图
        :param n:
        :return:
        """
        lineStyle = self.genStyle.copy()
        lineStyle['line_width'] = 2

        title = u'波动率指数Boll通道'
        overlap = Overlap(width=self.width, height=self.height)
        kline = self.plotKline(title)
        overlap.add(kline)

        df = self.getCsvData()
        dateList = df.index.tolist()

        lineMid = Line()
        lineMid.add(u'布林MID', dateList, self.addCloseMa(n).tolist(), **lineStyle)
        overlap.add(lineMid)

        upList = self.addCloseMa(n) + 2 * self.addCloseStd(n)
        upList = upList.tolist()
        lineUp = Line()
        lineUp.add(u'布林UP', dateList, upList, **lineStyle)
        overlap.add(lineUp)

        downList = self.addCloseMa(n) - 2 * self.addCloseStd(n)
        downList = downList.tolist()
        lineDown = Line()
        lineDown.add(u'布林Down', dateList, downList, **lineStyle)
        overlap.add(lineDown)

        return overlap
        # overlap.render(getTestPath('qvixkline_boll.html'))

    def plotAll(self):
        """
        按顺序绘制所有的叠加图
        :return:
        """
        page = Page()
        etf = self.plotEtf()
        ma = self.plotMa()
        boll = self.plotBollChanel()

        page.add(etf)
        page.add(boll)
        page.add(ma)

        htmlName = 'qvix_daily.html'
        outputDir = self.jqsdk.getResearchPath(OPTION, 'dailytask')
        page.render(os.path.join(outputDir, htmlName))


class Plotter(LoggerWrapper):
    """
    绘图器。
    """

    def __init__(self, collector, df=None):
        super(Plotter, self).__init__()
        self.collector = collector
        self.width = 1600
        self.height = 600
        self.data = df

        self.page = Page()
        self.genStyle = {
            'is_datazoom_show': True,
            'datazoom_type': 'both',
            'tooltip_trigger': 'axis'
        }

    def getCsvData(self):
        return self.data

    def setCsvData(self, df):
        self.data = df

    def plotLine(self, title, colNameList):
        """
        绘制折线
        :param colNameList: list
        :return:
        """
        lineStyle = self.genStyle.copy()
        lineStyle['line_width'] = 2
        overlap = Overlap(width=self.width, height=self.height)

        df = self.getCsvData()
        dateList = df.index.tolist()
        for col in colNameList:
            line = Line(title=title)
            line.add(col, dateList, df[col].tolist(), **lineStyle)
            overlap.add(line)
        return overlap

    def addRenderItem(self, overlap):
        self.page.add(overlap)

    def renderHtml(self, path, filename):
        dir_ = self.collector.getResearchPath(OPTION, path)
        fp = os.path.join(dir_, filename)
        self.page.render(fp)


class KlinePlotter(LoggerWrapper):
    """
    k线行情及指标绘图器。
    """

    def __init__(self, df):
        super(KlinePlotter, self).__init__()
        self.jqsdk = JQDataCollector()
        self.width = 1600
        self.height = 600
        self.data = df

        self.genStyle = {
            'is_datazoom_show': True,
            'datazoom_type': 'both'
        }

    def getCsvData(self):
        return self.data

    def setCsvData(self, df):
        self.data = df

    def addCloseMa(self, n):
        """
        计算均线，并保存到数据df中
        :param n: int
        :return: pd.Series
        """
        df = self.getCsvData()
        colName = 'close_ma{}'.format(n)
        df[colName] = df['close'].rolling(n).mean()
        df[colName] = df[colName].map(lambda x: round(x, 5))
        return df[colName]

    def addCloseStd(self, n):
        """
        计算标准差，并保存到数据df中
        :param n: int
        :return: pd.Series
        """
        df = self.getCsvData()
        colName = 'close_std{}'.format(n)
        df[colName] = df['close'].rolling(n).std()
        return df[colName]

    def plotKline(self, title=u'k线', addStyle=None):
        """
        返回波动率k线图对象
        :param title:
        :param addStyle: dict
        :return:
        """
        kStyle = self.genStyle.copy()
        if addStyle is not None:
            kStyle.update(addStyle)

        df = self.getCsvData()
        dateList = df.index.tolist()

        gen = df.iterrows()
        ohlc = [[data['open'], data['close'], data['low'], data['high']] for idx_, data in gen]

        kline = Kline(title)
        kline.add(title, dateList, ohlc, tooltip_formatter=kline_tooltip_formatter, **kStyle)
        # kline.render(getTestPath('qvixkline.html'))

        return kline

    def getOverlapWithKline(self, title, **kwargs):
        """
        返回一个已经叠加了k线图的叠加图。
        :param title: string
                叠加图标题
        :return:
        """
        overlap = Overlap(width=self.width, height=self.height)
        kline = self.plotKline(title, **kwargs)
        overlap.add(kline)
        return overlap

    def plotMa(self, title, n1=5, n2=20, n3=60):
        """
        绘制均线系统叠加图
        :param title: str
        :param n1: int
        :param n2: int
        :param n3: int
        :return:
        """
        lineStyle = self.genStyle.copy()
        lineStyle['line_width'] = 2

        overlap = Overlap(width=self.width, height=self.height)
        kline = self.plotKline(title)
        overlap.add(kline)

        df = self.getCsvData()
        dateList = df.index.tolist()

        for ma in [n1, n2, n3]:
            label = u'MA{}'.format(ma)
            line = Line()
            line.add(label, dateList, self.addCloseMa(ma).tolist(), **lineStyle)
            overlap.add(line)

        return overlap

    def plotBollChanel(self, title, n=20):
        """
        绘制boll通道叠加图
        :param title: str
        :param n:
        :return:
        """
        lineStyle = self.genStyle.copy()
        lineStyle['line_width'] = 2

        # title = u'波动率指数Boll通道'
        overlap = Overlap(width=self.width, height=self.height)
        kline = self.plotKline(title)
        overlap.add(kline)

        df = self.getCsvData()
        dateList = df.index.tolist()

        lineMid = Line()
        lineMid.add(u'布林MID', dateList, self.addCloseMa(n).tolist(), **lineStyle)
        overlap.add(lineMid)

        upList = self.addCloseMa(n) + 2 * self.addCloseStd(n)
        upList = upList.tolist()
        lineUp = Line()
        lineUp.add(u'布林UP', dateList, upList, **lineStyle)
        overlap.add(lineUp)

        downList = self.addCloseMa(n) - 2 * self.addCloseStd(n)
        downList = downList.tolist()
        lineDown = Line()
        lineDown.add(u'布林Down', dateList, downList, **lineStyle)
        overlap.add(lineDown)

        return overlap
        # overlap.render(getTestPath('qvixkline_boll.html'))

    def plotAll(self, title, path, filename, item=None):
        """
        按顺序绘制所有的叠加图
        :return:
        """
        page = Page()
        # etf = self.plotEtf()

        if item is None:
            ma = self.plotMa(title)
            boll = self.plotBollChanel(title)

            # page.add(etf)
            page.add(boll)
            page.add(ma)
        else:
            if 'kline' in item:
                kline = self.plotKline(title)
                page.add(kline)
            if 'ma' in item:
                ma = self.plotMa(title, n1=5, n2=10, n3=20)
                page.add(ma)
            if 'boll' in item:
                boll = self.plotBollChanel(title)
                page.add(boll)

        # htmlName = 'qvix_daily.html'
        outputDir = self.jqsdk.getResearchPath(OPTION, path)
        page.render(os.path.join(outputDir, filename))


def kline_tooltip_formatter(params):
    """
    修改原显示格式，添加了时间。
    :param params:
    :return:
    """
    text = (params[0].seriesName + '<br/>' +
            u'- 日期:' + params[0].name + '<br/>' +
            u'- 开盘:' + params[0].data[1] + '<br/>' +
            u'- 收盘:' + params[0].data[2] + '<br/>' +
            u'- 最低:' + params[0].data[3] + '<br/>' +
            u'- 最高:' + params[0].data[4])
    return text


def line_tooltip_formatter(params):
    return params[0].name + ' : ' + params[0].data[1]


def yaxis_formatter(params):
    return params.se + ' percent'


def get_qvix_data():
    """
    获取期权论坛波动率指数日线数据。
    :return:
    """
    try:
        resp = requests.get(QVIX_URL)
        csv_reader = csv.DictReader(resp.iter_lines())
        csv_list = list(csv_reader)
    except:
        print('error occur when get qvix.')
    else:
        name_dict = {
            '1': 'datetime',
            '2': 'open',
            '3': 'high',
            '4': 'low',
            '5': 'close'
        }
        df = pd.DataFrame(csv_list)
        df.rename(columns=name_dict, inplace=True)
        df.dropna(inplace=True)
        df['datetime'] = df['datetime'].map(lambda t_stamp: dateToStr(datetime.fromtimestamp(float(t_stamp) / 1000)))
        for item in ['open', 'high', 'low', 'close']:
            df[item] = df[item].map(lambda str_num: float(str_num))

        df.set_index('datetime', inplace=True)

        filename = 'vixBar.csv'
        fp = os.path.join(getDataDir(), RESEARCH, OPTION, 'qvix', filename)

        df.to_csv(fp)
        print('update completely!')
        return df


if __name__ == '__main__':
    get_qvix_data()
    # import pandas as pd
    #
    # cs01 = pd.DataFrame([1, 2, 3], index=['a', 'b', 'c'])
    # cs02 = pd.DataFrame([5, 6, 7], index=['b', 'c', 'd'], columns=['city'])
    # s02 = cs02['city']
    # cs01['kity'] = s02
    # # cs03 = pd.concat([cs01, s02], axis=1, sort=False)
    # print(cs01)
