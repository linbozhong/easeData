# coding:utf-8

import pandas as pd
import os
import pymongo
import matplotlib.pyplot as plt
import seaborn as sns
import tushare as ts
from datetime import datetime
from collections import OrderedDict
from dateutil.relativedelta import relativedelta
from jqdatasdk import opt, query
from pyecharts import Line
from pyecharts import Page

from database import MongoDbConnector
from base import LoggerWrapper
from collector import JQDataCollector
from functions import (strToDate, dateToStr, getTestPath)
from const import *


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


class PositionDiffPlotter(object):
    """
    期权品种仓差走势图
    """

    def __init__(self):
        self.jqsdk = JQDataCollector()
        self.tradingContractInfo = None
        self.mainContract = None
        self.displayContract = None
        self.contractName = None
        self.dailyPriceDict = OrderedDict()
        self.posDiffDict = OrderedDict()

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
        self.displayExercisePrice = [
            2.5,
            2.3,
            2.35,
        ]

    @staticmethod
    def calcPositionDiff(dailyDf):
        df = dailyDf.set_index('date')
        df['prePosition'] = df['position'].shift(1)
        df['positionDiff'] = df['position'] - df['prePosition']
        df['positionDiff'].fillna(0, inplace=True)
        return df['positionDiff'].map(int)

    def getTradingContract(self):
        if self.tradingContractInfo is None:
            date = self.date
            exchange = 'XSHG'
            db = opt.OPT_DAILY_PREOPEN
            q = query(db).filter(db.exchange_code == exchange, db.date == date)
            self.tradingContractInfo = self.jqsdk.run_query(q)
        return self.tradingContractInfo

    def getMainContract(self):
        if self.mainContract is None:
            df = self.getTradingContract()
            filterWord = self.queryMonth + 'M'
            filterFunc = lambda x: filterWord in x
            mask1 = df.trading_code.map(filterFunc)
            mask2 = df.exercise_price.isin(self.queryExercisePrice)
            df = df[mask1 & mask2]
            df = df.sort_values(by='name')
            self.mainContract = df
        return self.mainContract

    def getContractName(self, contractCode):
        if self.contractName is None:
            df = self.getMainContract()
            df = df.set_index('code')
            self.contractName = df['name']
        return self.contractName[contractCode]

    def getDisplayContract(self):
        if self.displayContract is None:
            df = self.getMainContract()
            mask = df.exercise_price.isin(self.displayExercisePrice)
            self.displayContract = df[mask]
        return self.displayContract

    def getDailyPrice(self, contract):
        db = opt.OPT_DAILY_PRICE
        q = query(db).filter(db.code == contract)
        df = self.jqsdk.run_query(q)
        return df

    def getMainContractDailyPrice(self):
        if not self.dailyPriceDict:
            mainContract = self.getMainContract().code
            for contract in mainContract:
                self.dailyPriceDict[contract] = self.getDailyPrice(contract)
        return self.dailyPriceDict

    def getPositonDiff(self):
        price = self.getMainContractDailyPrice()
        for code, priceDf in price.items():
            self.posDiffDict[code] = self.calcPositionDiff(priceDf)
        df = pd.DataFrame(self.posDiffDict)
        df.fillna(0, inplace=True)
        return df

    def plotPosDiff(self):
        df = self.getPositonDiff()
        # df = pd.read_csv(getTestPath('posDiff.csv'), index_col=0)
        xtickLabels = df.index.tolist()

        width = 1600

        page = Page()

        # 对比图
        lineList = []
        multiLine = Line(u'{}期权合约仓差对比走势图'.format(self.queryMonth), width=width)
        displayCode = self.getDisplayContract().code.values.tolist()
        for code, series in df.iteritems():
            singleLine = Line(u'{}仓差走势'.format(self.getContractName(code)), width=width)
            singleLine.add(self.getContractName(code), xtickLabels, series.values.tolist())
            lineList.append(singleLine)
            if code in displayCode:
                multiLine.add(self.getContractName(code), xtickLabels, series.values.tolist())

        page.add(multiLine)
        for line in lineList:
            page.add(line)

        page.render(getTestPath('posDiff.html'))


