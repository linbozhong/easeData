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


class PositionDiffPlotter(LoggerWrapper):
    """
    期权品种仓差走势图
    """

    def __init__(self):
        super(PositionDiffPlotter, self).__init__()
        self.jqsdk = JQDataCollector()
        self.tradingContractInfo = None
        self.mainContract = None
        self.displayContract = None
        self.contractName = None
        self.dailyPriceDict = OrderedDict()
        self.posDiffDf = None

        self.isExcludeAdjusted = True
        self.underlyingSymbol = '510050.XSHG'
        self.exchange = 'XSHG'
        self.date = datetime(2018, 12, 26)
        self.queryMonth = '1812'
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
            2.5,
            2.3,
            2.35,
        ]

    @staticmethod
    def calcPositionDiff(dailyDf):
        """
        从日行情DataFrame计算仓差。
        :param dailyDf:
        :return:
        """
        df = dailyDf.set_index('date')
        df['prePosition'] = df['position'].shift(1)
        df['positionDiff'] = df['position'] - df['prePosition']
        df['positionDiff'].fillna(0, inplace=True)
        return df['positionDiff'].map(int)

    def getTradingContract(self):
        """
        获取某个月份的交易合约。
        :return:
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
            print(df.name.values[0], type(df.name.values[0]))
        return self.tradingContractInfo

    def getContractName(self, contractCode):
        """
        获取期权合约的中文名。
        :param contractCode: string
                期权代码
        :return:
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
        :return:
        """
        db = opt.OPT_DAILY_PRICE
        q = query(db).filter(db.code == contract)
        df = self.jqsdk.run_query(q)
        return df

    def getDisplayContract(self):
        """
        仅显示设定的部分合约。
        :return:
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
        :return:
        """
        if self.displayContract is None:
            df = self.getDisplayContract()
            mask = df.exercise_price.isin(self.compExercisePrice)
            self.displayContract = df[mask]
        return self.displayContract

    def getAllContractDailyPrice(self):
        """
        获取该月份的所有合约的日线数据，并保存为字典。
        :param args:
        :param kwargs:
        :return:
        """
        if not self.dailyPriceDict:
            contracts = self.getTradingContract().code
            for contract in contracts:
                self.dailyPriceDict[contract] = self.getDailyPrice(contract)
        return self.dailyPriceDict

    def getGoupedCode(self):
        """
        获取按行权价分组的期权代码。
        :return:
        """
        df = self.getTradingContract()
        df = df.sort_values(by='exercise_price')
        grouped = df.groupby('exercise_price')
        groupCode = OrderedDict()
        for exercisePrice, df in grouped:
            groupCode[exercisePrice] = df['code'].tolist()
        return groupCode

    def getPositonDiff(self):
        """
        获取所有的仓差数据。
        :return:
        """
        if self.posDiffDf is None:
            price = self.getAllContractDailyPrice()
            posDiffDict = dict()
            for code, priceDf in price.items():
                posDiffDict[code] = self.calcPositionDiff(priceDf)
            df = pd.DataFrame(posDiffDict)
            df.fillna(0, inplace=True)
            self.posDiffDf = df
        return self.posDiffDf

    @staticmethod
    def nameToColor(name):
        """
        通过期权名称获取要使用的颜色
        :return:
        """
        sell = u'沽'
        buy = u'购'
        color = 'null'

        if not isinstance(name, unicode):
            name = name.decode('utf-8')

        if sell in name:
            color = 'red'
        elif buy in name:
            color = 'blue'

        return color

    def plotPosDiff(self):
        """
        绘制并输出仓差走势图。
        :return:
        """

        df = self.getPositonDiff()
        groupCode = self.getGoupedCode()
        # df = pd.read_csv(getTestPath('posDiff.csv'), index_col=0)
        xtickLabels = df.index.tolist()
        width = 1600

        page = Page()

        # 对比组图
        multiLine = Line(u'期权仓差对比走势图', width=width)
        for price in self.compExercisePrice:
            codeList = groupCode[price]
            for code in codeList:
                multiLine.add(self.getContractName(code), xtickLabels, df[code].values.tolist())
        page.add(multiLine)

        # 行权价组图
        lineList = []
        for exercisePrice, codeList in groupCode.items():
            line = Line(u'行权价{}仓差走势'.format(str(int(exercisePrice * 1000))), width=width)
            for code in codeList:
                line.add(self.getContractName(code), xtickLabels, df[code].values.tolist())
            lineList.append(line)
        for line in lineList:
            page.add(line)

        page.render(getTestPath('posDiff{}.html'.format(self.queryMonth)))


class SellBuyRatioPlotter(LoggerWrapper):
    """
    期权沽购比走势图
    """

    def __init__(self):
        super(SellBuyRatioPlotter, self).__init__()
        self.jqsdk = JQDataCollector()
        self.contractTypeDict = None
        self.underlyingSymbol = '510050.XSHG'

    def getContractInfo(self):
        filename = u'{}_{}.csv'.format(OPTION, BASIC)
        path = os.path.join(self.jqsdk.getBasicPath(OPTION), filename)
        df = pd.read_csv(path, index_col=0)
        mask = (df.underlying_symbol == self.underlyingSymbol)
        df = df[mask]
        return df

    def getContractType(self, code):
        """
        合约代码到看涨看跌的映射
        :param code:
        :return:
        """
        if self.contractTypeDict is None:
            df = self.getContractInfo()
            df = df.set_index('code')
            self.contractTypeDict = df['contract_type'].to_dict()
        return self.contractTypeDict.get(code, 'NO')

    def calcRatioByDate(self, fp):
        """
        计算单日的沽购比值
        :param fp: filepath
                单日价格文件
        :return:
        """
        df = pd.read_csv(fp, index_col=0)

        df['contract_type'] = df.code.map(self.getContractType)
        df.to_csv(getTestPath('ratiosource.csv'))

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

    def getRatio(self):
        filename = 'sb_ratio_data.csv'
        root = self.jqsdk.getPricePath(OPTION, DAILY)
        path = os.path.join(root, filename)
        if not os.path.exists(path):
            fps = [os.path.join(root, fn) for fn in os.listdir(root)]
            data = []
            for fp in fps:
                self.info(u'计算{}'.format(fp))
                data.append(self.calcRatioByDate(fp))
            df = pd.DataFrame(data)
            df.to_csv(path, encoding='utf-8-sig')
        else:
            df = pd.read_csv(path, index_col=0)

        df = df.set_index('date')
        start = '2015-02-09'
        end = df.index[-1]
        underlyingPrice = self.jqsdk.get_price(self.underlyingSymbol, start_date=start, end_date=end)
        underlyingPrice.index = underlyingPrice.index.map(dateToStr)
        close = underlyingPrice['close']
        df[self.underlyingSymbol] = close
        return df

    def plotRatio(self):
        df = self.getRatio()
        xtickLabels = df.index.tolist()
        width = 1600

        nameDict = {'moneyRatio': u'沽购金额比',
                    'volumeRatio': u'沽购成交比',
                    'positionRatio': u'沽购持仓比',
                    self.underlyingSymbol: u'50ETF'
                    }

        page = Page()

        # 对比组图
        multiLine = Line(u'沽购比走势图', width=width)
        for name, series in df.iteritems():
            multiLine.add(nameDict[name], xtickLabels, series.values.tolist(), is_datazoom_show=True,
                          datazoom_type='both')
        page.add(multiLine)

        page.render(getTestPath('ratioTrend.html'))
