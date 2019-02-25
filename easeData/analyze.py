# coding:utf-8

import numpy as np
import pandas as pd
import os
import pymongo
import matplotlib.pyplot as plt
import seaborn as sns
from copy import copy
from datetime import datetime, timedelta
from collections import OrderedDict
from dateutil.relativedelta import relativedelta

import tushare as ts
from jqdatasdk import opt, query

from pyecharts import Line, Kline, Bar
from pyecharts import Page, Overlap

from database import MongoDbConnector
from base import LoggerWrapper
from collector import JQDataCollector
from functions import (strToDate, dateToStr, getTestPath, getDataDir, roundFloat)
from const import *
from text import *


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
            2.25,
            2.3,
            2.35,
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

    def setAtmStart(self, date):
        self.atmStart = date

    def setAtmEnd(self, date):
        self.atmEnd = date

    def getAtmContract(self, fp, keepCurrent=False):
        """
        获取某个交易日的平值期权合约数据。
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
        return atmDf

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

        contractDf = copy(self.getContractInfo())   # 避免修改操作影响到原缓存文件
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
            d['entry'], d['exit'] = roundFloat(entry*10000, 1), roundFloat(exit_*10000, 1)
            recList.append(d)
        df = pd.DataFrame(recList)
        df['return'] = (df['entry'] - df['exit']).map(lambda x: round(x, 1))
        df['returnRate'] = (df['return'] / df['entry']).map(lambda x: round(x, 3))
        return df

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

        df['positionRatio_ma5'] = df['positionRatio'].rolling(5).mean()
        df['positionRatioToMa5'] = df['positionRatio'] / df['positionRatio_ma5']
        df['positionRatioToMa5'] = df['positionRatioToMa5'].map(f)
        df['moneyRatio_ma5'] = df['moneyRatio'].rolling(5).mean()
        df['moneyRatioToMa5'] = df['moneyRatio'] / df['moneyRatio_ma5']
        df['moneyRatioToMa5'] = df['moneyRatioToMa5'].map(f)
        return df

    def plotRatio(self):
        """
        绘制沽购各项指标走势图。
        :return:
        """
        width = 1500
        height = 600
        displayItem = ['volumeRatio', self.underlyingSymbol, 'moneyRatio']

        if self.isOnlyNearby:
            title = u'50et期权近月沽购比'
            htmlName = 'ratio_nearby.html'
        else:
            title = u'50etf期权全月份沽购比'
            htmlName = 'ratio_all_month.html'

        nameDict = {'moneyRatio': u'沽购成交金额比',
                    'volumeRatio': u'沽购成交量比',
                    'positionRatio': u'沽购持仓量比',
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
            filename = 'qvix_daily.csv'
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



if __name__ == '__main__':
    import pandas as pd

    cs01 = pd.DataFrame([1, 2, 3], index=['a', 'b', 'c'])
    cs02 = pd.DataFrame([5, 6, 7], index=['b', 'c', 'd'], columns=['city'])
    s02 = cs02['city']
    cs01['kity'] = s02
    # cs03 = pd.concat([cs01, s02], axis=1, sort=False)
    print(cs01)