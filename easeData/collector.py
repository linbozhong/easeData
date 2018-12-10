# coding:utf-8

import re
import os
import jqdatasdk
import pymongo
import pandas as pd
import numpy as np
from collections import OrderedDict
from jaqs.data import DataApi
from os.path import abspath, dirname
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from functools import wraps

from const import *
from text import *
from base import DataVendor
from functions import (loadSetting, saveSetting, getParentDir,
                       rmDateDash, strToDate, dateToStr, dateToDtStr, getTodayStr
                       )
from database import MongoDbConnector, VnpyAdaptor


class DataCollector(DataVendor):
    """
    数据采集器基类
    """

    def __init__(self):
        super(DataCollector, self).__init__()
        self._dbAdaptor = None
        self._isConnected = False
        self._sdk = None
        self._unpopularFuture = ['wr', 'fb', 'bb', 'RS', 'RI', 'LR', 'JR', 'WH', 'PM', 'CY', 'TS']

    def connectApi(self, user=None, token=None, address=None):
        raise NotImplementedError

    def setUnpopularFuture(self, unpopularList):
        self._unpopularFuture = unpopularList

    def setDbAdaptor(self, adaptor):
        self._dbAdaptor = adaptor

    def getDbAdaptor(self):
        if self._dbAdaptor is None:
            self.setDbAdaptor(VnpyAdaptor())
        return self._dbAdaptor


class JQDataCollector(DataCollector):
    """
    聚宽数据采集类
    """

    EXCHANGE_MAP = {
        EXCHANGE_SSE: 'XSHG',  # 上交所
        EXCHANGE_SZSE: 'XSHE',  # 深交所
        EXCHANGE_CFFEX: 'CCFX',  # 中金所
        EXCHANGE_SHFE: 'XSGE',  # 上期所
        EXCHANGE_CZCE: 'XZCE',  # 郑商所
        EXCHANGE_DCE: 'XDCE',  # 大商所
        EXCHANGE_INE: 'XINE'  # 上海国际能源交易中心
    }

    def __init__(self):
        super(JQDataCollector, self).__init__()
        self.vendor = VENDOR_JQ
        self.today = None

        self._sdk = jqdatasdk
        self._dailyCount = 0
        self._dailyCountCollection = None
        self._jqDominantDf = None

        self.futureExchangeMap = None
        self.dominantContinuousSymbolMap = None
        self.jqSymbolMap = {}

        self._initDailyCount()
        self._syncDailyCount()
        self.connectApi()

    def __getattr__(self, name):
        # 代理访问logger的方法
        loggerMethod = ['debug', 'info', 'warn', 'error', 'critical']
        if name in loggerMethod:
            return getattr(self._logger, name)

        # 代理访问sdk的方法
        jqGetMethod = [i for i in self._sdk.__dict__ if i.startswith('get_')]
        if name in jqGetMethod:
            return self._runFunc(name)

    def _runFunc(self, name):
        """
        获取sdk方法
        ----------------------------------------------------------------------------------------------------------------
        :param name: string
                jqdatasdk的方法名
        :return:
        """

        def wrapper(*args, **kwargs):
            if self._isConnected:
                func = getattr(self._sdk, name)
                df = func(*args, **kwargs)
                self._addDailyCount(len(df))
                return df
            else:
                self.error(API_NOT_CONNECTED)

        return wrapper

    # def dataChecker(self, func):
    #     """
    #     检查获取的数据的装饰器
    #     :param func:
    #     :return:
    #     """
    #     @wraps(func)
    #     def wrapper(*args, **kwargs):
    #         try:
    #             df = func(*args, **kwargs)
    #             if df is not None and not df.empty:
    #                 return df
    #         except Exception as e:
    #             msg = u"{}:{}".format(ERROR_UNKNOWN, e.message.decode('gb2312'))
    #             self.error(msg)
    #     return wrapper

    def _initDailyCount(self):
        """
        初始化数据库。因为jqdata免费版有每日100万条记录的限制，这里用数据库来记录每日已使用的数据量。
        ----------------------------------------------------------------------------------------------------------------
        :return:
        """
        client = MongoDbConnector().connect()
        db = client['JQData_setting']
        collection = db['dailyCount']
        if not collection.index_information():
            collection.create_index([('date', pymongo.ASCENDING)], unique=True)
        self._dailyCountCollection = collection

        today = datetime.today()
        self.today = datetime(today.year, today.month, today.day)  # 将时间信息重置为0

    def _syncDailyCount(self):
        """
        同步数据库中本日已使用的数据量。
        ----------------------------------------------------------------------------------------------------------------
        :return:
        """
        flt = {'date': self.today}
        res = self._dailyCountCollection.find_one(flt)
        if res is not None:
            self._dailyCount = res['dailyCount']
            self.info(u"{}:{}".format(JQ_SYNC_SUCCEED, self._dailyCount))
        else:
            doc = {'date': self.today, 'dailyCount': self._dailyCount}
            self._dailyCountCollection.replace_one(flt, doc, upsert=True)

    def _addDailyCount(self, num):
        """
        把本次使用的数据量和本日已使用的数据量累加。
        ----------------------------------------------------------------------------------------------------------------
        :param num: int
                    本次获取的数据条数
        :return:
        """
        self._dailyCount += num
        self._dailyCountCollection.update_one({'date': self.today}, {'$inc': {'dailyCount': num}})
        self.info(u"{}:{},{}:{}".format(JQ_THIS_COUNT, num, JQ_TODAY_COUNT, self._dailyCount))

    def _getFutureBasic(self, date=None):
        """
        获取期货品种基本数据。
        ----------------------------------------------------------------------------------------------------------------
        :return:
        """
        if date is None:
            date = self.today
        df = self.get_all_securities('futures', date)
        if df is not None:
            self.futureExchangeMap = self.getExchange(df.index)  # 保存品种代码和交易所代码的字典
            # 筛选出jq主力连续合约的记录
            fltFunc = lambda x: '9999' in x
            mask = df.index.map(fltFunc)
            self._jqDominantDf = df[mask]
            self.dominantContinuousSymbolMap = {self.getUnderlyingSymbol(symbol): symbol for symbol in
                                                self._jqDominantDf.index}

    def _monthToRange(self, year, month):
        """
        输入年份和月份，得到月份的开始日期和结束日期，用于查询行情。
        开始日期从2005年开始，如果结束日期于或大于当前日期，结束日期就是当前日期。
        ----------------------------------------------------------------------------------------------------------------
        :param year: int
                年份，从2005年开始
        :param month: int
                月份
        :return: Tuple(datetime, datetime)
                开始日期和结束日期构成的tuple
        """
        curYear = self.today.year
        curMonth = self.today.month
        year = year if year >= 2005 else 2005
        begin = datetime(year, month, 1)
        isFutureMonth = (year > curYear) or (year == curYear and month >= curMonth)
        if isFutureMonth:
            begin = datetime(curYear, curMonth, 1)
            end = self.today
        else:
            end = begin + relativedelta(months=1)
        return begin, end

    def connectApi(self, user=None, token=None, address=None):
        """
        进行JQData的用户认证
        ----------------------------------------------------------------------------------------------------------------
        :param user: string
                聚宽用户名
        :param token: string
                聚宽密码
        :param address: string
                聚宽地址，默认不输入
        :return:
        """
        if self._isConnected is False:
            settingFile = os.path.join(dirname(abspath(__file__)), FILE_SETTING)
            setting = loadSetting(settingFile)
            if user is None:
                user = setting['jq_user']
            if token is None:
                token = setting['jq_token']
            try:
                jqdatasdk.auth(user, token)
                self._isConnected = True
                self.info(u'{}:{}'.format(API_SUCCEED, self.vendor))
            except Exception as e:
                msg = e.message.decode('gb2312')
                msg = u"{}:{}".format(API_FAILED, msg)
                self.error(msg)
        else:
            self.info(API_IS_CONNECTED)

    def convertFutureSymbol(self, symbol):
        """
        把普通期货合约代码转换为jqdata规则的代码，并做缓存。
        ----------------------------------------------------------------------------------------------------------------
        :param symbol: string
                普通期货合约代码
        :return: string
                聚宽的期货合约代码
        """
        if self.jqSymbolMap.get(symbol) is None:
            exMap = self.getFutureExchangeMap()
            exSymbol = exMap[self.getUnderlyingSymbol(symbol).upper()]
            self.jqSymbolMap[symbol] = '.'.join([symbol.upper(), exSymbol])
        return self.jqSymbolMap[symbol]

    def convertStockSymbol(self, symbol):
        pass

    def getFutureExchangeMap(self):
        """
        获取期货品种对应jqdata交易所代码的字典。
        ----------------------------------------------------------------------------------------------------------------
        :return: dict
            字典示例数据：{'Y': 'XDCE', 'FU': 'XSGE', ...}
        """
        if self.futureExchangeMap is None:
            self._getFutureBasic()
        return self.futureExchangeMap

    def getDominantContinuousSymbolMap(self):
        """
        获取期货品种对应jqdata主力连续合约代码的字典。
        ----------------------------------------------------------------------------------------------------------------
        :return: dict
            字典示例数据：{'Y': 'Y9999.XDCE', 'FU': 'FU9999.XSGE', ...}
        """
        if self.dominantContinuousSymbolMap is None:
            self._getFutureBasic()
        return self.dominantContinuousSymbolMap

    def getPopularFuture(self, unpopular=None):
        """
        获取当前热门的期货合约品种。
        ----------------------------------------------------------------------------------------------------------------
        :param unpopular: iterable container
                冷门的期货品种，用于被排除
        :return: list
                热门期货品种列表
        """
        if unpopular is None:
            unpopular = self._unpopularFuture
        shfe = self.EXCHANGE_MAP[EXCHANGE_SHFE]
        dce = self.EXCHANGE_MAP[EXCHANGE_DCE]
        ine = self.EXCHANGE_MAP[EXCHANGE_INE]
        exMap = self.getFutureExchangeMap()
        varieties = []
        for variety, exchange in exMap.items():
            if exchange in [shfe, dce, ine]:
                varieties.append(variety.lower())
            else:
                varieties.append(variety)
        return [variety for variety in varieties if variety not in unpopular]

    def downloadContinuousDaily(self, symbol, **kwargs):
        """
        下载期货主力连续合约的日线数据。
        ----------------------------------------------------------------------------------------------------------------
        :param symbol: string
                期货合约
        :param kwargs:
        :return:
        """
        filename = u'{}.csv'.format(symbol)
        path = os.path.join(self.getPricePath(FUTURE, DAILY, symbol), filename)
        jqSymbol = self.convertFutureSymbol(symbol)
        start = strToDate('2005-01-01')
        end = self.today

        if not os.path.exists(path):
            try:
                df = self.get_price(jqSymbol, start_date=start, end_date=end, **kwargs)
                if not df.empty:
                    df.dropna(inplace=True)
                    df.to_csv(path, encoding='utf-8-sig')
                    self.info(u'{}:{}'.format(FILE_DOWNLOAD_SUCCEED, filename))
                else:
                    self.info(u'{}:{}'.format(DATA_IS_NONE, filename))
            except Exception as e:
                msg = u"{}:{}".format(ERROR_UNKNOWN, e.message.decode('gb2312'))
                self.error(msg)
        else:
            df_file = pd.read_csv(path, encoding='utf-8-sig', index_col=0, parse_dates=True)
            lastDate = df_file.index[-1].to_pydatetime()
            if lastDate == end:
                self.info(u'{}:{}'.format(FILE_IS_NEWEST, symbol))
                return
            start = lastDate + timedelta(days=1)
            df_inc = self.get_price(jqSymbol, start_date=start, end_date=end)
            if not df_inc.empty:
                df_inc.dropna(inplace=True)
                df_new = pd.concat([df_file, df_inc])
                df_new.to_csv(path, encoding='utf-8-sig')
                self.info(u'{}:{}'.format(FILE_UPDATE_SUCCEED, symbol))
            else:
                self.info(u'{}:{}'.format(DATA_IS_NONE, symbol))

    def downloadAllContinuousDaily(self, **kwargs):
        for variety in self.getPopularFuture():
            variety += '9999'
            self.downloadContinuousDaily(variety, **kwargs)

    def downloadContinuousBarByMonth(self, symbol, year, month, overwrite=False, skipThisMonth=False):
        """
        按月获取单个期货品种的主力连续1分钟线数据。不包含运行当天的数据。
        ----------------------------------------------------------------------------------------------------------------
        :param symbol: string
                期货合约代码
        :param year: int
                年份，从2005年开始
        :param month: int
                月份
        :param overwrite: bool
                是否覆盖本地已经存在的csv文件
        :param skipThisMonth: bool
                是否忽略本月份数据（在当月运行无法获取完整的当月数据，所以该选项可以设置不下载本月数据，防止出现没有完整数据的文件）
        :return:
        """
        if skipThisMonth:
            curYear, curMonth = self.today.year, self.today.month
            if year == curYear and month == curMonth:
                self.info(JQ_IGNORE_THIS_MONTH)
                return

        filename = u'{}_{:0>4d}_{:0>2d}.csv'.format(symbol, year, month)
        path = os.path.join(self.getPricePath(FUTURE, BAR, symbol), filename)
        if not overwrite:
            if os.path.exists(path):
                self.info(u'{}:{}'.format(FILE_IS_EXISTED, filename))
                return

        jqSymbol = self.convertFutureSymbol(symbol)
        begin, end = self._monthToRange(year, month)
        # jq查询是包含首尾数据的，结束要减去1分钟，防止上期所金属类包含下个月第一条记录。
        end = end - timedelta(minutes=1)
        try:
            df = self.get_price(jqSymbol, start_date=begin, end_date=end, frequency='1m')
            if not df.empty:
                df.dropna(inplace=True)
                df.index.name = 'datetime'
                df.to_csv(path, encoding='utf-8-sig')
                self.info(u'{}:{}'.format(FILE_DOWNLOAD_SUCCEED, filename))
            else:
                self.info(u'{}:{}'.format(DATA_IS_NONE, filename))
        except Exception as e:
            msg = u"{}:{}".format(ERROR_UNKNOWN, e.message.decode('gb2312'))
            self.error(msg)

    def downloadAllContinuousBarByMonth(self, year, month, varieties=None, **kwargs):
        """
        按月下载所有活跃的期货品种的主力连续1分钟数据。
        ----------------------------------------------------------------------------------------------------------------
        :param year: int
                年份，从2005年开始
        :param month: int
                月份
        :param varieties: list
                指定要下载的期货品种
        :return:
        """
        if varieties is None:
            varieties = self.getPopularFuture()
        for variety in varieties:
            variety += '9999'
            self.downloadContinuousBarByMonth(variety, year, month, **kwargs)

    def downloadAllContinuousBarByRange(self, start, end, **kwargs):
        """
        按时间范围下载所有活跃的期货品种的主力连续1分钟数据，开始日期和结束日期的月份都包含在内。
        ----------------------------------------------------------------------------------------------------------------
        :param start: str or datetime-like
                开始日期
        :param end: str or datetime-like
                结束日期
        :return:
        """
        dateRange = pd.date_range(start, end, freq='MS')
        for date in dateRange:
            self.downloadAllContinuousBarByMonth(date.year, date.month, **kwargs)

    def updateCsvContinuousBar(self, symbol):
        """
        对单个期货品种的主力连续合约的1分钟数据（当前月份的数据）进行增量更新。
        虽然可以更新到最新时间的数据，但是考虑到主力连续合约的数据主要用于研究，而不是交易，因此增量更新到当日白天收盘后的数据，可以节约资源。
        ----------------------------------------------------------------------------------------------------------------
        :param symbol: string
                期货合约代码
        :return:
        """
        curYear, curMonth = self.today.year, self.today.month
        filename = u'{}_{:0>4d}_{:0>2d}.csv'.format(symbol, curYear, curMonth)
        path = os.path.join(self.getPricePath(FUTURE, BAR, symbol), filename)
        if not os.path.exists(path):
            self.downloadContinuousBarByMonth(symbol, curYear, curMonth)  # 不存在本月文件直接下载。
        else:
            df_file = pd.read_csv(path, encoding='utf-8-sig', index_col=0, parse_dates=True)
            try:
                lastTime = df_file.index[-1].to_pydatetime()
            except IndexError:
                # 如果index错误，表示直到现在还没有数据，文件数据最新时间改成上个月最后1分钟。
                lastTime = datetime(curYear, curMonth, 1) - timedelta(minutes=1)
            targetTime = self.today.replace(hour=15, minute=0)
            if lastTime >= targetTime:
                self.info(u'{}:{}'.format(FILE_IS_NEWEST, symbol))
            else:
                jqSymbol = self.convertFutureSymbol(symbol)
                start = lastTime + timedelta(minutes=1)  # 聚宽获取bar数据的方法是包含首尾的,所以lasttime要加1
                df_inc = self.get_price(jqSymbol, start_date=start, end_date=targetTime, frequency='1m')
                if not df_inc.empty:
                    df_inc.dropna(inplace=True)  # 防止盘中更新时，引入jq预设的空数据。
                    df_new = pd.concat([df_file, df_inc])
                    df_new.to_csv(path, encoding='utf-8-sig')
                    self.info(u'{}:{}'.format(FILE_UPDATE_SUCCEED, symbol))
                else:
                    self.info(u'{}:{}'.format(DATA_IS_NONE, symbol))
                    # 存在一个问题，最后一天的夜盘数据这个方法是没办法更新的，后面再解决。

    def updateAllCsvContinuousBar(self):
        """
        对所有活跃期货品种的主力连续合约的1分钟数据（当前月份的数据）进行增量更新。
        ----------------------------------------------------------------------------------------------------------------
        :return:
        """
        for variety in self.getPopularFuture():
            variety += '9999'
            self.updateCsvContinuousBar(variety)

    def updateDb(self, freq, symbol=None):
        """
        更新单个合约的日线或1分钟线价格序列。
        ----------------------------------------------------------------------------------------------------------------
        :param freq: string
                时间周期，‘bar’或‘daily’
        :param symbol: string
                期货合约代码或股票代码
        :return:
        """
        adaptor = self.getDbAdaptor()
        adaptor.setFreq(freq)
        adaptor.setActiveConverter(self.vendor + 'Converter')
        col = adaptor.getCollection(symbol)
        cursor = col.find().sort('datetime', pymongo.DESCENDING).limit(1)
        lastTime = list(cursor)[0]['datetime']

        if freq == DAILY:
            start = lastTime + timedelta(days=1)
            end = self.today - timedelta(days=1)
            queryFreq = 'daily'
        else:
            start = lastTime + timedelta(minutes=1)
            end = self.today.replace(hour=15)
            queryFreq = '1m'

        df = self.get_price(self.convertFutureSymbol(symbol), start_date=start, end_date=end, frequency=queryFreq)
        df.dropna(inplace=True)
        df['datetime'] = df.index
        df['datetime'] = df['datetime'].map(dateToDtStr)

        adaptor.dfToDb(df, symbol)
        self.info(u'{}:{}:{}'.format(DB_UPDATE_COMPLETE, freq, symbol))

    def updateAllDb(self, freq, symbols=None, exclude=None):
        """
        更新多个合约的数据库数据。
        ----------------------------------------------------------------------------------------------------------------
        :param freq: string
                时间周期，‘bar’或‘daily’
        :param symbols: iterable
                代码集合
        :param exclude: iterable
                排除的代码集合
        :return:
        """
        adaptor = self.getDbAdaptor()
        adaptor.setFreq(freq)
        cols = adaptor.getDb().collection_names()
        if symbols is not None:
            symbols = [symbol for symbol in symbols if symbol in cols]
        else:
            symbols = cols
        for symbol in symbols:
            if exclude and symbol in exclude:
                continue
            self.updateDb(freq, symbol)


class RQDataCollector(DataCollector):
    """
    米筐数据采集器
    目前仅完成从米筐研究模块下载数据文件并进行解析的功能，封装米筐的api尚未开发。
    在米筐研究模块中运行本项目的rqData.py内的代码，即可下载数据文件。
    """

    def __init__(self):
        super(RQDataCollector, self).__init__()
        self.vendor = VENDOR_RQ
        self.requiredDataDir = os.path.join(getParentDir(), DIR_EXTERNAL_DATA, self.vendor)

    def connectApi(self, user=None, token=None, address=None):
        # 尚未实现
        pass

    def getRequiredFilePath(self, filename):
        return os.path.join(self.requiredDataDir, filename)

    def getFutureMainContractDate(self, filePath=None):
        """
        从csv文件中获取期货品种对应的历史主力合约日期区间，需要先从米筐的研究模块下载所需的数据。
        ----------------------------------------------------------------------------------------------------------------
        :param filePath: string.
                csv文件路径
        :return: dict{string: [(string, string, string)]}
                    key: 期货品种合约，如‘rb’
                    value: 列表，元素为3项的tuple，分别是（期货合约代码, 开始日期, 结束日期）
                ---------------------------------------------------------
                数据范例：
                {
                    'rb': [('rb1801, '2018-01-01', '2018-04-01'),
                           (...),
                           ...]
                    'cu': ...
                    ...
                }
        """

        if filePath is None:
            filePath = self.getRequiredFilePath('main_contract_history.csv')
        mainContractMap = dict()
        df = pd.read_csv(filePath, encoding='utf-8', low_memory=False, index_col=[0])
        lastDay = df.index.sort_values()[-1]
        columns = df.iteritems()
        for colName, colData in columns:
            if colName not in ['exchange']:
                colData.dropna(inplace=True)
                colData.sort_index(inplace=True)
                colIterator = colData.iteritems()

                def getPeriod(iterator):
                    """
                    遍历迭代器并获取日期区间的函数
                    :param iterator:
                            迭代器，格式范例：[('2018-01-03', 'rb1801'), ('2018-04-05', 'rb1805'), ...]
                    :return: list[tuple(string: symbol, string: begin-date, string: end-date)]
                            列表，元素为3项的tuple
                            格式范例：[('rb1801', '2018-01-03', '2018-05-01'), ('rb1805',.., ..), ...]
                    """
                    res = []
                    oldSymbol = ''
                    begin = ''
                    end = ''
                    for date, symbol in iterator:
                        if colName.islower():
                            symbol = symbol.lower()
                        # 遍历的时候如果上下合约不一致，说明开始了新的主力合约
                        if symbol != oldSymbol:
                            # 如果是第一条记录，不添加，因为即使某个主力合约只维持一天，也要遍历到第二天的时候才会知道。
                            if oldSymbol != '':
                                res.append((oldSymbol, begin, end))
                            oldSymbol = symbol
                            begin = date
                            end = date if date != lastDay else ''
                        # 如果上下合约一致，只需要把结束日期改为当前记录的日期
                        else:
                            end = date if date != lastDay else ''
                    # 添加最后一个记录
                    res.append((oldSymbol, begin, end))
                    return res

                result = getPeriod(colIterator)
                mainContractMap[colName] = result
        return mainContractMap

    def getCurrentMainContract(self, filePath=None):
        """
        从json文件得到期货品种当前主力合约代码的数据（有序字典）。需要先从米筐的研究模块下载所需的数据，需要定期更新。
        ----------------------------------------------------------------------------------------------------------------
        :param filePath: string
                json文件路径
        :return: OrderedDict {string: string}
                    key: 品种代码, 如'rb'
                    value: 主力合约代码, 如'rb1810'
        """
        if filePath is None:
            filePath = self.getRequiredFilePath('current_main_contract.json')
        return loadSetting(filePath, object_pairs_hook=OrderedDict)


class JaqsDataCollector(DataCollector):
    """
    Jaas数据收集类，通过jaqs的Api获取各类金融数据，数据可以保存到csv文件或者数据库。
    数据是从2012年开始的。
    --------------------------------------------------------------------------------------------------------------------
    """
    # 定义Jaqs交易所代码的字典
    EXCHANGE_MAP = {
        EXCHANGE_SSE: 'SH',  # 上交所
        EXCHANGE_SZSE: 'SZ',  # 深交所
        EXCHANGE_CFFEX: 'CFE',  # 中金所
        EXCHANGE_SHFE: 'SHF',  # 上期所
        EXCHANGE_CZCE: 'CZC',  # 郑商所
        EXCHANGE_DCE: 'DCE',  # 大商所
        EXCHANGE_INE: ''  # 上海国际能源交易中心
    }

    # jaqs预设的市场类别
    VIEW_INSTRUMENT_INFO = 'jz.instrumentInfo'
    VIEW_TRADING_DAY_INFO = 'jz.secTradeCal'
    VIEW_INDEX_INFO = 'lb.indexInfo'
    VIEW_CONSTITUENT_OF_INDEX = 'lb.indexCons'
    VIEW_INDUSTRY_INFO = 'lb.secIndustry'
    VIEW_SUSPEND_STOCK = 'lb.secSusp'

    # 定义jaqs的inst_type
    INST_TYPE_STOCK = (1,)
    INST_TYPE_FUND = (2, 3, 4, 5)
    INST_TYPE_FUTURE_BASIC = (6, 7)
    INST_TYPE_BOND = (8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20)
    INST_TYPE_ABS = (16,)
    INST_TYPE_INDEX = (100,)
    INST_TYPE_FUTURE = (101, 102, 103)
    INST_TYPE_OPTION = (201, 202, 203)

    # 定义要输出的列
    ALL_FIELD_INSTRUMENT = ('inst_type', 'delist_date', 'status', 'currency',
                            'buylot', 'selllot', 'pricetick',
                            'underlying', 'product', 'market', 'multiplier')
    ALL_FIELD_TRADING_DAY = ('isweekday', 'isweekend', 'isholiday')

    # 定义完整的数据的起止时间
    TRADE_BEGIN_TIME = ('090100', '091600', '093100', '210100')
    TRADE_END_TIME = ('150000', '151500')

    def __init__(self):
        super(JaqsDataCollector, self).__init__()
        self.vendor = VENDOR_JAQS

        self._sdk = None
        self.dbAdaptor = None
        self.rqCollector = RQDataCollector()

        self.tradeCalArray = None
        self.instTypeNameMap = None
        self.futureExchangeMap = None
        self.futureCurrentMainContractMap = None
        self.futureHistoryMainContractMap = None
        self.basicDataMap = dict()
        self.symbolMap = dict()

        self.connectApi()

    def _queryBasicData(self, category, outputField, inputParameter):
        """
        封装jaqs.DataApi.query()方法，原始参数和返回数据查询jaqs文档。
        ----------------------------------------------------------------------------------------------------------------
        :param category: string
                数据类别
        :param outputField: iterable container. list[string] or tuple(string)
                输出列名
        :param inputParameter: dict
                输入参数
        :return: pandas.DataFrame
        """
        if outputField is None:
            outputField = ''
        outputField = ','.join(outputField)
        generator = ('{}={}'.format(key, value) for key, value in inputParameter.items())
        inputParameter = '&'.join(generator)
        self.debug(category, outputField, inputParameter)
        df, msg = self._sdk.query(view=category, fields=outputField, filter=inputParameter, data_format='pandas')
        return df

    def _queryInstrumentInfo(self, outputFiled=None, **kwargs):
        """
        调用self.queryBasicData, 获取证券市场基础信息。
        ----------------------------------------------------------------------------------------------------------------
        :param outputFiled: list[string] or tuple(string)
                输出列名
        :param kwargs:
                jaqs's Api支持的输入参数
        :return: pandas.DataFrame
        """
        self.debug(outputFiled, kwargs)
        return self._queryBasicData(self.VIEW_INSTRUMENT_INFO, outputFiled, kwargs)

    def _queryInstrumentInfoByType(self, inst_types, outputFiled=None, refresh=False, outputPath=None, **kwargs):
        """
        查询特定类别的证券市场基础信息，比如股票或期货，并保存成文件到数据目录。
        ----------------------------------------------------------------------------------------------------------------
        :param inst_types: tuple(int)
                定义市场类别的整数tuple，直接传入设定好的类属性
        :param outputFiled: iterable container. list[string] or tuple(string)
                输出列名
        :param refresh: bool
                是否覆盖已有的文件
        :param outputPath: string
                文件路径
        :param kwargs:
        :return: pandas.DataFrame
        """
        df_list = [self._queryInstrumentInfo(outputFiled=outputFiled, inst_type=i, **kwargs) for i in inst_types]
        df = pd.concat(df_list)
        if outputPath is None:
            outputPath = self.getBasicDataFilePath(inst_types)
        if not os.path.exists(outputPath) or refresh:
            df.to_csv(outputPath, encoding='utf-8-sig', index=False)
        return df

    def _querySecTradeCal(self, outputFiled=None, **kwargs):
        """
        查询交易日历。
        ----------------------------------------------------------------------------------------------------------------
        :param outputFiled: iterable container. list[string] or tuple(string)
                输出列名
        :param kwargs:
        :return: pandas.DataFrame
        """
        return self._queryBasicData(self.VIEW_TRADING_DAY_INFO, outputFiled, kwargs)

    def _queryIndexInfo(self, outputFiled=None, **kwargs):
        """
        查询基金基础信息.
        ----------------------------------------------------------------------------------------------------------------
        :param outputFiled: iterable container. list[string] or tuple(string)
                输出列名
        :param kwargs:
        :return: pandas.DataFrame
        """
        return self._queryBasicData(self.VIEW_INDEX_INFO, outputFiled, kwargs)

    @staticmethod
    def getBarFilename(symbol, fromDate, fromTime, toDate, toTime, fileExt):
        """
        生成jaqs的1分钟线数据的文件名，按天保存。
        ----------------------------------------------------------------------------------------------------------------
        :param symbol: string
                股票或期货合约代码
        :param fromDate: int
                开始日期
        :param fromTime: int
                开始时间
        :param toDate: int
                结束日期
        :param toTime: int
                结束时间
        :param fileExt: string
                文件扩展名，通常是'csv'
        :return: string
                文件名，如'rb1805_20180101-2101_to_20180301-1500.csv'
        """
        return '{}_{:0>8d}-{:0>6d}_to_{:0>8d}-{:0>6d}.{}'.format(symbol, fromDate, fromTime, toDate, toTime, fileExt)

    @staticmethod
    def parseBarFilename(filename):
        """
        从1分钟数据文件名解析对应的信息。
        ----------------------------------------------------------------------------------------------------------------
        :param filename: string
                文件名
        :return: tuple(string...)
                文件信息的tuple
        """
        pattern = r'[_.-]'
        symbol, beginDate, beginTime, _to, endDate, endTime, _ext = re.split(pattern, filename)
        return symbol, beginDate, beginTime, endDate, endTime

    def connectApi(self, user=None, token=None, address=None):
        """
        连接Jaqs的Api，默认从config.json读取账号设置。
        ----------------------------------------------------------------------------------------------------------------
        :param address: string
                地址，详查jaqs文档
        :param user: string
                用户名
        :param token: string
                令牌
        """
        if self._sdk is None:
            settingFile = os.path.join(dirname(abspath(__file__)), FILE_SETTING)
            setting = loadSetting(settingFile)
            if user is None:
                user = setting['jaqs_user']
            if token is None:
                token = setting['jaqs_token']
            if address is None:
                address = setting['jaqs_address']
            try:
                self._sdk = DataApi(address)
                self._sdk.login(user, token)
                self._isConnected = True
                self.info(u'{}:{}'.format(API_SUCCEED, self.vendor))
            except Exception as e:
                msg = u"{}:{}".format(ERROR_UNKNOWN, e)
                self.error(msg)

    def setDbAdaptor(self):
        """
        设置vnpy数据库转换器
        ----------------------------------------------------------------------------------------------------------------
        :return:
        """
        convertClsName = self.vendor + 'Converter'
        self.dbAdaptor = VnpyAdaptor()
        self.dbAdaptor.setActiveConverter(convertClsName)

    def getDbAdaptor(self):
        """
        获取vnpy数据库转换器
        ----------------------------------------------------------------------------------------------------------------
        :return: vnpyAdaptor
        """
        if self.dbAdaptor is None:
            self.setDbAdaptor()
        return self.dbAdaptor

    def getInstTypeToNameMap(self):
        """
        获取inst_type与市场名称的映射字典。
        ----------------------------------------------------------------------------------------------------------------
        :return: dict{tuple(int): string}
                key: inst_type的tuple，已经定义在类属性中
                value: 市场名称，如'stock'、'future'等
        """
        if self.instTypeNameMap is None:
            typeMap = {key: value for key, value in JaqsDataCollector.__dict__.items() if 'INST_TYPE' in key}
            self.instTypeNameMap = {value: key.replace('INST_TYPE_', '').lower() for key, value in typeMap.items()}
        return self.instTypeNameMap

    def getBasicDataFilePath(self, inst_types):
        """
        获取保存基础数据的文件路径。
        ----------------------------------------------------------------------------------------------------------------
        :param inst_types: tuple(int)
                inst_types已经在类属性中预先定义，可以直接传入类属性的值
        :return: string
                文件路径
        """
        instTypeNameMap = self.getInstTypeToNameMap()
        filename = '{}_basic.csv'.format(instTypeNameMap.get(inst_types))
        return os.path.join(self.getBasicDataPath(), filename)

    def getFutureCurrentMainContract(self):
        """
        借助rqCollector对象获取期货品种当前主力合约的合约代码，因为jaqs自身不提供查询这个数据的api。
        ----------------------------------------------------------------------------------------------------------------
        :return: OrderedDict {string: string}
                key: 品种代码，如'rb'
                value: 主力合约代码，如'rb1810'
        """
        if self.futureCurrentMainContractMap is None:
            self.futureCurrentMainContractMap = self.rqCollector.getCurrentMainContract()
        return self.futureCurrentMainContractMap

    def getPopularFuture(self, unpopular=None):
        """
        获取当前热门的期货合约品种。
        ----------------------------------------------------------------------------------------------------------------
        :param unpopular: iterable
                冷门的期货品种，用于被排除
        :return: list
                热门期货品种列表
        """
        if unpopular is None:
            unpopular = self._unpopularFuture
        varieties = self.getFutureCurrentMainContract().keys()
        return [variety for variety in varieties if variety not in unpopular]

    def setFutureCurrentMainContract(self, newMainDict, updateFile=True):
        """
        修改当前期货品种主力合约代码.
        ----------------------------------------------------------------------------------------------------------------
        :param newMainDict: dict{string: string}
                新的合约字典：{key: 期货品种代码
                            value: 主力合约代码}
        :param updateFile: bool
                是否更新rq的依赖文件
        """
        for key, newValue in newMainDict.items():
            if key in self.futureCurrentMainContractMap:
                self.futureCurrentMainContractMap[key] = newValue.decode()
        if updateFile:
            filePath = self.rqCollector.getRequiredFilePath('current_main_contract.json')
            saveSetting(self.futureCurrentMainContractMap, filePath)

    def getFutureExchangeMap(self):
        """
        从自己的基础数据中获取期货品种代码与交易所代码（Jaqs）的映射。
        ----------------------------------------------------------------------------------------------------------------
        :return: dict{string: string}
                    key: 交易品种代码. eg.'rb'
                    value: 交易所代码. eg. 'CZE'
        """
        if self.futureExchangeMap is None:
            df = self.getBasicData(self.INST_TYPE_FUTURE)
            self.futureExchangeMap = self.getExchange(df.symbol)
        return self.futureExchangeMap

    def addFutureExchangeToSymbol(self, symbol):
        """
        添加市场代码到期货合约品种，并做缓存加速。
        ----------------------------------------------------------------------------------------------------------------
        :param symbol: string
                期货合约代码，如'rb1801'
        :return: string
                jaqs定义的合约代码，如'rb1801.SHF'
        """
        if symbol not in self.symbolMap:
            exchangeMap = self.getFutureExchangeMap()
            varietySymbol = self.getUnderlyingSymbol(symbol)
            exchangeSymbol = exchangeMap[varietySymbol]
            if varietySymbol in exchangeMap:
                self.symbolMap[symbol] = self.addExchange(symbol, exchangeSymbol)
        return self.symbolMap.get(symbol)

    def isCZCE(self, symbol):
        """
        判断期货合约是否属于郑商所。
        因为郑商所的合约代码数字部分是三位数的，例如AP901，但是有的数据商保存成四位数的，如AP1901，所以可能需要进行转换。
        ----------------------------------------------------------------------------------------------------------------
        :param symbol: string
                期货合约代码
        :return: bool
        """
        underlying = self.getUnderlyingSymbol(symbol)
        return self.getFutureExchangeMap().get(underlying) == self.EXCHANGE_MAP[EXCHANGE_CZCE]

    def adjustCZCESymbol(self, symbol):
        """
        转换郑商所期货合约代码。将四位数字的期货代码（部分数据商保存的代码，如米筐）转换为三位数的可交易代码（实际代码）。
        :param symbol: string
                四位数字的郑商所合约代码，eg. 'SR1909'
        :return: string
                三位数字的郑商所合约代码，eg. 'SR909'
        """
        return '{}{}'.format(self.getUnderlyingSymbol(symbol), symbol[-3:])

    def getFutureHistoryMainContractMap(self):
        """
        获取历史期货主力合约的起止日期数据。
        ----------------------------------------------------------------------------------------------------------------
        :return: dict{string: [(string, string, string), ...]}
                key: 期货品种代码，如'rb'
                value: 列表，元素是3项tuple(期货合约代码, 开始主力日期, 结束主力日期).
        """
        if self.futureHistoryMainContractMap is None:
            self.futureHistoryMainContractMap = self.rqCollector.getFutureMainContractDate()
        return self.futureHistoryMainContractMap

    def getFutureContractLifespan(self, symbol):
        """
        获取期货合约品种的生存期限。
        ----------------------------------------------------------------------------------------------------------------
        :param symbol: string
                期货品种代码或合约代码，如'rb1801' or 'rb'
        :return:tuple(string, string)
                由上市日期和退市日期构成的tuple，格式%Y%m%d
        """
        # 如果传入一个期货品种代码
        map_ = self.getFutureHistoryMainContractMap()
        if symbol in map_:
            mainContracts = map_.get(symbol)
            begin = mainContracts[0][1]
            end = mainContracts[-1][2] if mainContracts[-1][2] != '' else dateToStr(datetime.today())
            jaqsStart = datetime(2012, 1, 1)
            if strToDate(end) <= jaqsStart:
                self.info(DATA_IS_NONE)
                return None
            else:
                begin = begin if strToDate(begin) > jaqsStart else dateToStr(jaqsStart)
                return rmDateDash(begin).encode(), rmDateDash(end)  # 统一格式
        # 如果传入期货合约代码
        df = self.getBasicData(self.INST_TYPE_FUTURE)
        queryRes = df.loc[df.symbol == self.addFutureExchangeToSymbol(symbol)]
        if queryRes.empty:
            self.info(DATA_IS_NONE)
            return None
        else:
            return str(queryRes.iloc[0].list_date), str(queryRes.iloc[0].delist_date)

    def getMainContractListByPeriod(self, underlyingSymbol, start=None, end=None):
        """
        获取期货品种在某个日期区间的主力合约构成列表。
        ----------------------------------------------------------------------------------------------------------------
        :param underlyingSymbol: string
                期货品种代码 eg.'rb'
        :param start: string
                区间开始日期，日期格式：'%Y-%m-%d'或'%Y%m%d'
        :param end: string
                区间结束日期，日期格式：'%Y-%m-%d'或'%Y%m%d'

        :return: list[tuple(string, string, string)]
                返回数据范例：
                [(u'zn0710', u'2007-08-01', u'2007-08-29'),
                (u'zn0711', u'2007-08-30', u'2007-09-26'),
                ...]
        """

        mainContracts = self.getFutureHistoryMainContractMap().get(underlyingSymbol)
        allBeginDt = strToDate(mainContracts[0][1])
        if mainContracts[-1][2] != '':
            allEndDt = strToDate(mainContracts[-1][2])
        else:
            allEndDt = datetime.today()

        # 如果没有输入日期，则默认采用列表的开始和结束日期
        start = allBeginDt if start is None else strToDate(self.getNextTradeDay(start))
        end = allEndDt if end is None else strToDate(self.getPreTradeDay(end))

        # 如果输入的时间窗口与列表的时间范围没有交集或者刚好相切。
        if start == allEndDt:
            return mainContracts[-1]
        if end == allBeginDt:
            return mainContracts[0]
        if start > allEndDt or end < allBeginDt:
            return []

        # 如果输入的时间窗口在列表的时间范围之内。
        if start > end:
            self.error(JAQS_END_LT_START)
            return []
        startIdx = None
        endIdx = None
        for index, element in enumerate(mainContracts):
            symbol, beginDate, endDate = element
            beginDate = strToDate(beginDate)
            endDate = datetime.today() if endDate == '' else strToDate(endDate)
            if beginDate <= start and start <= endDate:
                startIdx = index
            if beginDate <= end and end <= endDate:
                endIdx = index + 1

        startIdx = 0 if startIdx is None else startIdx
        endIdx = len(mainContracts) if endIdx is None else endIdx
        return mainContracts[startIdx: endIdx]

    def getMainContractSymbolByDate(self, underlyingSymbol, date):
        """
        获取期货品种特定日期的期货主力合约代码
        ----------------------------------------------------------------------------------------------------------------
        :param underlyingSymbol: string
                期货品种代码，如'rb'
        :param date: string
                日期格式：'%Y-%m-%d' or '%Y%m%d'
        :return: symbol: string
                期货合约代码，如'rb1801'
        """
        date = strToDate(date)
        mainContracts = self.getFutureHistoryMainContractMap().get(underlyingSymbol)
        if mainContracts:
            for symbol, start, end in mainContracts:
                start = strToDate(start)
                if end == '':
                    end = datetime.today()
                else:
                    end = strToDate(end)
                if start <= date and date <= end:
                    return symbol

    def getBar(self, symbol, **kwargs):
        """
        封装jaqs.DataApi.bar()方法。
        ----------------------------------------------------------------------------------------------------------------
        :param symbol: string
                期货合约代码
        :param kwargs:
        :return: pandas.DataFrame
        """
        df, msg = self._sdk.bar(symbol=self.addFutureExchangeToSymbol(symbol), **kwargs)
        tradeDay = kwargs.get('trade_date')
        if df is None or df.empty:
            self.warn(u'{}:{}@{}'.format(DATA_IS_NONE, symbol, tradeDay))
        else:
            beginDate = df.iloc[0].date
            beginTime = df.iloc[0].time
            endDate = df.iloc[-1].date
            endTime = df.iloc[-1].time
            self.info(u'{}完成。区间: {} {} - {} {}'.format(symbol, beginDate, beginTime, endDate, endTime))
            return df, (beginDate, beginTime, endDate, endTime)

    def getDaily(self, symbol, startDate, endDate, **kwargs):
        """
        封装jaqs.DataApi.daily()方法，调用日线数据。
        ----------------------------------------------------------------------------------------------------------------
        :param symbol: string
                支持期货品种代码（eg.'rb', 'AP')或合约代码（eg.'rb1901', 'AP901'），交易合约代码必须是正确的。
        :param startDate:
                区间开始日期，日期格式：'%Y-%m-%d'或'%Y%m%d'
        :param endDate:
                区间开始日期，日期格式：'%Y-%m-%d'或'%Y%m%d'
        :param kwargs:
                daily()支持的其他参数，请参考jaqs文档。
        :return: pandas.DataFrame
        """

        # 如果代码是品种代码（如rb），需要手动拼接主力合约，因为jaqs的api不支持直接查询主力合约的日线数据。
        if symbol in self.getFutureCurrentMainContract().keys():
            contractList = self.getMainContractListByPeriod(symbol, startDate, endDate)
            lastIdx = len(contractList) - 1
            dfList = []
            for index, (tradeSymbol, start, end) in enumerate(contractList):
                # 历史主力合约列表的开始日期和结束日期可能和查询的开始日期和结束日期不一致，需要做调整。
                if index == 0:
                    start = startDate
                if end == '' or index == lastIdx:
                    end = endDate
                # 如果是郑商所的合约，需要从4位编码转为3位编码
                if self.isCZCE(tradeSymbol):
                    tradeSymbol = self.adjustCZCESymbol(tradeSymbol)
                df, msg = self._sdk.daily(self.addFutureExchangeToSymbol(tradeSymbol), start, end)
                if df is not None and not df.empty:
                    dfList.append(df)
                else:
                    self.info(u'{}:{}'.format(DATA_IS_NONE, symbol))
            if not dfList:
                return
            else:
                finalDf = pd.concat(dfList, ignore_index=True)
        else:
            finalDf, msg = self._sdk.daily(symbol=symbol, start_date=startDate, end_date=endDate, **kwargs)
        return finalDf

    def getBasicData(self, inst_types):
        """
        从数据文件获取特定市场的证券基础数据，并做缓存。如果数据不存在则自动下载。
        ----------------------------------------------------------------------------------------------------------------
        :param inst_types: tuple(int)
                定义市场类别的整数tuple，直接传入设定好的类属性
        :return: dict{string: DataFrame}
                保存各市场数据的字典
                key: 市场名称，如'future'
                value: pandas.DataFrame
        """
        typeName = self.getInstTypeToNameMap()[inst_types]
        if typeName not in self.basicDataMap:
            path = self.getBasicDataFilePath(inst_types)
            if os.path.exists(path):
                df = pd.read_csv(path)
            else:
                df = self._queryInstrumentInfoByType(inst_types, outputFiled=self.ALL_FIELD_INSTRUMENT)
            self.basicDataMap[typeName] = df
        return self.basicDataMap[typeName]

    def getTradeCal(self):
        """
        获取交易日列表并缓存。
        ----------------------------------------------------------------------------------------------------------------
        :return: np.array(np.int64)
                交易日列表，数据范例：[19900101, ..., 20180801]
        """
        if self.tradeCalArray is None:
            path = os.path.join(self.getBasicDataPath(), FILE_TRADE_CAL)
            if os.path.exists(path):
                df = pd.read_csv(path)
            else:
                df = self._querySecTradeCal()
                df.to_csv(path, encoding='utf-8')
            self.tradeCalArray = df.trade_date.values.astype(np.int64)
        return self.tradeCalArray

    def getTradingDayArray(self, start, end=None):
        """
        获取某个日期区间的交易日列表。
        ----------------------------------------------------------------------------------------------------------------
        :param start: string
                开始日期，格式：format:'%-%m-%d' or '%Y%m%d'
        :param end: string
                结束日期
        :return: np.array(string)
                交易日列表，格式：'%Y%m%d'
        """
        tradeCal = self.getTradeCal()
        if end is None:
            end = datetime.today().strftime('%Y%m%d')
        if '-' in start:
            start = rmDateDash(start)
        if '-' in end:
            end = rmDateDash(end)
        tradeDayRange = tradeCal[(tradeCal >= int(start)) & (tradeCal <= int(end))]
        return tradeDayRange.astype(np.string_)

    def getNextTradeDay(self, date, nDays=1):
        """
        获取指定日期之后n天的交易日。
        ----------------------------------------------------------------------------------------------------------------
        :param date: string.
                指定日期，格式：%Y-%m-%d or %Y%m%d
        :param nDays: int
                几天
        :return: string
                交易日，格式: %Y%m%d
        """
        if '-' in date:
            date = rmDateDash(date)
        tradeCal = self.getTradeCal()
        nextDays = tradeCal[tradeCal > int(date)]
        if len(nextDays) == 0:
            return date
        try:
            next_ = str(nextDays[nDays - 1])
        except IndexError:
            next_ = str(nextDays[-1])
        return next_

    def getPreTradeDay(self, date, nDays=1):
        """
        获取指定日期之前n天的交易日。
        ----------------------------------------------------------------------------------------------------------------
        :param date: string.
                指定日期，格式：%Y-%m-%d or %Y%m%d
        :param nDays: int
                几天
        :return: string
                交易日，格式: %Y%m%d
        """
        if '-' in date:
            date = rmDateDash(date)
        tradeCal = self.getTradeCal()
        preDays = tradeCal[tradeCal < int(date)]
        if len(preDays) == 0:
            return date
        try:
            pre = str(preDays[-nDays])
        except IndexError:
            pre = str(preDays[0])
        return pre

    def downloadBarByContract(self, symbol, start=None, end=None, refresh=False, saveToDb=False, needFull=True,
                              **kwargs):
        """
        通过期货品种代码或期货合约代码获取期货1分钟线数据。如果是品种代码则下载主力连续数据，如果是合约代码则下载对应数据。
        ----------------------------------------------------------------------------------------------------------------
        :param symbol: string
                期货品种代码或合约代码，如'rb1805' or 'rb'
        :param start: string
                开始日期，格式：%Y-%m-%d or %Y%m%d
        :param end: string
                结束日期
        :param refresh: bool
                是否覆盖已有文件
        :param saveToDb: bool
                是否保存到数据库
        :param needFull: bool
                是否需要完整数据
        """
        folder = self.getPricePath(FUTURE, BAR, symbol)

        # 获取已经存在的数据文件并移除不是完整数据的文件
        existedDay = []
        files = os.listdir(folder)
        if files:
            for filename in os.listdir(folder):
                _symbol, beginDate, beginTime, endDate, endTime = self.parseBarFilename(filename)
                if not self.isCompleteFile(filename):
                    os.remove(os.path.join(folder, filename))
                else:
                    existedDay.append(endDate)

        # 如果没有指定开始或结束日期，则下载整个合约的生存周期
        if start is None or end is None:
            lifespan = self.getFutureContractLifespan(symbol)
            if lifespan is None:  # 没有生存周期，不是合法的代码
                self.error(u'{}:{}'.format(ERROR_INVALID_SYMBOL, symbol))
                return
            else:  # 合法的交易合约代码
                if start is None:
                    start = lifespan[0]
                if end is None:
                    end = lifespan[1]
                    if strToDate(end) > datetime.today():  # 如果退市日期大于今日，则结束日期设为今日
                        end = dateToStr(datetime.today())
        tradingDay = self.getTradingDayArray(start, end)  # 获取要获取数据的交易日列表

        if refresh:  # 如果需要覆盖已存在的文件则直接使用交易日列表
            missionDay = tradingDay
        else:  # 如果不覆盖已存在数据，则要去掉本地已有数据的交易日
            missionDay = [day for day in tradingDay if day not in existedDay]

        for trade_date in missionDay:
            filename = df = None
            result = self.getBar(symbol, trade_date=int(trade_date), **kwargs)
            if result is not None:
                df, (beginDate, beginTime, endDate, endTime) = result
                filename = self.getBarFilename(symbol, beginDate, beginTime, endDate, endTime, 'csv')
            else:
                # 如果查询的代码是代表主力连续的期货品种代码，因为jaqs可能有缺失数据，尝试用缺失数据当日对应的主力合约代码去查询
                if symbol in self.getFutureExchangeMap().keys():
                    replaceSymbol = self.getMainContractSymbolByDate(symbol, trade_date)
                    if replaceSymbol is None:
                        self.info("Get Alternative data failed.")
                        continue
                    self.info("Trying to get from {}".format(replaceSymbol))
                    result = self.getBar(replaceSymbol, trade_date=int(trade_date), **kwargs)
                    if result is not None:
                        df, (beginDate, beginTime, endDate, endTime) = result
                        filename = self.getBarFilename(symbol, beginDate, beginTime, endDate, endTime, 'csv')
                        df.code = symbol
                        df.symbol = self.addFutureExchangeToSymbol(symbol)
                        df.oi = np.nan
                        df.settle = np.nan
                    else:
                        self.info("Alternative method failed.")

            # 保存到文件和数据库
            if filename and df is not None:
                df = df[df.volume != 0]
                # 设置不需要完整数据或数据本身就是是完整的才保存文件。
                if not needFull or self.isCompleteFile(filename):
                    path = os.path.join(folder, filename)
                    df.to_csv(path, encoding='utf-8-sig')
                # 保存到数据库
                if saveToDb:
                    self.saveBarToDb(df)

    def downloadMainContractBar(self, underlyingSymbol, start=None, end=None, **kwargs):
        """
        下载某个期货品种的历史主力合约的完整1分钟线数据。
        ----------------------------------------------------------------------------------------------------------------
        :param underlyingSymbol: string.
                期货品种代码，如'rb'
        :param start: string
                开始日期，格式：%Y-%m-%d or %Y%m%d
        :param end: string
                结束日期
        :param kwargs:
        :return:
        """
        contracts = [item[0] for item in self.getMainContractListByPeriod(underlyingSymbol, start, end)]
        self.debug(contracts)
        for contract in contracts:
            if self.isCZCE(contract):
                contract = self.adjustCZCESymbol(contract)
            self.downloadBarByContract(contract, **kwargs)

    def downloadAllMainContractBar(self, start=None, end=None, skipSymbol=None, **kwargs):
        """
        下载所有期货品种的历史主力合约的完整1分钟线数据。
        ----------------------------------------------------------------------------------------------------------------
        :param start: string
                开始日期，格式：%Y-%m-%d or %Y%m%d
        :param end: string
                结束日期
        :param skipSymbol: iterable container<string> list or tuple.
                跳过的期货品种。
        :param kwargs:
        :return:
        """
        if skipSymbol is None:
            skipSymbol = []
        symbols = self.getFutureExchangeMap().keys()
        for symbol in symbols:
            print(symbol)
            if symbol in self.getFutureCurrentMainContract().keys() and symbol not in skipSymbol:
                self.downloadMainContractBar(symbol, start, end, **kwargs)

    def downloadAllContinuousMainBar(self, start=None, end=None, skipSymbol=None, **kwargs):
        """
        下载所有期货主力连续合约的1分钟线数据。
        ----------------------------------------------------------------------------------------------------------------
        :param start: string
                开始日期，格式：%Y-%m-%d or %Y%m%d
        :param end: string
                结束日期
        :param skipSymbol: iterable container<string> list or tuple.
                跳过的期货品种。
        :param kwargs:
        :return:
        """
        if skipSymbol is None:
            skipSymbol = []
        symbols = self.getFutureExchangeMap().keys()
        for symbol in symbols:
            if symbol in self.getFutureCurrentMainContract().keys() and symbol not in skipSymbol:
                self.downloadBarByContract(symbol, start, end, **kwargs)

    def downloadAllContinuousMainDaily(self, start, end, **kwargs):
        """
        下载所有期货主力连续合约的日线数据。
        ----------------------------------------------------------------------------------------------------------------
        :param start: string
                开始日期，格式：%Y-%m-%d or %Y%m%d
        :param end: string
                结束日期
        :param kwargs:
        :return:
        """
        folder = self.getPricePath(FUTURE, DAILY, 'continuous')
        for symbol in self.getPopularFuture():
            filename = '{}0000.csv'.format(symbol)
            self.info('{} Downloading..'.format(filename))
            df = self.getDaily(symbol, start, end, **kwargs)
            if df is not None:
                path = os.path.join(folder, filename)
                df.to_csv(path, encoding='utf-8-sig')

    def downloadCurrentMainContractBar(self, start=None, skipSymbol=None, refresh=False):
        """
        下载所有期货品种当前主力合约的1分钟线数据。
        ----------------------------------------------------------------------------------------------------------------
        :param start: string
                开始日期，格式：%Y-%m-%d or %Y%m%d
        :param skipSymbol: iterable container<string> . list or tuple.
                跳过的期货品种
        :param refresh: bool
                是否覆盖本地文件和数据库
        :return:
        """
        currentMain = self.getFutureCurrentMainContract()
        symbols = [value for key, value in currentMain.items() if key not in skipSymbol]
        self.debug(symbols)
        if start is None:
            start = self.getPreTradeDay(dateToStr(datetime.today()), nDays=3)
        for symbol in symbols:
            self.downloadBarByContract(symbol, start=start, refresh=refresh, saveToDb=True)

    def saveBarToDb(self, df):
        """
        把1分钟bar写入数据库
        ----------------------------------------------------------------------------------------------------------------
        :param df: panas.DataFrame.
                包含1分钟数据的df
        :return:
        """
        adaptor = self.getDbAdaptor()
        adaptor.setFreq('bar')
        for row in df.iterrows():
            index, barSeries = row
            doc = adaptor.activeConverter.convertToVnpyBar(barSeries)
            adaptor.saveBarToDb(doc)

    def isCompleteFile(self, filename):
        """
        判断1分钟数据文件是否包含完整数据
        ----------------------------------------------------------------------------------------------------------------
        :param filename: string
                文件名
        :return: bool
        """
        _symbol, beginDate, beginTime, endDate, endTime = self.parseBarFilename(filename)
        complete = beginTime in self.TRADE_BEGIN_TIME and endTime in self.TRADE_END_TIME
        return complete

    def cleanIncompleteFile(self):
        """
        清理不是完整数据的1分钟线数据文件
        ----------------------------------------------------------------------------------------------------------------
        :return:
        """
        path = self.getPricePath(FUTURE, BAR)
        for root, dirs, files in os.walk(path):
            if files:
                files = (file_ for file_ in files if file_.endswith('.csv'))
                for f in files:
                    self.debug(f)
                    if not self.isCompleteFile(f):
                        fp = os.path.join(root, f)
                        self.info("Delete file: {}".format(fp))
                        os.remove(fp)
