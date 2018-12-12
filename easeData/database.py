# coding:utf-8

import re
import csv
import os
import pymongo
import time
import codecs
import importlib
import pandas as pd
from datetime import datetime

from const import *
from text import *
from base import LoggerWrapper, DataVendor
from functions import (rmDateDash, strToDate)


class DataBaseConnector(LoggerWrapper):
    def __init__(self, host, port):
        super(DataBaseConnector, self).__init__()
        self.dbHost = host
        self.dbPort = port
        self.dbName = ''

    def setHost(self, host):
        self.dbHost = host

    def setPort(self, port):
        self.dbPort = port

    def connect(self):
        pass


class MongoDbConnector(DataBaseConnector):
    def __init__(self, host='localhost', port=27017):
        super(MongoDbConnector, self).__init__(host, port)
        self.client = None

    def connect(self):
        self.client = pymongo.MongoClient(self.dbHost, self.dbPort)
        return self.client


class CSVFilesLoader(LoggerWrapper):
    """
    CSV文件读取器
    """

    def __init__(self, adaptor):
        super(CSVFilesLoader, self).__init__()
        self._adaptor = adaptor

    def getTargetPath(self):
        return self._adaptor.targetPath

    def loadFiles(self, path):
        """
        读取文件数据，返回数据的生成器。
        :return: tuple(string, csvDictReader)
                返回元祖，第一个数据为股票或期货合约代码， 第二个为csv的数据
        """
        pathIterator = self.getFilePaths(path)
        return self.getCsvData(pathIterator)

    def getFilePaths(self, path):
        """
        获取目标路径csv文件路径的生成器。
        :param path:
        :return:
        """
        for root, dirs, files in os.walk(path):
            if files:
                files = [f for f in files if f.endswith('csv')]
                for filename in files:
                    if self._adaptor.freq == BAR:
                        converter = self._adaptor.activeConverter
                        if converter.isCsvBarInDB(filename):
                            self.info(u'{}:{}'.format(FILE_DATA_EXISTED_IN_DB, filename))
                            continue
                    self.info(u'{}:{}'.format(FILE_LOADING, filename))
                    yield os.path.join(root, filename)

    def getCsvData(self, pathIterator):
        """
        获取csv文件内数据的的生成器。
        ----------------------------------------------------------------------------------------------------------------
        :param pathIterator: generator
        :return:
        """
        for fp in pathIterator:
            symbol = self.parseFilename(os.path.basename(fp))[0]
            f = codecs.open(fp, encoding='utf-8-sig')
            csvDict = csv.DictReader(f)
            yield symbol, csvDict  # 有的csv文件内没有合约代码symbol的数据，需要另外返回symbol
            f.close()


class VnpyAdaptor(LoggerWrapper):
    """
    Vnpy数据库适配接口。将不同的数据源转换为vnpy预设的数据库格式。
    """

    DB_NAME_MAP = {
        DAILY: VNPY_DAILY_DB_NAME,
        BAR: VNPY_MINUTE_DB_NAME,
        TICK: VNPY_TICK_DB_NAME
    }

    def __init__(self):
        super(VnpyAdaptor, self).__init__()

        self._dbConnector = MongoDbConnector()
        self._lastCollection = None
        self._lastDbName = None
        self._db = None
        self.freq = None
        self.targetPath = None
        self.converters = dict()
        self.activeConverter = None
        self.csvLoader = CSVFilesLoader(self)

    def _setDb(self):
        """
        设置数据库属性。
        ----------------------------------------------------------------------------------------------------------------
        :return:
        """
        dbName = self.DB_NAME_MAP[self.freq]
        self._db = self._dbConnector.connect()[dbName]

    def _setDbCollection(self, colName):
        """
        设置mongo的集合对象，并为不存在的集合创建索引。
        ----------------------------------------------------------------------------------------------------------------
        :param colName: string.
                数据库集合名，与股票或合约代码名相同
        :return:
        """
        collection = self._db[colName]
        if not collection.index_information():  # 判断集合是否存在
            collection.create_index([('datetime', pymongo.ASCENDING)], unique=True)
        # 保存工作状态
        self._lastDbName = self._db.name
        self._lastCollection = collection

    def getDb(self):
        """
        获取db对象
        :return:
        """
        return self._db

    def getCollection(self, colName):
        """
        获取Mongo数据库集合。
        ----------------------------------------------------------------------------------------------------------------
        :param colName: string
                Mongo数据库集合名
        :return: pymongo.collection
        """
        if self._db is not None:
            return self._db[colName]

    def getConverter(self, converterClsName):
        """
        通过转换器类名获取转换器实例。
        ----------------------------------------------------------------------------------------------------------------
        :param converterClsName: string
                转换器类名
        :return:
        """
        if self.converters.get(converterClsName) is None:
            # moduleName = os.path.basename(__file__).rstrip('.py')
            # module_ = __import__(moduleName)
            module_ = importlib.import_module('easeData.database')
            converterClass = getattr(module_, converterClsName)
            self.converters[converterClsName] = converterClass(self)
        return self.converters[converterClsName]

    def setFreq(self, freq):
        """
        设置数据周期。
        ----------------------------------------------------------------------------------------------------------------
        :param freq: string
                数据周期。'daily', 'bar'或'tick'
        :return:
        """
        self.freq = freq
        self._setDb()

    def setActiveConverter(self, converterClsName):
        """
        设置使用的转换器。
        ----------------------------------------------------------------------------------------------------------------
        :param converterClsName: string
                转换器类名，目前支持'JQDataConverter', 'JaqsDataConverter', 'RQDataConverter'
        :return:
        """
        self.activeConverter = self.getConverter(converterClsName)

    def setTargetPath(self, types, symbol=None):
        """
        设置需要存入数据库的目标文件路径。
        ----------------------------------------------------------------------------------------------------------------
        :param types: string
                市场。支持'stock', 'future'
        :param symbol: string
                股票代码或期货合约
        :return:
        """
        self.targetPath = self.activeConverter.getPricePath(types, self.freq, symbol)

    def saveBarToDb(self, document):
        """
        把单根1分钟数据存入数据库。
        ----------------------------------------------------------------------------------------------------------------
        :param document: dict
        :return: None
        """
        curColName = document['symbol']
        if self._lastCollection is None or self._lastCollection.name != curColName or self._lastDbName != self._db.name:
            self._setDbCollection(curColName)
        flt = {'datetime': document['datetime']}
        self._lastCollection.replace_one(flt, document, upsert=True)

    def filesToDb(self):
        """
        把目标路径的csv文件存入mongo数据库
        ----------------------------------------------------------------------------------------------------------------
        """
        start = time.time()
        print(self.targetPath)
        dataGenerator = self.csvLoader.loadFiles(self.targetPath)
        for symbol, data in dataGenerator:
            self.activeConverter.bar['symbol'] = symbol
            self.activeConverter.bar['vtSymbol'] = symbol
            for bar in data:
                bar = self.activeConverter.convertToVnpyBar(bar)
                self.saveBarToDb(bar)
        self.info(u'{}:{}'.format(TIME_CONSUMPTION, time.time() - start))

    def dfToDb(self, df, symbol):
        self.activeConverter.bar['symbol'] = symbol
        self.activeConverter.bar['vtSymbol'] = symbol
        for (idx, bar) in df.iterrows():
            bar = self.activeConverter.convertToVnpyBar(bar)
            self.saveBarToDb(bar)

    def saveAllBar(self):
        pass

    def saveAllDaily(self):
        pass

    def saveAllTick(self):
        pass


class FormatConverter(DataVendor):
    """
    Vnpy分钟线数据格式转换基类
    """
    GENERAL_KEY = ('open', 'high', 'low', 'close', 'volume')

    def __init__(self, adaptor):
        super(FormatConverter, self).__init__()
        self._adaptor = adaptor

        # 初始化bar的默认字段
        self.bar = dict()
        self.bar['gatewayName'] = ''
        self.bar['exchange'] = ''
        self.bar['rawData'] = None

    @staticmethod
    def parseDatetime(dtStr):
        """
        从字符串格式的datetime返回datetime、字符串date、字符串time
        ----------------------------------------------------------------------------------------------------------------
        :param dtStr: string.
                日期字符串。支持格式: %Y-%m-%d %H:%M:%S 或 %Y%m%d %H%M%S
        :return: tuple(datetime, string, string)
        """
        strList = dtStr.split()
        if len(strList) == 1:  # 如果只有日期
            datetime_ = strToDate(strList[0])
            date_ = rmDateDash(strList[0])
            time_ = '00:00:00'
        else:  # 如果有日期和时间
            date_, time_ = strList
            if '-' in date_:
                date_ = rmDateDash(date_)
            if ':' in time_:
                time_ = ''.join(time_.split(':'))
            year, month, day, = date_[0:4], date_[4:6], date_[6:]
            hour, minute, second = time_[0:2], time_[2:4], time_[4:]
            datetime_ = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
            time_ = '{}:{}:{}'.format(hour, minute, second)
        return datetime_, date_, time_

    def isCsvBarInDB(self, filename):
        """
        判断1分钟数据文件的数据是否已在数据库存在。
        :return:
        """
        raise NotImplementedError

    def convertToVnpyBar(self, sourceDict):
        """
        把原始1分钟数据转为vnpy格式的数据.
        ----------------------------------------------------------------------------------------------------------------
        :param sourceDict: dict
        :return: dict
        """
        raise NotImplementedError


class RQDataConverter(FormatConverter):
    """
    米筐数据转换器
    --------------------------------------------------------------------------------------------------------------------
    CSV原始数据:
        {
            'close': '48040.0',
             'datetime': '2018-09-17 21:01:00',
             'high': '48050.0',
             'limit_down': '45730.0',
             'limit_up': '50540.0',
             'low': '47980.0',
             'open': '48010.0',
             'open_interest': '193856.0',
             'total_turnover': '490176400.0',
             'trading_date': '2018-09-18',
             'volume': '2042.0'
         }
    米筐的数据可能有一些错误，例如有些地方的价格可能是'48040.00000005'，有多位小数，这里直接用round()处理。
    """

    def __init__(self, adaptor):
        super(RQDataConverter, self).__init__(adaptor)
        self.vendor = VENDOR_RQ
        self.lastTradeDate = None  # 只是用来输出日志

    def isCsvBarInDB(self, filename):
        pass

    def convertToVnpyBar(self, sourceDict):
        bar = self.bar
        for key in self.GENERAL_KEY:
            bar[key] = round(float(sourceDict[key]), 3)
        bar['openInterest'] = float(sourceDict['open_interest']) if sourceDict['open_interest'] != '' else 0.0
        bar['datetime'], bar['date'], bar['time'] = self.parseDatetime(sourceDict['datetime'])
        tradingDate = sourceDict['trading_date']
        if tradingDate != self.lastTradeDate:
            self._adaptor.info('Symbol:{} Trading date: {} is saving..'.format(bar['symbol'], tradingDate))
            self.lastTradeDate = tradingDate
        return bar


class JaqsDataConverter(FormatConverter):
    """
    Jaqs数据转换接口
    --------------------------------------------------------------------------------------------------------------------
    原始CSV数据:
        {
            'close': '4324.0',
             'code': 'rb1810',
             'date': '20180816',
             'freq': '1M',
             'high': '4324.0',
             'low': '4324.0',
             'oi': '1798748.0',
             'open': '4324.0',
             'settle': '',
             'symbol': 'rb1810.SHF',
             'time': '1300',
             'trade_date': '20180816',
             'turnover': '0.0',
             'volume': '0.0',
             'vwap': '4335.24345859',
             '\xef\xbb\xbf': '192'
         }
    """

    def __init__(self, adaptor):
        super(JaqsDataConverter, self).__init__(adaptor)
        self.vendor = VENDOR_JAQS

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

    def isCsvBarInDB(self, filename):
        """
        通过csv文件名，判断该月的数据文件是否在数据库已存在，如果是当月的文件则强制重新写入。
        ----------------------------------------------------------------------------------------------------------------
        :param filename: string
                csv文件名
        :return: bool
        """
        try:
            symbol, _sd, _st, endDate, _et = self.parseBarFilename(filename)
            existedDays = self.getDbExistedDataDay(symbol)
            if endDate in existedDays:
                return True
        except ValueError:
            self.error(u'{}:{}'.format(FILE_INVALID_NAME, filename))

    def getDbExistedDataDay(self, colName):
        """
        获取某个合约分钟线数据已保存在数据库的交易日列表。适用于Jaqs的collector
        ----------------------------------------------------------------------------------------------------------------
        :param colName: string
                合约代码，如'rb1810'
        :return: list[string].
                日期Format '%Y%m%d'
        """
        collection = self._adaptor.getCollection(colName)
        cursor = collection.find(projection={'date': True, '_id': False})
        dateList = [doc['date'] for doc in cursor]
        return set(dateList)

    def convertToVnpyBar(self, sourceDict):
        """
        把原始1分钟数据转为vnpy格式的数据
        ----------------------------------------------------------------------------------------------------------------
        :param sourceDict: dict
        :return: dict
        """
        bar = self.bar
        self.bar['symbol'] = sourceDict['code']
        self.bar['vtSymbol'] = sourceDict['code']
        for key in self.GENERAL_KEY:
            bar[key] = float(sourceDict[key])
        bar['openInterest'] = float(sourceDict['oi']) if sourceDict['oi'] != '' else 0.0
        time_ = '{:0>6d}'.format(int(sourceDict['time']))
        dtStr = '{} {}'.format(sourceDict['date'], time_)
        dtTuple = self.parseDatetime(dtStr)
        bar['datetime'], bar['date'], bar['time'] = dtTuple
        return bar


class JQDataConverter(FormatConverter):
    """
    聚宽数据转换接口
    --------------------------------------------------------------------------------------------------------------------
    CSV原始数据：
    {
        '': '2018-11-12 09:01:00'
        'open': 3829.0,
        'close': 3857.0,
        'high': 3857.0,
        'low': 3825.0,
        'volume': 183768.0
        'money': 7062675620.0,
     }

    """

    def __init__(self, adaptor):
        super(JQDataConverter, self).__init__(adaptor)
        self.vendor = VENDOR_JQ

    def isMonthExisted(self, colName, year, month):
        """
        判断数据库是否保存合约在某个具体月份的分钟线数据，从而决定是否写入重复数据。适用于聚宽的collector。
        通过从每月1日开始，发起1条9点10分的查询，直到月尾，如果某日记录存在，则立即返回true，到月尾都没有记录则返回false。
        这种判断方式不需要额外的交易日历。
        ----------------------------------------------------------------------------------------------------------------
        :param colName: string
                期货合约代码
        :param year: int
                年份
        :param month: int
                月份
        :return:
        """
        dateRange = self.getMonthBusinessDay(year, month)
        collection = self._adaptor.getCollection(colName)
        for date in dateRange:
            date = date.replace(hour=9, minute=10)
            if collection.find_one({'datetime': date}):
                return True
        return False  # 遍历结束后没返回结果，则返回false

    def isCsvBarInDB(self, filename):
        """
        通过csv文件名，判断该月的数据文件是否在数据库已存在，如果是当月的文件则强制重新写入。
        ----------------------------------------------------------------------------------------------------------------
        :param filename: string
                csv文件名
        :return: bool
        """
        if self.isCurrentMonthCsvBar(filename):  # 当月文件强制返回false
            return False
        try:
            symbol, year, month, _ = self.parseFilename(filename)
            return self.isMonthExisted(symbol, int(year), int(month))
        except ValueError:
            self.error(u'{}:{}'.format(FILE_INVALID_NAME, filename))

    def isCurrentMonthCsvBar(self, filename):
        """
        通过csv文件名判断是否当月的1分钟数据。
        :param filename:
        :return:
        """
        today = datetime.today()
        curYear, curMonth = today.year, today.month
        try:
            symbol, year, month, _ = self.parseFilename(filename)
            if int(year) == curYear and int(month) == curMonth:
                return True
        except ValueError:
            self.error(u'{}:{}'.format(FILE_INVALID_NAME, filename))

    def convertToVnpyBar(self, sourceDict):
        """
        把原始1分钟数据转为vnpy格式的数据
        ----------------------------------------------------------------------------------------------------------------
        :param sourceDict: dict
        :return: dict
        """
        bar = self.bar
        for key in self.GENERAL_KEY:
            bar[key] = float(sourceDict[key])
        bar['openInterest'] = 0.0
        dtTuple = self.parseDatetime(sourceDict['datetime'])
        bar['datetime'], bar['date'], bar['time'] = dtTuple
        return bar

    def rmCurrentMonthBarCsv(self):
        """
        删除当月的1分钟数据文件。
        :return:
        """
        path = self.getPricePath(FUTURE, BAR)
        for root, dirs, files in os.walk(path):
            if files:
                files = (file_ for file_ in files if file_.endswith('.csv'))
                for f in files:
                    if self.isCurrentMonthCsvBar(f):
                        fp = os.path.join(root, f)
                        self.info(u"Delete file: {}".format(fp))
                        os.remove(fp)

    def rmEmptyBarCsv(self):
        """
        删除空的1分钟数据文件
        :return:
        """
        path = self.getPricePath(FUTURE, BAR)
        for root, dirs, files in os.walk(path):
            if files:
                files = (file_ for file_ in files if file_.endswith('.csv'))
                for f in files:
                    fp = os.path.join(root, f)
                    df = pd.read_csv(fp)
                    if df.empty:
                        self.info(u"Delete file: {}".format(fp))
                        # os.remove(fp)
