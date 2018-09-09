# coding:utf-8

import csv
import os
import pymongo
import time
from datetime import datetime
from const import *
from functions import (initDataDirectory, parseFilename)


class DataBaseConnector(object):
    def __init__(self, host, port):
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


class Adaptor(object):
    pass


class VnpyAdaptor(object):
    """
    Load the csv data files and save to mongodb.
    --------------------------------------------------------------------------------------------------------------------
    The format of source csv file as following:
    {'close': '4324.0',
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
     '\xef\xbb\xbf': '192'}
    --------------------------------------------------------------------------------------------------------------------
    """

    def __init__(self, dbConnector, logger, dataPath):
        self.dbConnector = dbConnector
        self.name = 'vnpy'
        self.logger = logger
        self.db = None
        self.lastCollection = None
        self.dataPath = dataPath

    def setDbConnector(self, dbConnector):
        """
        :param dbConnector: instance
        :return:
        """
        self.dbConnector = dbConnector

    def setDataPath(self, dataPath):
        """
        :param dataPath: string
        :return:
        """
        self.dataPath = dataPath

    def setDb(self, dbName=None):
        """
        :param dbName: string
        :return:
        """
        if dbName is None:
            dbName = VNPY_MINUTE_DB_NAME
        self.db = self.dbConnector.connect()[dbName]

    def setDbCollection(self, colName):
        """
        Set a mongodb collection instance and create index if the collection is not exist.
        ----------------------------------------------------------------------------------------------------------------
        :param colName: string. It's same with symbol
        :return: None
        """
        if self.db is None:
            self.setDb()
        collection = self.db[colName]
        if not collection.index_information():
            collection.create_index([('datetime', pymongo.ASCENDING)], unique=True)
        self.lastCollection = collection

    def getDbExistedDataDay(self, colName):
        """
        Get days of existed data in mongodb.
        ----------------------------------------------------------------------------------------------------------------
        :param colName: string. contract symbol. eg.'rb1810'
        :return: list[string]. Format '%Y%m%d'
        """
        if self.db is None:
            self.setDb()
        collection = self.db[colName]
        cursor = collection.find(projection={'date': True, '_id': False})
        existedDay = []
        for doc in cursor:
            if doc['date'] not in existedDay:
                existedDay.append(doc['date'])
        return existedDay

    def convertBar(self, sourceBar):
        """
        Convert the dict format of source data to vnpy format.
        ----------------------------------------------------------------------------------------------------------------
        :param sourceBar: dict
        :return: dict
        """
        sameKey = ['open', 'high', 'low', 'close', 'volume']
        newDict = {key: float(value) for key, value in sourceBar.items() if key in sameKey}
        newDict['openInterest'] = float(sourceBar['oi'])
        newDict['symbol'] = sourceBar['code']
        newDict['vtSymbol'] = sourceBar['code']
        dtTuple = self.parseDatetime(sourceBar['date'], sourceBar['time'])
        newDict['datetime'], newDict['date'], newDict['time'] = dtTuple
        newDict['gatewayName'] = ''
        newDict['exchange'] = ''
        newDict['rawData'] = None
        return newDict

    @staticmethod
    def parseDatetime(date, time_):
        """
        Convert csv file's datetime format for vnpy bar. It is faster.
        ----------------------------------------------------------------------------------------------------------------
        :param date: string. format: %Y%m%d
        :param time_: string.
        :return:
        tuple(datetime, string, string). format of return time is %H:%M:%S
        """
        time_ = '{:0>6d}'.format(int(time_))
        date = str(date)
        year, month, day, = date[0:4], date[4:6], date[6:]
        hour, minute, second = time_[0:2], time_[2:4], time_[4:]
        datetime_ = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        time_ = '{}:{}:{}'.format(hour, minute, second)
        return datetime_, date, time_

    def genFilePaths(self, dataPath, skipDbExisted=True):
        """
        Get a generator of data file paths
        ----------------------------------------------------------------------------------------------------------------
        :param skipDbExisted:
        :param dataPath: string. Root path of price data
        :return: string. File path.
        """
        for path, dirList, fileList in os.walk(dataPath):
            if fileList:
                fileList = [fname for fname in fileList if fname.endswith('.csv')]
                existedDayMap = dict()
                for filename in fileList:
                    if skipDbExisted:
                        symbol, _sd, _st, endDate, _et = parseFilename(filename)
                        if existedDayMap.get(symbol) is None:
                            existedDayMap[symbol] = self.getDbExistedDataDay(symbol)
                        if endDate in existedDayMap[symbol]:
                            self.logger.info('{} is existed in db.'.format(endDate))
                            continue
                    self.logger.info('[Saving..] {}'.format(filename))
                    yield os.path.join(path, filename)

    @staticmethod
    def genAllDayBars(filePaths):
        """
        Get a generator of minute bars of every trade day. The element itself is csv.DictReader instance(iterator).
        ----------------------------------------------------------------------------------------------------------------
        :param filePaths: iterator.
        :return: iterator. Minute bars of every trade day.
        """
        for path in filePaths:
            f = open(path)
            csvDict = csv.DictReader(f)
            yield csvDict
            f.close()

    @staticmethod
    def genBar(allDayBars):
        """
        Get a generator of every minute bar.
        ----------------------------------------------------------------------------------------------------------------
        :param allDayBars: iterator.
        :return: dict. The source data of minute bar in csv file.
        """
        for singleDayBars in allDayBars:
            for sourceBar in singleDayBars:
                yield sourceBar

    def saveBarToDb(self, document):
        """
        Save a single bar to mongodb.
        ----------------------------------------------------------------------------------------------------------------
        :param document: dict
        :return: None
        """
        curColName = document['symbol']
        if self.lastCollection is None or self.lastCollection.name != curColName:
            self.setDbCollection(curColName)
        flt = {'datetime': document['datetime']}
        self.lastCollection.replace_one(flt, document, upsert=True)

    def allFutureFilesToDb(self, path=None, skipDbExisted=True):
        """
        Read all price data and save all bars to db.
        ----------------------------------------------------------------------------------------------------------------
        :return: None
        """
        start = time.time()
        if path is None:
            path = os.path.join(self.dataPath, DIR_JAQS_PRICE_DATA, FUTURE)
        filePaths = self.genFilePaths(path, skipDbExisted)
        allDayBars = self.genAllDayBars(filePaths)
        barGenerator = self.genBar(allDayBars)
        for bar in barGenerator:
            dbBar = self.convertBar(bar)
            self.saveBarToDb(dbBar)
        self.logger.info('Mission is over. time consumption is {}'.format(time.time() - start))
