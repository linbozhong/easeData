# coding:utf-8

import csv
import os
import pymongo
import time
from datetime import datetime
from const import *
from functions import (parseFilename, getUnderlyingSymbol, rmDateDash)


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
    """

    def __init__(self, dbConnector, logger, dataPath):
        self.name = 'vnpy'
        self.dbConnector = dbConnector
        self.logger = logger
        self.dataPath = dataPath
        self.db = None
        self.lastCollection = None
        self.converters = dict()

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

    def addConverter(self, converter):
        if self.converters.get(converter.name) is None:
            self.converters[converter.name] = converter

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
        self.logger.info('Checking existed record for {}'.format(colName))
        cursor = collection.find(projection={'date': True, '_id': False})
        dateList = [doc['date'] for doc in cursor]
        return set(dateList)

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

    def getContractPath(self, converterName, symbol):
        """
        Get csv file path of price data.
        ----------------------------------------------------------------------------------------------------------------
        :param converterName:
        :param symbol: string. e.g. 'rb1901' or 'rb'
        :return: string. path.
        """
        converter = self.converters.get(converterName)
        if converter:
            underlying = getUnderlyingSymbol(symbol)
            return os.path.join(self.dataPath, converter.dataDirName, FUTURE, underlying, symbol)

    def filesToDb(self, converterName, path=None, skipDbExisted=True):
        """
        Read all price data and save all bars to db.
        ----------------------------------------------------------------------------------------------------------------
        :return: None
        """
        start = time.time()
        converter = self.converters[converterName]
        if path is None:
            path = os.path.join(self.dataPath, converter.dataDirName, FUTURE)
        fileGen = converter.getFiles(path, skipDbExisted)
        csvGen = converter.getData(fileGen)
        for csvFileData in csvGen:
            for bar in csvFileData:
                dbBar = converter.convertToBar(bar)
                self.saveBarToDb(dbBar)
        self.logger.info('Mission is over. time consumption is {}'.format(time.time() - start))


class FormatConverter(object):
    """
    General format converter.From csv files to vnpy database record.
    """
    generalKey = ('open', 'high', 'low', 'close', 'volume')

    def __init__(self, adaptor):
        self.adaptor = adaptor
        self.name = 'generalConverter'
        self.bar = dict()
        self.bar['gatewayName'] = ''
        self.bar['exchange'] = ''
        self.bar['rawData'] = None

    @staticmethod
    def getSymbol(filename):
        """
        Get symbol from csv file name.
        ----------------------------------------------------------------------------------------------------------------
        :param filename: string
        :return: string
        """
        raise NotImplementedError

    @staticmethod
    def parseDatetime(dtStr):
        """
        Get tuple of datetime, date and time from datetime string.
        ----------------------------------------------------------------------------------------------------------------
        :param dtStr: string. format: %Y-%m-%d %H:%M:%S or %Y%m%d %H%M%S
        :return: tuple(datetime, string, string)
        """
        date_, time_ = dtStr.split()
        if '-' in date_:
            date_ = rmDateDash(date_)
        if ':' in time_:
            time_ = ''.join(time_.split(':'))
        year, month, day, = date_[0:4], date_[4:6], date_[6:]
        hour, minute, second = time_[0:2], time_[2:4], time_[4:]
        datetime_ = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        time_ = '{}:{}:{}'.format(hour, minute, second)
        return datetime_, date_, time_

    def getFiles(self, path, skipDbExisted=True):
        """
        Get generator of csv file path.
        ----------------------------------------------------------------------------------------------------------------
        :param path: string.
        :param skipDbExisted: bool. If true, existed record in db will be skipped. Not all converter support.
        :return:
        """
        for path, subDir, files in os.walk(path):
            if files:
                files = [f for f in files if f.endswith('csv')]
                for fileName in files:
                    self.adaptor.logger.info('[Saving..] {}'.format(fileName))
                    yield os.path.join(path, fileName)

    def getData(self, fileIterator):
        """
        Get generator of data in a csv file.
        ----------------------------------------------------------------------------------------------------------------
        :param fileIterator: generator
        :return:
        """
        for file_ in fileIterator:
            symbol = self.getSymbol(os.path.basename(file_))
            self.bar['symbol'] = symbol
            self.bar['vtSymbol'] = symbol
            f = open(file_)
            csvDict = csv.DictReader(f)
            yield csvDict
            f.close()

    @staticmethod
    def getSourceBarDict(dataIterator):
        """
        Get generator of source bar.
        ----------------------------------------------------------------------------------------------------------------
        :param dataIterator: generator.
        :return:
        """
        for singleFileData in dataIterator:
            for sourceDict in singleFileData:
                yield sourceDict

    def convertToBar(self, sourceDict):
        """
        Convert source bar to vnpy database bar.
        ----------------------------------------------------------------------------------------------------------------
        :param sourceDict: dict
        :return: dict
        """
        raise NotImplementedError


class RqConverter(FormatConverter):
    """
    Converter of Rice quant data.
    --------------------------------------------------------------------------------------------------------------------
    Source csv file format:
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
    """

    def __init__(self, adaptor):
        super(RqConverter, self).__init__(adaptor)
        self.name = 'rqConverter'
        self.dataDirName = DIR_RQ_PRICE_DATA
        self.lastTradeDate = None

    @staticmethod
    def getSymbol(filename):
        return filename.split('.')[0]

    def convertToBar(self, sourceDict):
        bar = self.bar
        for key in self.generalKey:
            bar[key] = float(sourceDict[key])
        bar['openInterest'] = float(sourceDict['open_interest']) if sourceDict['open_interest'] != '' else 0.0
        bar['datetime'], bar['date'], bar['time'] = self.parseDatetime(sourceDict['datetime'])
        tradingDate = sourceDict['trading_date']
        if tradingDate != self.lastTradeDate:
            self.adaptor.logger.info('Symbol:{} Trading date: {} is saving..'.format(bar['symbol'], tradingDate))
            self.lastTradeDate = tradingDate
        return bar


class JaqsConverter(FormatConverter):
    """
    Converter of Jaqs data.
    --------------------------------------------------------------------------------------------------------------------
    Source csv file format:
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
        super(JaqsConverter, self).__init__(adaptor)
        self.name = 'jaqsConverter'
        self.dataDirName = DIR_JAQS_PRICE_DATA

    @staticmethod
    def getSymbol(filename):
        return filename.split('_')[0]

    def getFiles(self, path, skipDbExisted=True):
        existedDayMap = dict()
        for path, subDir, files in os.walk(path):
            if files:
                files = [f for f in files if f.endswith('csv')]
                for fileName in files:
                    if skipDbExisted:
                        symbol, _sd, _st, endDate, _et = parseFilename(fileName)
                        if existedDayMap.get(symbol) is None:
                            self.adaptor.logger.info(
                                "{}:Getting existed day from database. Please wait...".format(symbol))
                            existedDayMap[symbol] = self.adaptor.getDbExistedDataDay(symbol)
                        if endDate in existedDayMap[symbol]:
                            self.adaptor.logger.info('{} is existed in db.'.format(endDate))
                            continue
                    self.adaptor.logger.info('[Saving..] {}'.format(fileName))
                    yield os.path.join(path, fileName)

    def convertToBar(self, sourceDict):
        bar = self.bar
        for key in self.generalKey:
            bar[key] = float(sourceDict[key])
        bar['openInterest'] = float(sourceDict['oi']) if sourceDict['oi'] != '' else 0.0
        time_ = '{:0>6d}'.format(int(sourceDict['time']))
        dtStr = '{} {}'.format(sourceDict['date'], time_)
        dtTuple = self.parseDatetime(dtStr)
        bar['datetime'], bar['date'], bar['time'] = dtTuple
        return bar