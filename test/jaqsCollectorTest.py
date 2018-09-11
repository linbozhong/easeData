# coding:utf-8

import os
import unittest
import logging
import easeData.functions as fn
from easeData.collector import JaqsDataCollector
from easeData.const import *
from easeData.log import Logger
from easeData.dbConnector import MongoDbConnector, VnpyAdaptor

dataPath = fn.initDataDirectory()

logger = Logger('logger')
logger.setLevel(logging.INFO)
logger.addConsoleHandler()
logger.addFileHandler()

dbConnector = MongoDbConnector()
vnpyDbAdaptor = VnpyAdaptor(dbConnector, logger, dataPath)

jaqsCollector = JaqsDataCollector(logger, dataPath)
jaqsCollector.connectApi()
jaqsCollector.setDbAdaptor(vnpyDbAdaptor)


def printDict(dictData):
    for key, value in dictData.items():
        print('{}:{}'.format(key, value))


class TestJaqsCollector(unittest.TestCase):
    divideSymbol = '=' * 30

    def setUp(self):
        print('{}{}{}'.format(self.divideSymbol, 'Begin Cutting Line', self.divideSymbol))

    def tearDown(self):
        print('{}{}{}\n\n'.format(self.divideSymbol, 'End Cutting Line', self.divideSymbol))

    def setData(self):
        pass

    def testDataPath(self):
        print(jaqsCollector.dataPath)
        basicDataPath = os.path.join(jaqsCollector.dataPath, DIR_JAQS_BASIC_DATA)
        priceDataPath = os.path.join(jaqsCollector.dataPath, DIR_JAQS_PRICE_DATA)
        self.assertTrue(os.path.exists(basicDataPath))
        self.assertTrue(os.path.exists(priceDataPath))

    def testTypenameMap(self):
        typenameMap = jaqsCollector.getInstTypeToNameMap()
        printDict(typenameMap)

    def testGetBasicDataFilePath(self):
        types = jaqsCollector.getInstTypeToNameMap().keys()
        for type_ in types:
            print(jaqsCollector.getBasicDataFilePath(type_))

    def testApi(self):
        self.assertTrue(jaqsCollector.api is not None)
        print(jaqsCollector.api)

    def testCZC(self):
        print(jaqsCollector.isCZC('WT1909'))
        print(jaqsCollector.isCZC('WT909'))
        print(jaqsCollector.isCZC('AP909'))
        print(jaqsCollector.isCZC('ZN909'))
        print(jaqsCollector.isCZC('zn1809'))
        print(jaqsCollector.adjustSymbolOfCZC('WT1909'))
        print(jaqsCollector.adjustSymbolOfCZC('AP1909'))

    def testOperateCurrentMainContract(self):
        map_ = jaqsCollector.getFutureCurrentMainContract()
        print(type(map_))
        printDict(map_)

        print('Original main contract:')
        originalBak = jaqsCollector.futureCurrentMainContractMap['rb']
        print(originalBak)

        print('Change main contract:')
        jaqsCollector.setFutureCurrentMainContract({'rb': 'rb1810'})
        jaqsCollector.futureCurrentMainContractMap = None
        jaqsCollector.getFutureCurrentMainContract()
        print(jaqsCollector.futureCurrentMainContractMap['rb'])

        print('Restore main contract:')
        jaqsCollector.setFutureCurrentMainContract({'rb': originalBak})
        jaqsCollector.futureCurrentMainContractMap = None
        jaqsCollector.getFutureCurrentMainContract()
        print(jaqsCollector.futureCurrentMainContractMap['rb'])

    def testGetInstrumentInfo(self):
        df = jaqsCollector.queryInstrumentInfo()
        print(df.head(5))

    def testGetInstrumentByType(self):
        df = jaqsCollector.queryInstrumentInfoByType(jaqsCollector.INST_TYPE_FUTURE)
        print(df.head(5))
        df = jaqsCollector.queryInstrumentInfoByType(jaqsCollector.INST_TYPE_STOCK)
        print(df.head(5))

    def testGetTradeCal(self):
        df = jaqsCollector.querySecTradeCal()
        print(df.head(5))

    def testLoadTradeCal(self):
        print(jaqsCollector.getTradeCal())
        print('Try again:')
        print(jaqsCollector.getTradeCal())

    def testGetTradingDay(self):
        print('Next T-Day:')
        print(jaqsCollector.getNextTradeDay('2018-10-01'))
        print(type(jaqsCollector.getNextTradeDay('2018-10-01')))
        print(jaqsCollector.getNextTradeDay('20181001'))
        print(jaqsCollector.getNextTradeDay('20290201'))
        print(jaqsCollector.getNextTradeDay('20180904', nDays=20000))
        print(jaqsCollector.getNextTradeDay('20180904', nDays=5))
        print('Pre T-Day:')
        print(jaqsCollector.getPreTradeDay('2018-10-09'))
        print(jaqsCollector.getPreTradeDay('20181009'))
        print(jaqsCollector.getPreTradeDay('19901219'))
        print(jaqsCollector.getPreTradeDay('19901221', nDays=10))
        print('T-days:')
        print(jaqsCollector.getTradingDayArray('2018-01-01', '20180401'))
        print('Newest T-days:')
        print(jaqsCollector.getTradingDayArray('2018-07-01'))

    def testLoadBasicData(self):
        df = jaqsCollector.getBasicData(jaqsCollector.INST_TYPE_FUTURE)
        print(df.head(5))

    def testExchangeMap(self):
        print('Future exchange map:')
        printDict(jaqsCollector.getFutureExchangeMap())
        print('Add exchange:')
        print(jaqsCollector.addFutureExchangeToSymbol('rb1810'))
        print(jaqsCollector.addFutureExchangeToSymbol('AP810'))
        print(jaqsCollector.addFutureExchangeToSymbol('m1901'))
        print(jaqsCollector.addFutureExchangeToSymbol('IF1809'))

    def testGetMainContract(self):
        contractMap = jaqsCollector.getFutureHistoryMainContractMap()
        printDict(contractMap)
        print('Main contract list in period:')
        print('rb:')
        print(jaqsCollector.getMainContractListByPeriod('rb', '2017-01-01'))
        # print(jaqsCollector.getMainContractListByPeriod('rb', '2017-01-01', '2017-12-01'))
        # print(jaqsCollector.getMainContractListByPeriod('JR', '2018-07-01'))

        # WT is from 1999-01-04 to 2012-11-22
        print("Begin > end: 2018-7-1 to 2012-11-22")
        print(jaqsCollector.getMainContractListByPeriod('WT', start='2018-07-01'))

        print("Non-intersection: 1992-1-1 to 1995-1-1")
        print(jaqsCollector.getMainContractListByPeriod('WT', '1992-01-01', '1995-01-01'))

        print("Begin > end: 1999-1-4 to 1991-1-1")
        print(jaqsCollector.getMainContractListByPeriod('WT', end='1991-01-01'))

        print("Touch at end:2012-11-22 to 2012-11-22")
        print(jaqsCollector.getMainContractListByPeriod('WT', start='2012-11-22'))

        print("Touch at begin:1999-1-4 to 1999-1-4")
        print(jaqsCollector.getMainContractListByPeriod('WT', end='1999-01-04'))

        print("Another touch at begin:1993-7-1 to 1999-1-4")
        print(jaqsCollector.getMainContractListByPeriod('WT', start='1993-07-01', end='1999-01-04'))

        print("Another touch at end:2012-11-22 to 2015-1-1")
        print(jaqsCollector.getMainContractListByPeriod('WT', start='2012-11-22', end='2015-01-01'))

        print("None of date")
        print(jaqsCollector.getMainContractListByPeriod('WT'))

        print("Error of input date")
        print(jaqsCollector.getMainContractListByPeriod('WT', start="2008-01-01", end="2002-01-01"))

        print("sc")
        print(jaqsCollector.getMainContractListByPeriod('sc', start="2018-08-01"))

        print("fu")
        print(jaqsCollector.getMainContractListByPeriod('fu', start="2018-01-01"))

    def testGetMainContractSymbolByDate(self):
        print(jaqsCollector.getMainContractSymbolByDate('rb', '2014-12-29'))
        print(jaqsCollector.getMainContractSymbolByDate('rb', '2017-04-19'))
        print(jaqsCollector.getMainContractSymbolByDate('ru', '2017-04-19'))
        print(jaqsCollector.getMainContractSymbolByDate('ru', '2018-04-20'))
        print(jaqsCollector.getMainContractSymbolByDate('rb', '2018-04-20'))
        print(jaqsCollector.getMainContractSymbolByDate('ag', '2013-01-04'))
        print(jaqsCollector.getMainContractSymbolByDate('IH', '2013-10-25'))

    def testGetFutureContractLifespan(self):
        print(jaqsCollector.getFutureContractLifespan('rb1801'))
        print(jaqsCollector.getFutureContractLifespan('rb'))
        print(jaqsCollector.getFutureContractLifespan('ag'))
        print(jaqsCollector.getFutureContractLifespan('al'))

    def testIsCompleteFile(self):
        files = ['ag1812_20180827-210100_to_20180827-211300.csv',
                 'rb1810_20171103-210100_to_20171106-150000.csv',
                 'rb1810_20180102-090100_to_20180102-150000.csv']
        for file_ in files:
            print(jaqsCollector.isCompleteFile(file_))

    def testCleanIncompleteFile(self):
        jaqsCollector.cleanIncompleteFile()

    def testQueryBar(self):
        jaqsCollector.queryBar('rb1901', trade_date='2018-08-22')
        jaqsCollector.queryBar('rb1810', trade_date='2018-07-04')
        jaqsCollector.queryBar('rb1901')
        jaqsCollector.queryBar('rb', trade_date='2017-04-19')

    def testDownloadBarByContract(self):
        # jaqsCollector.downloadBarByContract('rb1810', '2018-07-01', '2018-07-15')
        # jaqsCollector.downloadBarByContract('rb1901', '2018-07-01', saveToDb=True)
        jaqsCollector.downloadBarByContract('rb1901', '2018-07-01')
        jaqsCollector.downloadBarByContract('rb1901', '2018-07-01')
        # jaqsCollector.downloadBarByContract('rb1810', '2018-07-01', '2018-07-15', refresh=True)
        # jaqsCollector.downloadBarByContract('rb', '2018-01-01', '2018-07-15')
        # jaqsCollector.downloadBarByContract('rb')
        # jaqsCollector.downloadBarByContract('ru')
        # jaqsCollector.downloadBarByContract('cu')
        # jaqsCollector.downloadBarByContract('rb1710', start='2017-04-19', end='2017-04-20')
        # jaqsCollector.downloadBarByContract('rb1505', start='2014-12-29', end='2014-12-29')

    def testDownloadLatestMainContractBar(self):
        skip = ['T', 'TC', 'TS']
        jaqsCollector.downloadCurrentMainContractBar(skipSymbol=skip)

    def testSaveBarToDb(self):
        df = jaqsCollector.queryBar('cu1810', trade_date='2018-09-03')[0]
        jaqsCollector.saveBarToDb('vnpy', df)


if __name__ == '__main__':
    unittest.main()
