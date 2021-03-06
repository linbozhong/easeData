# coding:utf-8

import os
import unittest
import easeData.functions as fn
from easeData.collector import JaqsDataCollector, JQDataCollector, RQDataCollector
from easeData.const import *
# from easeData.database import MongoDbConnector, VnpyAdaptor
from datetime import datetime

# 创建聚宽收集器实例
jqdataCollector = JQDataCollector()

# 创建米筐收集器实例
rqdataCollector = RQDataCollector()

# 创建jaqs收集器实例
jaqsDataCollector = JaqsDataCollector()


def printDict(dictData):
    for key, value in dictData.items():
        print('{}:{}'.format(key, value))


class TestJQDataCollector(unittest.TestCase):
    def setUp(self):
        self.obj = jqdataCollector

    def testConnect(self):
        self.obj.connectApi(user='test', token='testtoken')
        self.obj.connectApi()

    def testRunSdkMethod(self):
        self.obj.info('Run logger mehtod')
        self.obj.get_all_securities('futures', '2018-11-20')
        self.obj.get_price('RB1905.XSGE', '2018-06-20', '2018-11-20')

    def testMonthToRange(self):
        print(self.obj._monthToRange(2018, 10))
        print(self.obj._monthToRange(2018, 11))
        print(self.obj._monthToRange(2019, 2))
        print(self.obj._monthToRange(2001, 2))

    def testConvertFutureSymbol(self):
        print(self.obj.convertFutureSymbol('rb9999'))
        print(self.obj.convertFutureSymbol('rb1901'))
        print(self.obj.convertFutureSymbol('m1901'))
        print(self.obj.convertFutureSymbol('SR1901'))
        print(self.obj.convertFutureSymbol('IF1812'))
        print(self.obj.convertFutureSymbol('sc1901'))
        print(self.obj.jqSymbolMap)

    def testGetTradeDay(self):
        date = datetime(2019, 1, 1)
        print(self.obj.getNextTradeDay(date), type(self.obj.getNextTradeDay(date)))
        print(self.obj.getPreTradeDay(date), type(self.obj.getPreTradeDay(date)))
        print("is Trade-day")
        print(self.obj.isTradeDay(datetime(2019, 3, 10)))
        print(self.obj.isTradeDay(datetime(2019, 3, 11)))


    def testGetFutureExchangeMap(self):
        print(self.obj.getFutureExchangeMap())
        # print(self.obj.getPopularFuture())

    def testGetPopular(self):
        print(self.obj.getPopularFuture())

    def testGetDominantSymbol(self):
        print(self.obj.getDominantContinuousSymbolMap())

    def testGetPath(self):
        print(self.obj.getPriceDataPath())
        print(self.obj.getBasicDataPath())
        print(self.obj.getPricePath('stock', 'bar'))
        print(self.obj.getPricePath('future', 'bar'))
        print(self.obj.getPricePath('future', 'bar', 'rb1905'))

    def testGetPrice(self):
        begin = datetime(2018, 11, 22)
        end = datetime(2018, 11, 23)
        # df = self.obj.get_price('RB9999.XSGE', start_date='2018-11-22', end_date='2018-12-01', frequency='1m')
        df = self.obj.get_price('CU9999.XSGE', start_date=begin, end_date=end, frequency='1m')
        df.to_csv(fn.getTestPath('testjq.csv'), encoding='utf-8-sig')
        print(df.iloc[0])
        print(df.iloc[-1])

    def testDownloadContinuousDaily(self):
        self.obj.downloadContinuousDaily('cu9999')

    def testDownloadAllContinuousDaily(self):
        self.obj.downloadAllContinuousDaily()

    def testDownloadOptionData(self):
        self.obj.downloadOptionData('basic')
        # for i in range(100):
        #     self.obj.downloadOptionData('daily')

    def testDownloadOptionDailyByDate(self):
        today = datetime(2019, 1, 1)
        df = self.obj.downloadOptionDailyByDate(today)

    def testDownloadAllOptionDaily(self):
        start = datetime(2019, 5, 1)
        # self.obj.downloadAllOptionDaily(start)
        # self.obj.downloadAllOptionDaily()

        self.obj.downloadAllOptionGreece(start)
        # self.obj.downloadAllOptionGreece()

    def testGetContinuousBarByMonth(self):
        self.obj.downloadContinuousBarByMonth('rb9999', 2018, 11)
        # self.obj.downloadContinuousBarByMonth('cu9999', 2018, 11, overwrite=True)
        self.obj.downloadContinuousBarByMonth('AP9999', 2017, 11)
        self.obj.downloadContinuousBarByMonth('sc9999', 2018, 3)
        self.obj.downloadContinuousBarByMonth('sc9999', 2018, 2)
        self.obj.downloadContinuousBarByMonth('sc9999', 2018, 1)
        self.obj.downloadContinuousBarByMonth('sp9999', 2018, 11, overwrite=True)
        # self.obj.downloadContinuousBarByMonth('hc9999', 2018, 11, overwrite=True)
        # self.obj.downloadContinuousBarByMonth('sp9999', 2018, 11, overwrite=True)
        # self.obj.downloadContinuousBarByMonth('cu9999', 2018, 12, skipThisMonth=True)
        # self.obj.downloadContinuousBarByMonth('rb9999', 2018, 11, True)

    def testGetAllContinuousBarByMonth(self):
        # self.obj.downloadAllContinuousBarByMonth(2018, 10)
        self.obj.downloadAllContinuousBarByMonth(2018, 11, overwrite=True)

    def testGetAllContinuousBarByRange(self):
        # varieties = ['cu', 'al', 'sc']
        # self.obj.downloadAllContinuousBarByRange('2018-01-01', '2018-11-01', varieties=varieties)

        self.obj.downloadAllContinuousBarByRange('2005-01-01', '2006-12-01', skipThisMonth=True)
        # self.obj.downloadAllContinuousBarByRange('2016-01-01', '2016-12-01')

    def testUpdateContinuousBar(self):
        self.obj.updateCsvContinuousBar('rb9999')
        self.obj.updateCsvContinuousBar('cu9999')

    def testUpdateAllContinuousBar(self):
        self.obj.updateAllCsvContinuousBar()

    def testUpdateDb(self):
        self.obj.updateDb('daily', 'ru9999')
        self.obj.updateDb('bar', 'ru9999')

    def testUpdateAllDb(self):
        # self.obj.updateAllDb('daily', ['cu9999', 'a9999', 'AP9999'])
        # self.obj.updateAllDb('bar', ['cu9999', 'a9999', 'AP9999'])
        self.obj.updateAllDb('daily')
        self.obj.updateAllDb('bar', exclude=['rb88'])


class TestRQDataCollector(unittest.TestCase):
    def setUp(self):
        self.obj = rqdataCollector

    def testGetRequiredFilePath(self):
        print(self.obj.getRequiredFilePath('test.csv'))

    def testGetFutureMainContractDate(self):
        histDate = self.obj.getFutureMainContractDate()
        printDict(histDate)

    def testGetCurrentMainContract(self):
        curMain = self.obj.getCurrentMainContract()
        print(curMain)


class TestJaqsCollector(unittest.TestCase):
    def setUp(self):
        self.obj = jaqsDataCollector
        self.setData()

    def setData(self):
        self.filename1 = 'rb1705_20160527-210100_to_20160530-150000.csv'
        self.filename2 = 'rb1705_20160527-210100_to_20160530-000001.csv'

    def testApi(self):
        self.assertTrue(self.obj._sdk is not None)
        print(self.obj._sdk)

    def testApiFunction(self):
        df, msg = self.obj._sdk.daily('rb1810.SHF', start_date='2018-01-01', end_date='2018-08-15')
        print(df)

    def testQueryInstrumentInfo(self):
        df = self.obj._queryInstrumentInfo()
        print(df.head(5))

    def testGetInstrumentByType(self):
        df = self.obj._queryInstrumentInfoByType(self.obj.INST_TYPE_FUTURE)
        print(df.head(5))
        df = self.obj._queryInstrumentInfoByType(self.obj.INST_TYPE_STOCK)
        print(df.head(5))

    def testGetBarFileName(self):
        print('FileName: {}'.format(self.obj.getBarFilename('rb1905', 20180101, 210001, 20180101, 2, 'csv')))

    def testParseBarFilename(self):
        print(self.obj.parseBarFilename(self.filename1))
        print(self.obj.parseBarFilename(self.filename2))

    def testGetDbAdaptor(self):
        adaptor = self.obj.getDbAdaptor()
        print(adaptor)
        print(adaptor.activeConverter)

    def testGetTypenameMap(self):
        typenameMap = self.obj.getInstTypeToNameMap()
        printDict(typenameMap)

    def testGetBasicDataFilePath(self):
        types = self.obj.getInstTypeToNameMap().keys()
        for type_ in types:
            print(self.obj.getBasicDataFilePath(type_))

    def testGetFutureCurrentMainContract(self):
        map_ = self.obj.getFutureCurrentMainContract()
        print(type(map_))
        printDict(map_)

    def testSetFutureCurrentMainContract(self):
        map_ = self.obj.getFutureCurrentMainContract()
        print(type(map_))
        printDict(map_)

        print('Original main contract:')
        originalBak = self.obj.futureCurrentMainContractMap['rb']
        print(originalBak)

        print('Change main contract:')
        self.obj.setFutureCurrentMainContract({'rb': 'rb1810'})
        self.obj.futureCurrentMainContractMap = None
        self.obj.getFutureCurrentMainContract()
        print(self.obj.futureCurrentMainContractMap['rb'])

        print('Restore main contract:')
        self.obj.setFutureCurrentMainContract({'rb': originalBak})
        self.obj.futureCurrentMainContractMap = None
        self.obj.getFutureCurrentMainContract()
        print(self.obj.futureCurrentMainContractMap['rb'])

    def testGetPopular(self):
        print(self.obj.getPopularFuture())

    def testFutureExchangeMap(self):
        print('Future exchange map:')
        printDict(self.obj.getFutureExchangeMap())
        print('Add exchange:')
        print(self.obj.addFutureExchangeToSymbol('rb1810'))
        print(self.obj.addFutureExchangeToSymbol('AP810'))
        print(self.obj.addFutureExchangeToSymbol('m1901'))
        print(self.obj.addFutureExchangeToSymbol('IF1809'))

    def testCZCE(self):
        print(self.obj.isCZCE('WT1909'))
        print(self.obj.isCZCE('WT909'))
        print(self.obj.isCZCE('AP909'))
        print(self.obj.isCZCE('ZN909'))
        print(self.obj.isCZCE('zn1809'))
        print(self.obj.adjustCZCESymbol('WT1909'))
        print(self.obj.adjustCZCESymbol('AP1909'))

    def testGetFutureHistoryMainContractMap(self):
        contractMap = self.obj.getFutureHistoryMainContractMap()
        printDict(contractMap)

    def testGetFutureContractLifespan(self):
        print(self.obj.getFutureContractLifespan('rb1801'))
        print(self.obj.getFutureContractLifespan('rb'))
        print(self.obj.getFutureContractLifespan('ag'))
        print(self.obj.getFutureContractLifespan('al'))

    def testGetMainContractListByPeriod(self):
        print('rb:')
        print(self.obj.getMainContractListByPeriod('rb', '2017-01-01'))
        # print(self.obj.getMainContractListByPeriod('rb', '2017-01-01', '2017-12-01'))
        # print(self.obj.getMainContractListByPeriod('JR', '2018-07-01'))

        print(self.obj.getMainContractListByPeriod('c', '2018-01-01', '2018-11-13'))

        # WT is from 1999-01-04 to 2012-11-22
        print("Begin > end: 2018-7-1 to 2012-11-22")
        print(self.obj.getMainContractListByPeriod('WT', start='2018-07-01'))

        print("Non-intersection: 1992-1-1 to 1995-1-1")
        print(self.obj.getMainContractListByPeriod('WT', '1992-01-01', '1995-01-01'))

        print("Begin > end: 1999-1-4 to 1991-1-1")
        print(self.obj.getMainContractListByPeriod('WT', end='1991-01-01'))

        print("Touch at end:2012-11-22 to 2012-11-22")
        print(self.obj.getMainContractListByPeriod('WT', start='2012-11-22'))

        print("Touch at begin:1999-1-4 to 1999-1-4")
        print(self.obj.getMainContractListByPeriod('WT', end='1999-01-04'))

        print("Another touch at begin:1993-7-1 to 1999-1-4")
        print(self.obj.getMainContractListByPeriod('WT', start='1993-07-01', end='1999-01-04'))

        print("Another touch at end:2012-11-22 to 2015-1-1")
        print(self.obj.getMainContractListByPeriod('WT', start='2012-11-22', end='2015-01-01'))

        print("None of date")
        print(self.obj.getMainContractListByPeriod('WT'))

        print("Error of input date")
        print(self.obj.getMainContractListByPeriod('WT', start="2008-01-01", end="2002-01-01"))

        print("sc")
        print(self.obj.getMainContractListByPeriod('sc', start="2018-08-01"))

        print("fu")
        print(self.obj.getMainContractListByPeriod('fu', start="2018-01-01", end='2018-11-15'))

    def testGetMainContractSymbolByDate(self):
        print(self.obj.getMainContractSymbolByDate('rb', '2014-12-29'))
        print(self.obj.getMainContractSymbolByDate('rb', '2013-04-09'))
        print(self.obj.getMainContractSymbolByDate('rb', '2017-04-19'))
        print(self.obj.getMainContractSymbolByDate('ru', '2017-04-19'))
        print(self.obj.getMainContractSymbolByDate('ru', '2018-04-20'))
        print(self.obj.getMainContractSymbolByDate('rb', '2018-04-20'))
        print(self.obj.getMainContractSymbolByDate('ag', '2013-01-04'))
        print(self.obj.getMainContractSymbolByDate('IH', '2013-10-25'))

    def testGetBar(self):
        print(self.obj.getBar('rb1905'))
        print(self.obj.getBar('rb1905', trade_date='2018-11-28'))
        print(self.obj.getBar('rb'))

    def testGetDaily(self):
        print(self.obj.getDaily('rb', '2017-06-01', '2018-11-14'))
        print(self.obj.getDaily('AP', '2018-01-01', '2018-11-15'))

    def testGetBasicData(self):
        df = self.obj.getBasicData(self.obj.INST_TYPE_FUTURE)
        print(df.head(5))

    def testGetTradeCal(self):
        array = self.obj.getTradeCal()
        print(array)

    def testGetTradingDayArray(self):
        print(self.obj.getTradingDayArray('2018-10-01', '20181015'))
        print('Newest T-days:')
        print(self.obj.getTradingDayArray('2018-07-01'))

    def testGetPreAndNextTradingDay(self):
        print('Next T-Day:')
        print(self.obj.getNextTradeDay('2018-10-01'))
        print(type(self.obj.getNextTradeDay('2018-10-01')))
        print(self.obj.getNextTradeDay('20181001'))
        print(self.obj.getNextTradeDay('20290201'))
        print(self.obj.getNextTradeDay('20180904', nDays=20000))
        print(self.obj.getNextTradeDay('20180904', nDays=5))
        print('Pre T-Day:')
        print(self.obj.getPreTradeDay('2018-10-09'))
        print(self.obj.getPreTradeDay('20181009'))
        print(self.obj.getPreTradeDay('19901219'))
        print(self.obj.getPreTradeDay('19901221', nDays=10))



    def testDownloadBarByContract(self):
        # self.obj.downloadBarByContract('rb1810', '2018-07-01', '2018-07-15')
        # self.obj.downloadBarByContract('rb1901', )
        # self.obj.downloadBarByContract('rb1901', '2018-07-01')
        # self.obj.downloadBarByContract('rb1901', '2018-07-01')
        # self.obj.downloadBarByContract('rb1310')
        # self.obj.downloadBarByContract('rb1905')
        self.obj.downloadBarByContract('rb1905', start='2018-11-15', saveToDb=True)
        self.obj.downloadBarByContract('rb1905', start='2018-11-15', saveToDb=True, refresh=True)
        # self.obj.downloadBarByContract('rb1810', '2018-07-01', '2018-07-15', refresh=True)
        # self.obj.downloadBarByContract('rb', '2018-01-01', '2018-07-15')
        # self.obj.downloadBarByContract('ru')
        # self.obj.downloadBarByContract('cu')
        # self.obj.downloadBarByContract('rb1710', start='2017-04-19', end='2017-04-20')
        # self.obj.downloadBarByContract('rb1505', start='2014-12-29', end='2014-12-29')

    def testDownloadLatestMainContractBar(self):
        skip = ['T', 'TC', 'TS']
        self.obj.downloadCurrentMainContractBar(skipSymbol=skip)

    def testDownloadAllContinuousMainDaily(self):
        self.obj.downloadAllContinuousMainDaily('2018-06-01', end='2018-11-13')

    def testSaveBarToDb(self):
        df = self.obj.getBar('cu1810', trade_date='2018-09-03')[0]
        self.obj.saveBarToDb(df)

    def testIsCompleteFile(self):
        files = ['ag1812_20180827-210100_to_20180827-211300.csv',
                 'rb1810_20171103-210100_to_20171106-150000.csv',
                 'rb1810_20180102-090100_to_20180102-150000.csv']
        for file_ in files:
            print(self.obj.isCompleteFile(file_))

    def testCleanIncompleteFile(self):
        self.obj.cleanIncompleteFile()


if __name__ == '__main__':
    unittest.main()
