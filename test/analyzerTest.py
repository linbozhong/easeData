# coding:utf-8

import unittest
import os
from easeData.functions import getTestPath
from easeData.analyze import CorrelationAnalyzer, PositionDiffPlotter, SellBuyRatioPlotter
from datetime import datetime

corrAnalyzer = CorrelationAnalyzer()
positionDiffPlotter = PositionDiffPlotter()
sellBuyRatioPlotter = SellBuyRatioPlotter()


class TestCorrAnalyzer(unittest.TestCase):
    def setUp(self):
        self.obj = corrAnalyzer
        # self.obj.setNum(3)

    def testIsTradeDay(self):
        print(self.obj.isTradeDay('2018-10-01'))
        print(self.obj.isTradeDay('2018-12-09'))
        print(self.obj.isTradeDay('2018-12-07'))

    def testGetDateRange(self):
        self.obj.setFreq('daily')
        print(self.obj.getDateRange())

        self.obj.setFreq('bar')
        print(self.obj.getDateRange())

    def testGetPrice(self):
        dt1 = datetime(2018, 1, 1)
        dt2 = datetime(2018, 11, 1)
        dt3 = datetime(2018, 11, 29, 9)
        dt4 = datetime(2018, 11, 29, 15)

        self.obj.setFreq('daily')
        self.obj.getPrice(dt1, dt2)

        self.obj.setFreq('bar')
        self.obj.setExclude(self.obj.cffex)
        # self.obj.getPrice(dt3, dt4, exclude=self.obj.cffex)
        self.obj.getPrice(dt3, dt4)

    def testGetCorrelationArray(self):
        # self.obj.setFreq('daily')
        # self.obj.getCorrelationArray()

        self.obj.setFreq('bar')
        self.obj.setEndDay('2018-11-30')
        self.obj.setExclude(self.obj.cffex)
        self.obj.getCorrelationArray()

    def testPlotCorrelationArray(self):
        self.obj.setFreq('bar')
        self.obj.setNum(30)
        self.obj.setEndDay('2018-12-10')
        self.obj.setExclude(self.obj.cffex)
        df = self.obj.getCorrelationArray()
        self.obj.plotCorrelationArray(df)

        # self.obj.setFreq('daily')
        # self.obj.setNum(12)
        # self.obj.setEndDay('2018-12-10')
        # self.obj.setExclude(['AP', 'fu', 'sc'])
        # df = self.obj.getCorrelationArray()
        # self.obj.plotCorrelationArray(df)


class TestPositionDiffTrend(unittest.TestCase):
    def setUp(self):
        self.obj = positionDiffPlotter

    def testGetTradingContractInfo(self):
        df = self.obj.getTradingContract()
        df.to_csv(getTestPath('optionTradingContract.csv'), encoding='utf-8-sig')

    def testGetDisplayContract(self):
        df = self.obj.getMainContract()
        df.to_csv(getTestPath('optionMainContract.csv'), encoding='utf-8-sig')

    def testGetCompContract(self):
        df = self.obj.getCompContract()
        df.to_csv(getTestPath('optionDisplayContract.csv'), encoding='utf-8-sig')

    def testGetDailyPrice(self):
        contract = '10001562.XSHG'
        df = self.obj.getDailyPrice(contract)
        df.to_csv(getTestPath('optionPrice.csv'), encoding='utf-8-sig')

    def testGetContractName(self):
        contracts = ['10001562.XSHG', '10001542.XSHG']
        print([self.obj.getContractName(c) for c in contracts])

    def testGetGroupedCode(self):
        codes = self.obj.getGoupedCode()
        print(codes)

    def testGetMainContractDailyPrice(self):
        price = self.obj.getAllContractDailyPrice()
        for code, df in price.items():
            fp = getTestPath('{}.csv'.format(code))
            df.to_csv(fp, encoding='utf-8-sig')

    def testGetPosDiff(self):
        df = self.obj.getPositionDiff()
        df.to_csv(getTestPath('posDiff.csv'), encoding='utf-8-sig')

    def testGetPosition(self):
        df = self.obj.getPosition()
        df.to_csv(getTestPath('position.csv'), encoding='utf-8-sig')

    def testPlotPosDiff(self):
        self.obj.setQueryMonth('1901')
        self.obj.plotData(self.obj.getPositionDiff)

    def testPlotPosition(self):
        # self.obj.setQueryMonth('1902')
        # self.obj.plotData(self.obj.getPosition)

        mission = ['1901', '1902', '1903', '1906']
        for m in mission:
            self.obj.setQueryMonth(m)
            self.obj.plotData(self.obj.getPosition)
            self.obj.cleanCache()


class TestSellBuyRatioPlotter(unittest.TestCase):
    def setUp(self):
        self.obj = sellBuyRatioPlotter

    def testGetContractInfo(self):
        df = self.obj.getContractInfo()
        df.to_csv(getTestPath('contractInfo.csv'), encoding='utf-8-sig')

    def testGetContractType(self):
        print(self.obj.getContractType('10000033.XSHG'))
        print(self.obj.contractTypeDict)

    def testCalcRatioByDate(self):
        filename = 'option_daily_2018-12-27.csv'
        fp = os.path.join(self.obj.jqsdk.getPricePath('option', 'daily'), filename)
        d = self.obj.calcRatioByDate(fp)
        print(d)

    def testGetRatio(self):
        df = self.obj.getRatio()
        df.to_csv(getTestPath('ratio.csv'), encoding='utf-8-sig')

    def testPlotRatio(self):
        self.obj.plotRatio()


if __name__ == '__main__':
    unittest.main()
