# coding:utf-8

import unittest
from easeData.analyze import CorrelationAnalyzer
from datetime import datetime

corrAnalyzer = CorrelationAnalyzer()


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
        # self.obj.setFreq('bar')
        # self.obj.setNum(12)
        # self.obj.setEndDay('2018-11-30')
        # self.obj.setExclude(self.obj.cffex)
        # df = self.obj.getCorrelationArray()
        # self.obj.plotCorrelationArray(df)

        self.obj.setFreq('daily')
        self.obj.setNum(12)
        self.obj.setEndDay('2018-12-10')
        self.obj.setExclude(['AP', 'fu', 'sc'])
        df = self.obj.getCorrelationArray()
        self.obj.plotCorrelationArray(df)


if __name__ == '__main__':
    unittest.main()
