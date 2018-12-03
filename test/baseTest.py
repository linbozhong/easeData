# coding:utf-8

import os
import unittest

import easeData.functions as fn
from easeData.const import *
from easeData.text import *
from easeData.base import DataVendor, LoggerWrapper

dv = DataVendor()
lw = LoggerWrapper()


class TestDataVendor(unittest.TestCase):
    def setUp(self):
        self.setTestData()

    def setTestData(self):
        self.symbol = 'rb1810'
        self.exchange = 'SHF'
        self.fullSymbol = 'rb1810.SHF'
        self.symbols = ['cu1810', 'IF1808', 'AP805']

        self.symbolLst = ['AG9999.XSGE']
        self.symbolLst2 = ['AL9999.XSGE', 'AU9999.XSGE', 'CF9999.XZCE', 'A9999.XDCE']

        self.filename = 'rb1705_20160527-210100_to_20160530-150000.csv'

    def testSymbolOperate(self):
        print('Add exchange:{}'.format(dv.addExchange(self.symbol, self.exchange)))
        print('Remove exchange:{}'.format(dv.rmExchange(self.fullSymbol)))

        underlyingSymbols = [dv.getUnderlyingSymbol(symbol) for symbol in self.symbols]
        print('Underlying symbol: {}'.format(underlyingSymbols))

    def testGetExchange(self):
        print(dv.getExchange(self.symbolLst))
        print(dv.getExchange(self.symbolLst2))

    def testGetMonthBusinessDay(self):
        print(dv.getMonthBusinessDay(2018, 2))
        print(dv.getMonthBusinessDay(2017, 2))
        print(dv.getMonthBusinessDay(2018, 9))
        print(dv.getMonthBusinessDay(2018, 10))
        print(dv.getMonthBusinessDay(2018, 7))

    def testInitPath(self):
        print(dv.getBasicDataPath())
        print(dv.getPriceDataPath())

    def testGetPricePath(self):
        print(dv.getPricePath('stock', 'bar'))
        print(dv.getPricePath('stock', 'bar', '000001'))
        print(dv.getPricePath('future', 'bar'))
        print(dv.getPricePath('future', 'bar', 'rb1905'))
        print(dv.getPricePath('future', 'daily', 'rb1905'))

    def testParseFilename(self):
        print(lw.parseFilename(self.filename))


if __name__ == '__main__':
    unittest.main()
