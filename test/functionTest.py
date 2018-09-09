# coding:utf-8

import os
import unittest
import easeData.functions as fn
from easeData.const import *
from collections import OrderedDict


def printDict(dictData):
    for key, value in dictData.items():
        print('{}:{}'.format(key, value))


class TestFunction(unittest.TestCase):
    divideSymbol = '=' * 30

    def setUp(self):
        print('{}{}{}'.format(self.divideSymbol, 'Begin Cutting Line', self.divideSymbol))

    def tearDown(self):
        print('{}{}{}\n\n'.format(self.divideSymbol, 'End Cutting Line', self.divideSymbol))

    def setData(self):
        self.symbol = 'rb1810'
        self.symbols = ['cu1810', 'IF1808', 'AP805']
        self.exchange = 'SHF'
        self.fullSymbol = 'rb1810.SHF'
        self.contracts = OrderedDict()
        self.fpath = 'test_setting_temp.json'
        self.dateWithDash = '2018-01-01'
        self.dateWithoutDash = '20180201'
        self.filename = 'rb1705_20160527-210100_to_20160530-150000.csv'

    def testSetting(self):
        self.setData()
        self.contracts['z'] = 100
        self.contracts['a'] = 9
        self.contracts['m'] = 80
        print(self.contracts)
        fn.saveSetting(self.contracts, self.fpath)
        print(fn.loadSetting(self.fpath, object_pairs_hook=OrderedDict))
        os.remove(self.fpath)

    def testGetDirectory(self):
        print(fn.getCurrentDirectory())
        print(fn.getParentDirectory())

    def testSymbolOperate(self):
        self.setData()
        print('Add exchange:{}'.format(fn.addExchange(self.symbol, self.exchange)))
        print('Remove exchange:{}'.format(fn.rmExchange(self.fullSymbol)))
        underlyingSymbols = [fn.getUnderlyingSymbol(symbol) for symbol in self.symbols]
        print('Underlying symbol: {}'.format(underlyingSymbols))

    def testDateConvert(self):
        self.setData()
        print('Remove Dash. {} to {}'.format(self.dateWithDash, fn.rmDateDash(self.dateWithDash)))
        print('Add Dash. {} to {}'.format(self.dateWithoutDash, fn.addDateDash(self.dateWithoutDash)))
        print('TodayStr: {}'.format(fn.getTodayStr()))
        print('strToDate 1: {}'.format(fn.strToDate(self.dateWithDash)))
        print('strToDate 2: {}'.format(fn.strToDate(self.dateWithoutDash)))
        print(fn.strToDate('2018-04-20'))

    def testMakeFileDir(self):
        dataPath = fn.getDataDirectory()
        print(fn.mkFileDir(dataPath, 'cu1810'))

    def testMakeFileName(self):
        self.setData()
        print('FileName: {}'.format(fn.mkFilename(self.symbol, 20180101, 210001, 20180101, 2, 'csv')))

    def testParseFilename(self):
        self.setData()
        print(fn.parseFilename(self.filename))

    def testDataPath(self):
        pathBackup = fn.getDataDirectory()
        print('Parent directory: {}'.format(pathBackup))
        print('Data directory: {}'.format(fn.getDataDirectory()))

        testDataPath = os.path.join(os.getcwd(), 'testData')
        print('Set test data directory: {}'.format(fn.setDataDirectory(testDataPath)))
        fn.initDataDirectory()
        print('Set original data directory: {}'.format(fn.setDataDirectory(pathBackup)))
        os.rmdir(testDataPath)

    def testMainContractInfo(self):
        rqFolder = os.path.join(fn.getParentDirectory(), DIR_EXTERNAL_DATA, DIR_RQ_DATA)
        curMainPath = os.path.join(rqFolder, FILE_CURRENT_MAIN_CONTRACT)
        histMainPath = os.path.join(rqFolder, FILE_MAIN_CONTRACT_HISTORY)
        curMain = fn.getCurrentMainContract(curMainPath)
        histMain = fn.getMainContract(histMainPath)
        print(curMain)
        print('=' * 30)
        printDict(histMain)


if __name__ == '__main__':
    unittest.main()
