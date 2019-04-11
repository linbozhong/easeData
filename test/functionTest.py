# coding:utf-8

import os
import unittest
from collections import OrderedDict
from datetime import datetime

import easeData.functions as fn
from easeData.const import *


class TestFunction(unittest.TestCase):
    def setUp(self):
        self.setTestData()

    def setTestData(self):
        self.dt = datetime.today()
        self.dateWithDash = '2018-01-01'
        self.dateWithoutDash = '20180201'
        self.settingFilename = 'setting_test.json'
        self.testPath = fn.getTestDataDir()
        self.settingPath = os.path.join(self.testPath, self.settingFilename)

    def testSaveSetting(self):
        # self.setTestData()

        testDict = OrderedDict()
        testDict['z'] = 100
        testDict['a'] = 9
        testDict['m'] = 80
        print(testDict)
        fn.saveSetting(testDict, self.settingPath)

    def testLoadSetting(self):
        self.testRestoreDefaultDataDir()
        self.setTestData()  # 重载数据

        print(fn.loadSetting(self.settingPath, object_pairs_hook=OrderedDict))

    def testGetDirs(self):
        print(fn.getCurrentDir())
        print(fn.getParentDir())
        print(fn.getDataDir())
        print(fn.getTestDataDir())
        print(fn.getTestPath('test.csv'))

    def testSetDataDir(self):
        fn.setDataDir()

    def testChangeDataDir(self):
        newPath = os.path.join(os.path.dirname(fn.getParentDir()), 'testDataPath')
        fn.setDataDir(newPath)

    def testRestoreDefaultDataDir(self):
        defaultPath = os.path.join(os.path.dirname(fn.getParentDir()), DIR_DATA_ROOT)
        fn.setDataDir(defaultPath)

    def testDateConvert(self):
        # self.setTestData()
        print('Remove Dash. {} to {}'.format(self.dateWithDash, fn.rmDateDash(self.dateWithDash)))
        print('Add Dash. {} to {}'.format(self.dateWithoutDash, fn.addDateDash(self.dateWithoutDash)))
        print('TodayStr: {}'.format(fn.getTodayStr()))
        print('strToDate 1: {}'.format(fn.strToDate(self.dateWithDash)))
        print('strToDate 2: {}'.format(fn.strToDate(self.dateWithoutDash)))
        print(fn.strToDate('2018-04-20'))
        print(fn.dateToStr(self.dt))
        print(fn.dateToDtStr(self.dt))

    def testLastFri(self):
        print(fn.getLastFriday())
        print(fn.getLastFriday(datetime(2019, 3, 8)))
        print(fn.getLastFriday(datetime(2019, 3, 9)))


if __name__ == '__main__':
    unittest.main()
