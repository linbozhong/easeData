# coding:utf-8

import unittest
from datetime import datetime
from easeData.const import *
from easeData.database import (VnpyAdaptor)

vnpyAdaptor = VnpyAdaptor()

vnpyAdaptor.setFreq('bar')
vnpyAdaptor.setActiveConverter('JQDataConverter')
vnpyAdaptor.setTargetPath('future', 'sp9999')

csvLoader = vnpyAdaptor.csvLoader


class TestCSVFilesLoader(unittest.TestCase):
    def setUp(self):
        self.obj = csvLoader

    def testGetFilePaths(self):
        path = self.obj.getTargetPath()
        print(path)
        gen = self.obj.getFilePaths(path)
        for path in gen:
            print(path)

    def testLoadFiles(self):
        path = self.obj.getTargetPath()
        gen = self.obj.loadFiles(path)
        csv1 = gen.next()
        print(csv1)


class TestVnpyAdaptor(unittest.TestCase):
    def setUp(self):
        self.obj = vnpyAdaptor

    def testInit(self):
        print(self.obj.freq)
        print(self.obj.activeConverter)
        print(self.obj.targetPath)

    def testConverterParseDt(self):
        converter = self.obj.activeConverter
        print(converter.parseDatetime('2018-09-20 21:08:48'))
        print(converter.parseDatetime('20180920 210848'))
        print(converter.parseDatetime('2018-09-20 210848'))
        print(converter.parseDatetime('2018-09-20'))

    def testJQDataIsFileExisted(self):
        self.obj.setActiveConverter('JQDataConverter')
        converter = self.obj.activeConverter
        for i in range(1, 13):
            print('{}:{}'.format(i, converter.isMonthExisted('a9999', 2018, i)))

        filename = '{}_{}_{}.csv'.format('a9999', datetime.today().year, datetime.today().month)
        print('is current month:{}'.format(converter.isCurrentMonthCsvBar(filename)))
        print('{}:{}'.format(filename, converter.isCsvBarInDB(filename)))

    def testJQDataRmCurrentMonthBar(self):
        self.obj.setActiveConverter('JQDataConverter')
        converter = self.obj.activeConverter
        # converter.rmCurrentMonthBarCsv()
        converter.rmEmptyBarCsv()

    def testJqDataBarFilesToDb(self):
        self.obj.setActiveConverter('JQDataConverter')
        vnpyAdaptor.setTargetPath('future')
        self.obj.filesToDb()

    def testJQDataDailyFilesToDb(self):
        self.obj.setActiveConverter('JQDataConverter')
        self.obj.setFreq('daily')
        self.obj.setTargetPath('future')
        self.obj.filesToDb()

    def testJaqsDataIsFileExisted(self):
        self.obj.setActiveConverter('JaqsDataConverter')
        converter = self.obj.activeConverter
        print(converter.getDbExistedDataDay('rb1905'))

        filename = 'rb1905_20181123-210100_to_20181126-150000.csv'
        print(converter.isCsvBarInDB(filename))

    def testJaqsFilesToDb(self):
        self.obj.setActiveConverter('JaqsDataConverter')
        self.obj.setTargetPath('future', 'rb1905')
        self.obj.filesToDb()

    def testRqDataFilesToDb(self):
        self.obj.setActiveConverter('RQDataConverter')
        self.obj.setTargetPath('future', 'rb88')
        self.obj.filesToDb()


if __name__ == '__main__':
    unittest.main()
