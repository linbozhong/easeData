# coding:utf-8

import logging
import unittest
import os
import easeData.functions as fn
from easeData.const import *
from easeData.dbConnector import VnpyAdaptor, MongoDbConnector
from easeData.log import Logger

dataPath = fn.initDataDirectory()
dbConnector = MongoDbConnector()
logger = Logger('logger')
logger.setLevel(logging.INFO)
logger.addConsoleHandler()
logger.addFileHandler()
vnpyAdaptor = VnpyAdaptor(dbConnector, logger, dataPath)


class TestVnpyAdaptor(unittest.TestCase):
    def testParseDatetime(self):
        print(vnpyAdaptor.parseDatetime('20180823', '210001'))
        print(vnpyAdaptor.parseDatetime('20180823', '1'))

    def testGenSourceBarDict(self):
        path = os.path.join(vnpyAdaptor.dataPath, DIR_JAQS_PRICE_DATA, FUTURE)
        genFiles = vnpyAdaptor.genFilePaths(path)
        print(genFiles.send(None))

        csvDictReader = vnpyAdaptor.genAllDayBars(genFiles)
        print(csvDictReader.send(None))

        bar = vnpyAdaptor.genBar(csvDictReader)
        print(bar.send(None))

        print('Convert Bar to vnpy format:')
        print(vnpyAdaptor.convertBar(bar.send(None)))

    def testSetDb(self):
        vnpyAdaptor.setDb()
        print(vnpyAdaptor.db.collection_names())

    def testGetExistedDay(self):
        days = vnpyAdaptor.getDbExistedDataDay('rb')
        print(days)

    def testGenFilePath(self):
        path = os.path.join(vnpyAdaptor.dataPath, DIR_JAQS_PRICE_DATA, FUTURE, 'rb', 'rb')
        genFiles = vnpyAdaptor.genFilePaths(path)
        while True:
            try:
                genFiles.send(None)
            except StopIteration:
                print('Over.')
                break

    def testGenFilePath2(self):
        path = os.path.join(vnpyAdaptor.dataPath, DIR_JAQS_PRICE_DATA, FUTURE, 'rb', 'rb')
        vnpyAdaptor.genFilePaths(path)

    def testGetSymbolPath(self):
        print(vnpyAdaptor.getContractPath('rb'))
        print(vnpyAdaptor.getContractPath('rb1901'))



if __name__ == '__main__':
    unittest.main()
