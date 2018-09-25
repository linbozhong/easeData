# coding:utf-8

import logging
import unittest
import os
import easeData.functions as fn
from easeData.const import *
from easeData.dbConnector import VnpyAdaptor, MongoDbConnector, FormatConverter, RqConverter, JaqsConverter
from easeData.log import Logger

dataPath = fn.initDataDirectory()
dbConnector = MongoDbConnector()
logger = Logger('logger')
logger.setLevel(logging.INFO)
logger.addConsoleHandler()
logger.addFileHandler()

vnpyAdaptor = VnpyAdaptor(dbConnector, logger, dataPath)

genConvertor = FormatConverter(vnpyAdaptor)
rqConverter = RqConverter(vnpyAdaptor)
jaqsConverter = JaqsConverter(vnpyAdaptor)

vnpyAdaptor.addConverter(rqConverter)
vnpyAdaptor.addConverter(jaqsConverter)

rqDataPath = os.path.join(dataPath, DIR_RQ_PRICE_DATA, FUTURE)
jaqsDataPath = os.path.join(dataPath, DIR_JAQS_PRICE_DATA, FUTURE)

class TestVnpyAdaptor(unittest.TestCase):
    def testAdaptorSetDb(self):
        vnpyAdaptor.setDb()
        print(vnpyAdaptor.db)
        print(vnpyAdaptor.db.collection_names())

    def testAdaptorAddConverter(self):
        print(vnpyAdaptor.converters)

    def testAdaptorGetExistedDay(self):
        print(vnpyAdaptor.getDbExistedDataDay('rb'))
        print(vnpyAdaptor.getDbExistedDataDay('rb88'))

    def testAdaptorGetContractPath(self):
        print(vnpyAdaptor.getContractPath('jaqsConverter', 'rb'))
        print(vnpyAdaptor.getContractPath('rqConverter', 'rb1901'))

    # def testGenSourceBarDict(self):
    #     path = os.path.join(vnpyAdaptor.dataPath, DIR_JAQS_PRICE_DATA, FUTURE)
    #     genFiles = vnpyAdaptor.genFilePaths(path)
    #     print(genFiles.send(None))
    #
    #     csvDictReader = vnpyAdaptor.genAllDayBars(genFiles)
    #     print(csvDictReader.send(None))
    #
    #     bar = vnpyAdaptor.genBar(csvDictReader)
    #     print(bar.send(None))
    #
    #     print('Convert Bar to vnpy format:')
    #     print(vnpyAdaptor.convertBar(bar.send(None)))
    #
    # def testGenFilePath(self):
    #     path = os.path.join(vnpyAdaptor.dataPath, DIR_JAQS_PRICE_DATA, FUTURE, 'rb', 'rb')
    #     genFiles = vnpyAdaptor.genFilePaths(path)
    #     while True:
    #         try:
    #             genFiles.send(None)
    #         except StopIteration:
    #             print('Over.')
    #             break
    #
    # def testGenFilePath2(self):
    #     path = os.path.join(vnpyAdaptor.dataPath, DIR_JAQS_PRICE_DATA, FUTURE, 'rb', 'rb')
    #     vnpyAdaptor.genFilePaths(path)

    def testConverterParseDt(self):
        print(genConvertor.parseDatetime('2018-09-20 21:08:48'))
        print(genConvertor.parseDatetime('20180920 210848'))
        print(genConvertor.parseDatetime('2018-09-20 210848'))

    def testConverterGeneralKey(self):
        print(rqConverter.generalKey)
        print(jaqsConverter.generalKey)

    def testRqConverterGetFiles(self):
        fileGen = rqConverter.getFiles(os.path.join(rqDataPath, 'test'))
        for i in range(3):
            print(fileGen.next())

    def testJaqsConverterGetFiles(self):
        # fileGen = jaqsConverter.getFiles(os.path.join(jaqsDataPath, 'rb'))
        fileGen = jaqsConverter.getFiles(os.path.join(jaqsDataPath, 'rb'), skipDbExisted=False)
        for i in range(3):
            print(fileGen.next())

    def testRqConverterGetData(self):
        fileGen = rqConverter.getFiles(os.path.join(rqDataPath, 'test'))
        csvGen = rqConverter.getData(fileGen)
        for csvItem in csvGen:
            print(rqConverter.bar['symbol'])
            print(csvItem)
            for i in range(3):
                rec = csvItem.next()
                print(rec)
                print(rqConverter.convertToBar(rec))

    def testJaqsConverterGetData(self):
        fileGen = jaqsConverter.getFiles(os.path.join(jaqsDataPath, 'test'), skipDbExisted=False)
        csvGen = jaqsConverter.getData(fileGen)
        for csvItem in csvGen:
            print(jaqsConverter.bar['symbol'])
            print(csvItem)
            for i in range(3):
                rec = csvItem.next()
                print(rec)
                print(jaqsConverter.convertToBar(rec))

    def testVnpyAdaptorFilesToDb(self):
        # path = vnpyAdaptor.getContractPath(rqConverter.name, 'rb88')
        # vnpyAdaptor.filesToDb(rqConverter.name, path)

        path2 = vnpyAdaptor.getContractPath(jaqsConverter.name, 'AP901')
        vnpyAdaptor.filesToDb(jaqsConverter.name, path2)


if __name__ == '__main__':
    unittest.main()
