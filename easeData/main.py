# coding:utf-8

import os
import logging
from functions import (initDataDir, )
from database import MongoDbConnector, VnpyAdaptor
from collector import JaqsDataCollector
from log import Logger


def getLogger(level):
    logger = Logger('mainLogger')
    logger.setLevel(level)
    logger.addConsoleHandler()
    logger.addFileHandler()
    return logger


def getDbAdaptor(logger, dp):
    mongo = MongoDbConnector()
    return VnpyAdaptor(mongo, logger, dp)


def getCollector(logger, dp, dbAdaptor):
    collector = JaqsDataCollector(logger, dp)
    collector.connectApi()
    collector.setDbAdaptor(dbAdaptor)
    return collector


def main():
    dp = initDataDir()
    level = logging.DEBUG

    logger = getLogger(level)
    dbAdaptor = getDbAdaptor(logger, dp)
    collector = getCollector(logger, dp, dbAdaptor)

    # collector.downloadBarByContract(symbol='rb')
    # collector.downloadAllMainContractBar(start='2018-01-01', skipSymbol=['T', 'TF', 'TS', 'sc'])
    # collector.downloadAllContinuousMainContract(skipSymbol=['T', 'TS', 'TF', 'sc', 'IH', 'IF', 'IC'])

    filePath = dbAdaptor.getContractPath('rb')
    dbAdaptor.filesToDb(path=filePath)


if __name__ == '__main__':
    main()
