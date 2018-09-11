# coding:utf-8

import logging
from functions import (initDataDirectory, )
from dbConnector import MongoDbConnector, VnpyAdaptor
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
    dp = initDataDirectory()
    level = logging.DEBUG

    logger = getLogger(level)
    dbAdaptor = getDbAdaptor(logger, dp)
    collector = getCollector(logger, dp, dbAdaptor)

    collector.downloadAllMainContractBar(start='2018-01-01', skipSymbol=['T', 'TF', 'TS', 'sc'])
    collector.downloadAllContinuousMainContract(skipSymbol=['T', 'TS', 'TF', 'sc', 'IH', 'IF', 'IC'])


if __name__ == '__main__':
    main()
