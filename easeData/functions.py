# coding:utf-8

import re
import os
import codecs
import json
import pandas as pd
from datetime import datetime, timedelta
from collections import OrderedDict
from const import *


def loadSetting(filename, encoding='utf-8', **kwargs):
    """
    Load json setting file.
    --------------------------------------------------------------------------------------------------------------------
    :param filename: string
    :param encoding: string
    :param kwargs:
    :return: dict
    """
    with codecs.open(filename, 'r', encoding) as f:
        setting = json.load(f, **kwargs)
    return setting


def saveSetting(obj, filename, encoding='utf-8', **kwargs):
    """
    Save json setting file.
    --------------------------------------------------------------------------------------------------------------------
    :param obj: dict.
    :param filename: string
    :param encoding: string
    :param kwargs:
    :return:
    """
    with codecs.open(filename, 'w', encoding) as f:
        json.dump(obj, f, indent=4, **kwargs)


def addExchange(symbol, exchange):
    """
    :param symbol: string, eg.'rb1805'
    :param exchange: string, eg. 'SHF'
    :return: string eg. 'rb1805.SHF'
    """
    return '{0}.{1}'.format(symbol, exchange)


def rmExchange(symbol):
    """
    :param symbol: string, eg. 'rb1805.SHF'
    :return: string. eg.'rb1805'
    """
    return symbol.split('.')[0]


def rmDateDash(date):
    """
    :param date: string. Format: %Y-%m-%d
    :return: string. Format %Y%m%d
    """
    return ''.join(date.split('-'))


def addDateDash(date):
    """
    :param date: string. Format %Y%m%d
    :return: string. Format %Y-%m-%d
    """
    return '-'.join([date[:4], date[4:6], date[6:]])


def getTodayStr():
    """
    :return: string. Format %Y-%m-%d
    """
    return datetime.now().strftime('%Y-%m-%d')


def strToDate(dateStr):
    """
    convert date string to datetime more quickly.
    --------------------------------------------------------------------------------------------------------------------
    :param dateStr: Format %Y-%m-%d or %Y%m%d
    :return: datetime
    """
    if '-' in dateStr:
        year, month, day = dateStr.split('-')
    else:
        year, month, day = dateStr[0:4], dateStr[4:6], dateStr[6:]
    return datetime(int(year), int(month), int(day))


def dateToStr(date):
    """
    convert datetime to date string more quickly.
    --------------------------------------------------------------------------------------------------------------------
    :param date: datetime
    :return: string. Format %Y-%m-%d
    """
    return '{:0>4d}-{:>02d}-{:>02d}'.format(date.year, date.month, date.day)


def mkFileDir(dataPath, symbol):
    """
    Generate price data directory name and create it if not existed.
    --------------------------------------------------------------------------------------------------------------------
    :param dataPath: string. The price date path.
    :param symbol: string. eg.'rb1805'
    :return: string. eg. 'jaqsPriceData/future/rb/rb1805'
    """
    folder = os.path.join(dataPath, DIR_JAQS_PRICE_DATA, 'future', getUnderlyingSymbol(symbol), symbol)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def mkFilename(symbol, fromDate, fromTime, toDate, toTime, fileExt):
    """
    Generate price data filename.
    --------------------------------------------------------------------------------------------------------------------
    :param symbol: string
    :param fromDate: int
    :param fromTime: int
    :param toDate: int
    :param toTime: int
    :param fileExt: string
    :return: string.eg. 'rb1805_20180101-2101_to_20180301-1500.csv'
    """
    return '{}_{:0>8d}-{:0>6d}_to_{:0>8d}-{:0>6d}.{}'.format(symbol, fromDate, fromTime, toDate, toTime, fileExt)


def parseFilename(filename):
    """
    Parse filename to get data info.
    --------------------------------------------------------------------------------------------------------------------
    :param filename: string
    :return: tuple(string...)
    """
    pattern = r'[_.-]'
    symbol, beginDate, beginTime, _to, endDate, endTime, _ext = re.split(pattern, filename)
    return symbol, beginDate, beginTime, endDate, endTime


def getUnderlyingSymbol(symbol):
    """
    Get underlying symbol from a symbol which is trade-able.
    --------------------------------------------------------------------------------------------------------------------
    :param symbol: string. a contract symbol, eg. 'rb1805'.
    :return: string. underlying symbol, eg. 'rb'.
    """
    index = 0
    for char in symbol:
        if char.isalpha():
            index += 1
        else:
            break
    return symbol[0:index]


def getCurrentDirectory():
    """
    Get current directory of this app file.
    --------------------------------------------------------------------------------------------------------------------
    :return: string. file path.
    """
    return os.path.dirname(os.path.abspath(__file__))


def getParentDirectory():
    """
    Get parent directory of this app file.
    --------------------------------------------------------------------------------------------------------------------
    :return: string. file path.
    """
    return os.path.dirname(getCurrentDirectory())


def getDataDirectory():
    """
    Get data directory of this app from config.json.
    --------------------------------------------------------------------------------------------------------------------
    :return: string. file path.
    """
    setting = loadSetting(os.path.join(getCurrentDirectory(), FILE_SETTING))
    return setting['dataPath']


def setDataDirectory(path=None):
    """
    Set data directory of this app and save to config.json.
    --------------------------------------------------------------------------------------------------------------------
    :param path: string. data file path you want to set.
    :return: path: string.
    """
    if path is None:
        path = os.path.join(os.path.dirname(getParentDirectory()), DIR_DATA_ROOT)
    f = os.path.join(getCurrentDirectory(), FILE_SETTING)
    setting = loadSetting(f)
    setting['dataPath'] = path
    saveSetting(setting, f)
    return path


def initDataDirectory(path=None):
    """
    Initial data directory of this app.
    If data path is not exist in config.json, set and create it.
    --------------------------------------------------------------------------------------------------------------------
    :param path: string. data file path you want to set.
    :return: path: string.
    """
    dataPath = getDataDirectory()
    if not dataPath:
        dataPath = setDataDirectory(path)
    if not os.path.exists(dataPath):
        os.mkdir(dataPath)
    return dataPath


def getMainContract(fileName):
    """
    Generate a main contract dict from a csv file.
    Use rqData.py to run in ricequant.com/research environment to download csv file you need.
    --------------------------------------------------------------------------------------------------------------------
    :param fileName: string. csv file.
    :return:
    dict{string: [(string, string, string)]}
        key: symbol. eg.‘rb’
        value: list of tuple(symbol, beginDate, endDate).
    eg.
    {
        'rb': [('rb1801, '2018-01-01', '2018-04-01'),
               (...),
               ...]
        'cu': ...
        ...
    }
    """
    mainContractMap = dict()
    df = pd.read_csv(fileName, encoding='utf-8', low_memory=False, index_col=[0])
    lastDay = df.index.sort_values()[-1]
    columns = df.iteritems()
    for colName, colData in columns:
        if colName not in ['exchange']:
            colData.dropna(inplace=True)
            # colData.drop_duplicates(inplace=True)
            colData.sort_index(inplace=True)
            # colData = list(colData.iteritems())
            colIterator = colData.iteritems()

            def getPeriod(iterator):
                """
                :param iterator: format: eg.[('2018-01-03', 'rb1801'), ('2018-04-05', 'rb1805'), ...]
                :return: list[tuple(string: symbol, string: begin-date, string: end-date)]
                    eg. [('rb1801', '2018-01-03', '2018-05-01'), ('rb1805',.., ..), ...]
                """
                res = []
                oldSymbol = ''
                begin = ''
                end = ''
                for date, symbol in iterator:
                    if colName.islower():
                        symbol = symbol.lower()
                    if symbol != oldSymbol:
                        if oldSymbol != '':
                            res.append((oldSymbol, begin, end))
                        oldSymbol = symbol
                        begin = date
                        end = date
                    else:
                        end = date if date != lastDay else ''
                res.append((oldSymbol, begin, end))
                return res

            result = getPeriod(colIterator)
            mainContractMap[colName] = result

            # # Now colData is a list of tuple. tuple(0) is begin date，tuple(1) is the symbol.
            # # eg.[('2018-01-03', 'rb1801'), ('2018-04-05', 'rb1805'), ...]
            # # list needs to be added the end date to get format like this:[(symbol, begin, end), ...]
            # result = []
            # for index in range(0, len(colData)):
            #     beginDate, symbol = colData[index]
            #     if colName.islower():
            #         symbol = symbol.lower()
            #     try:
            #         endDate = colData[index + 1][0]
            #         endDate = strToDate(endDate)
            #         endDate = endDate - timedelta(days=1)
            #         endDate = dateToStr(endDate)
            #     except IndexError:
            #         endDate = ''
            #     result.append((symbol, beginDate, endDate))

    return mainContractMap


def getCurrentMainContract(fileName):
    """
    Generate a ordered dict from a json file.
    Use rqData.py to run in ricequant.com/research environment to download json file you need.
    Maybe you should adjust the main contract by hand to update depending on market conditions.
    --------------------------------------------------------------------------------------------------------------------
    :param fileName: json file
    :return:
    dict{string: string}
        key: underlying symbol. eg.'rb'
        value: main contact symbol. eg.'rb1810'
    """
    return loadSetting(fileName, object_pairs_hook=OrderedDict)



