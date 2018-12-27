# coding:utf-8
import os
import re
import calendar
import pandas as pd
from log import Logger
from datetime import datetime, timedelta

from const import *
from text import *
from functions import (setDataDir)


class Base(object):
    def __init__(self):
        pass


class LoggerWrapper(Base):
    """
    日志包裹类
    """
    # 用类属性，让所有实例共享一个对象
    _logger = Logger('logger')
    _dataPath = setDataDir()

    def __init__(self):
        super(LoggerWrapper, self).__init__()

    def __getattr__(self, item):
        """
        访问类不直接存在的方法或属性的时候被调用，用于代理访问logger的方法
        """
        return getattr(self._logger, item)

    @staticmethod
    def parseFilename(filename):
        """
        通用文件名解析，把文件名转换成列表。以'-', '.'或'_'做为分界符。
        :param filename: string
        :return: list
                    包含文件名信息的列表
        """
        return re.split(r'[_.-]', filename)


class DataVendor(LoggerWrapper):
    """
    数据供应商基类
    """

    def __init__(self):
        super(DataVendor, self).__init__()
        self.vendor = VENDOR_BASE
        self.basicDp = ''
        self.priceDp = ''

    @staticmethod
    def addExchange(symbol, exchange):
        """
        在尾部添加交易所代码
        ----------------------------------------------------------------------------------------------------------------
        :param symbol: string,
                普通的股票或期货合约代码。如'rb1805'
        :param exchange: string
                数据供应商设定的交易所代码。如'SHF'
        :return: string
                合并的代码，如'rb1805.SHF'

        """
        return '{0}.{1}'.format(symbol, exchange)

    @staticmethod
    def rmExchange(symbol):
        """
        移除尾部的交易所代码
        --------------------------------------------------------------------------------------------------------------------
        :param symbol: string,
                包含交易所信息的期货合约代码， 通常是数据供应商使用的。如'rb1805.SHF'
        :return: string.
                普通的股票或期货合约代码。 如'rb1805'
        """
        return symbol.split('.')[0]

    @staticmethod
    def getUnderlyingSymbol(symbol):
        """
        从期货合约代码获取期货品种代码.
        --------------------------------------------------------------------------------------------------------------------
        :param symbol: string
                期货合约代码, eg. 'rb1805'.
        :return: string
                期货品种代码, eg. 'rb'.
        """
        stack = []
        for char in symbol:
            if char.isalpha():
                stack.append(char)
            else:
                break
        return ''.join(stack)

    @staticmethod
    def getMonthBusinessDay(year, month):
        """
        获取指定月份的工作日列表。
        ----------------------------------------------------------------------------------------------------------------
        :param year: int
                年份
        :param month: int
                月份
        :return:pandas.DatetimeIndex
                日期列表
        """
        start = datetime(year, month, 1)
        _, days = calendar.monthrange(year, month)
        end = start + timedelta(days=days)
        # closed=left，表示仅左区间闭合，不包含右边最后一个
        dateRange = pd.date_range(start, end, freq='B', closed='left')
        return dateRange

    def getExchange(self, symbolList):
        """
        从包含交易所信息的合约列表中提取合约对应交易所的映射字典。
        ----------------------------------------------------------------------------------------------------------------
        :param symbolList: iterable
                合约代码的列表
        :return: dict
                key：品种代码
                value： 交易所代码
        """
        exchangeMap = {}
        for symbol in symbolList:
            underlyingSymbol = self.getUnderlyingSymbol(symbol)
            if underlyingSymbol not in exchangeMap:
                market = symbol.split('.')[-1]
                exchangeMap[underlyingSymbol] = market
        return exchangeMap

    def getBasicDataPath(self):
        """
        获取基础数据目录。
        :return:
        """
        self.basicDp = os.path.join(self._dataPath, self.vendor + DIR_BASIC_DATA)
        if not os.path.exists(self.basicDp):
            os.makedirs(self.basicDp)
        return self.basicDp

    def getPriceDataPath(self):
        """
        获取价格数据目录。
        :return:
        """
        self.priceDp = os.path.join(self._dataPath, self.vendor + DIR_PRICE_DATA)
        if not os.path.exists(self.priceDp):
            os.makedirs(self.priceDp)
        return self.priceDp

    def getBasicPath(self, types):
        """
        获取基础数据文件目录，不存在则创建。
        :param types: string
                可选'future', 'stock', 'option'
        :return:
        """
        root = self.getBasicDataPath()
        path = os.path.join(root, types)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def getPricePath(self, types, freq, symbol=None):
        """
        获取价格csv文件的目录，如果不存在则创建该目录。
        :param types: string
                可选'future','stock', 'option'
        :param freq: string
                可选'daily', 'bar', 'tick'
        :param symbol: string
                股票代码或期货合约代码
        :return:
        """
        root = self.getPriceDataPath()
        if symbol is not None:
            if types == FUTURE:
                if freq == DAILY:
                    # 日线数据少，不需要为每个合约单独建一个文件夹
                    path = os.path.join(root, types, freq, self.getUnderlyingSymbol(symbol))
                else:
                    path = os.path.join(root, types, freq, self.getUnderlyingSymbol(symbol), symbol)
            else:
                path = os.path.join(root, types, freq, symbol)
        else:
            path = os.path.join(root, types, freq)
        if not os.path.exists(path):
            os.makedirs(path)
        return path


# class JaqsDataVendor(DataVendor):
#     """
#     Jaqs数据供应商基类
#     """
#
#     def __init__(self):
#         super(JaqsDataVendor, self).__init__()
#         self.vendor = VENDOR_JAQS
