# coding:utf-8

import os
import codecs
import json
from datetime import datetime
from const import *


def loadSetting(filename, encoding='utf-8', **kwargs):
    """
    读取配置文件。
    --------------------------------------------------------------------------------------------------------------------
    :param filename: string
            配置文件名
    :param encoding: string
            编码
    :param kwargs:
    :return: dict
            配置信息的字典
    """
    with codecs.open(filename, 'r', encoding) as f:
        setting = json.load(f, **kwargs)
    return setting


def saveSetting(obj, filename, encoding='utf-8', **kwargs):
    """
    保存配置文件。
    --------------------------------------------------------------------------------------------------------------------
    :param obj: dict.
            要保存的对象数据
    :param filename: string
            配置文件名
    :param encoding: string
            编码
    :param kwargs:
    :return:
    """
    with codecs.open(filename, 'w', encoding) as f:
        json.dump(obj, f, indent=4, **kwargs)


def getCurrentDir():
    """
    获取模块的根目录
    --------------------------------------------------------------------------------------------------------------------
    :return: string.
            文件目录
    """
    return os.path.dirname(os.path.abspath(__file__))


def getParentDir():
    """
    获取模块根目录的父目录
    --------------------------------------------------------------------------------------------------------------------
    :return: string.
            文件目录
    """
    return os.path.dirname(getCurrentDir())


def getDataDir():
    """
    从config.json文件中获取数据根目录。
    --------------------------------------------------------------------------------------------------------------------
    :return: string.
            文件目录
    """
    setting = loadSetting(os.path.join(getCurrentDir(), FILE_SETTING))
    return setting['dataPath']


def setDataDir(path=None):
    """
    设置数据目录并保存到config.json
    --------------------------------------------------------------------------------------------------------------------
    :param path: string.
            要设置的数据根目录
    :return: path: string.
            数据根目录
    """
    if path is None:
        path = getDataDir()
        if path is None:
            path = os.path.join(os.path.dirname(getParentDir()), DIR_DATA_ROOT)
    f = os.path.join(getCurrentDir(), FILE_SETTING)
    setting = loadSetting(f)
    setting['dataPath'] = path
    saveSetting(setting, f)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def getTestDataDir():
    """
    获取测试数据目录，用于单元测试
    --------------------------------------------------------------------------------------------------------------------
    :return: string
            文件目录
    """
    path = os.path.join(setDataDir(), DIR_DATA_TEST)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def getTestPath(filename):
    """
    生成测试文件路径，用于单元测试
    --------------------------------------------------------------------------------------------------------------------
    :return: string
            文件路径
    """
    return os.path.join(getTestDataDir(), filename)


def rmDateDash(date):
    """
    日期转换，去除破折号
    --------------------------------------------------------------------------------------------------------------------
    :param date: string.
            日期格式: %Y-%m-%d
    :return: string.
            日期格式: %Y%m%d
    """
    return ''.join(date.split('-'))


def addDateDash(date):
    """
    日期转换，添加破折号
    --------------------------------------------------------------------------------------------------------------------
    :param date: string.
            日期格式 %Y%m%d
    :return: string.
            日期格式 %Y-%m-%d
    """
    return '-'.join([date[:4], date[4:6], date[6:]])


def getTodayStr():
    """
    获取当天的日期（字符串格式）
    :return: string.
            日期格式 %Y-%m-%d
    """
    return datetime.now().strftime('%Y-%m-%d')


def strToDate(dateStr):
    """
    日期字符串转换为datetime，比datetime的方法快。
    --------------------------------------------------------------------------------------------------------------------
    :param dateStr: string
            日期格式 %Y-%m-%d or %Y%m%d
    :return: datetime
    """
    if '-' in dateStr:
        year, month, day = dateStr.split('-')
    else:
        year, month, day = dateStr[0:4], dateStr[4:6], dateStr[6:]
    return datetime(int(year), int(month), int(day))


def dateToStr(date):
    """
    datetime转换为日期字符串，比datetime的方法快。
    --------------------------------------------------------------------------------------------------------------------
    :param date: datetime
    :return: string.
            日期格式 %Y-%m-%d
    """
    return '{:0>4d}-{:>02d}-{:>02d}'.format(date.year, date.month, date.day)
