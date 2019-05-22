# coding:utf-8

import tushare as ts
from datetime import datetime, timedelta
from collections import OrderedDict

# import csv
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# from copy import copy
# from datetime import datetime, time
# from easeData.functions import getDataDir, getTestPath, dateToStr
# from easeData.const import *


today = datetime.now()


def analyze_50etf_amplitude(start=None):
    if start is None:
        start = '2005-02-23'
    df = ts.get_k_data('510050', start=start)
    # print(start)

    df['high_to_open'] = df['high'] - df['open']
    df['low_to_open'] = df['open'] - df['low']
    df['high_to_open_ratio'] = df['high_to_open'] / df['open']
    df['low_to_open_ratio'] = df['low_to_open'] / df['open']

    rise_stat = df['high_to_open_ratio'].describe()
    fall_stat = df['low_to_open_ratio'].describe()

    stat_dict = {'rise': rise_stat, 'fall': fall_stat}

    for k, v in stat_dict.items():
        if k == 'rise':
            name = u'上涨'
        else:
            name = u'下跌'

        print(u"%s幅度统计：" % name)
        print(u'%s幅度均值：%.2f %%' % (name, v['mean'] * 100))
        print(u'%s幅度标准差：%.2f %%' % (name, v['std'] * 100))
        print(u'%s幅度最小值：%.2f %%' % (name, v['min'] * 100))
        print(u'%s幅度最大值：%.2f %%' % (name, v['max'] * 100))
        print(u'%s幅度中位数：%.2f %%' % (name, v['50%'] * 100))
        print(u'%s幅度1/4分位：%.2f %%' % (name, v['25%'] * 100))
        print(u'%s幅度3/4分位：%.2f %%' % (name, v['75%'] * 100))
        print('-' * 30)

    rise_75 = rise_stat['75%']
    fall_75 = fall_stat['75%']
    rise_median = rise_stat['50%']
    fall_median = fall_stat['50%']

    df2 = df[(df['high_to_open_ratio'] > rise_75) | (df['low_to_open_ratio'] > fall_75)]
    count_75 = float(len(df2)) / float(len(df)) * 100

    df3 = df[(df['high_to_open_ratio'] > rise_median) | (df['low_to_open_ratio'] > fall_median)]
    count_median = float(len(df3)) / float(len(df)) * 100

    print(u'总交易日：%s' % len(df))
    print(u'相比开盘价，最大上涨或最大下跌超出各自3/4分位的天数：%s, 占比：%.2f %%' % (len(df2), count_75))
    print(u'相比开盘价，最大上涨或最大下跌超出各自中位数的天数：%s, 占比：%.2f %%' % (len(df3), count_median))
    print(u'')


def analyze_all():
    print('=' * 40)
    print(u'自成立以来，50etf日内涨跌幅统计（最高/最低价与开盘价对比）：')
    print('=' * 40)
    analyze_50etf_amplitude()


def analyze_recent():
    days_dict = OrderedDict()
    days_dict['rect_3_year'] = 365 * 3
    days_dict['rect_1_year'] = 365
    days_dict['rect_6_month'] = 365 / 2
    days_dict['rect_3_month'] = 365 / 4

    name_dict = {
                    'rect_3_year': u'最近3年',
                    'rect_1_year': u'最近1年',
                    'rect_6_month': u'最近6月',
                    'rect_3_month': u'最近3月'
                }

    for k, v in days_dict.items():
        print('=' * 40)
        print(u'%s，50etf日内涨跌幅统计（最高/最低价与开盘价对比）：' % name_dict[k])
        print('=' * 40)
        start = today - timedelta(days=v)
        start = start.strftime('%Y-%m-%d')
        analyze_50etf_amplitude(start=start)


if __name__ == '__main__':
    # analyze_50etf_amplitude()
    analyze_all()
    analyze_recent()
