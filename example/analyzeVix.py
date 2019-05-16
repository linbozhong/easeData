# coding:utf-8


import os
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, time
from easeData.functions import getDataDir, getTestPath
from easeData.const import *


def set_flag(dt):
    minute = dt.minute
    if minute != 0:
        minute = minute - 30
    return '{}-minute'.format(minute)


def analyze_single_atm(df):
    df['pre_close'] = df.iloc[0].close
    df['abs_change'] = df['close'] - df['pre_close']
    df['change_ratio'] = (df['close'] - df['pre_close']) / df['pre_close']
    df['time_flag'] = df.index.map(lambda dt: set_flag(dt))
    return df


def analyze_single_atm_open(df):
    df['pre_close'] = df.iloc[0].close
    df['abs_change'] = df['open'] - df['pre_close']
    df['change_ratio'] = (df['close'] - df['pre_close']) / df['pre_close']
    df['time_flag'] = df.index.map(lambda dt: set_flag(dt))
    return df


def analyze_single_atm_range(df, start_time, end_time):
    df = df.between_time(start_time, end_time)
    start_close = df.iloc[0].close

    new_df = pd.DataFrame(df.iloc[-1])
    new_df = new_df.T
    new_df['pre_close'] = start_close
    new_df['abs_change'] = new_df['close'] - new_df['pre_close']
    new_df['change_ratio'] = (new_df['close'] - new_df['pre_close']) / new_df['pre_close']
    return new_df


def analyze_qvix():
    filename = 'vixBar.csv'
    fp = os.path.join(getDataDir(), RESEARCH, OPTION, 'qvix', filename)
    df = pd.read_csv(fp)

    df.datetime = df.datetime.map(lambda timestamp: datetime.fromtimestamp(timestamp / 1000))
    df['pre_close'] = df['close'].shift(1)
    df['open_change'] = (df['open'] - df['pre_close']) / df['pre_close']
    df['open_change'].hist(bins=100)
    df['open_change'].hist(bins=100, cumulative=True, normed=True)


def analyze_atm_range(start_time, end_time):
    start_l = start_time.split(':')
    start_time = time(int(start_l[0]), int(start_l[1]))
    end_l = end_time.split(':')
    end_time = time(int(end_l[0]), int(end_l[1]))

    filename = 'atm_continuous_bar.csv'
    fp = os.path.join(getDataDir(), 'research', 'option', 'dailytask', 'atm_continuous_bar.csv')
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df = df.groupby('tradeDay').apply(analyze_single_atm_range, start_time, end_time)

    all_count = len(df)
    higer_count = len(df[df.abs_change > 0])
    lower_count = len(df[df.abs_change <= 0])
    min = df.abs_change.min()
    max = df.abs_change.max()
    mean = df.abs_change.mean()
    median = df.abs_change.median()

    # df2 = df.iloc[-240:]
    # volume_mean = df2.volume.mean()

    print('-' * 30)
    print(u'昨日平值期权今日%s-%s价和对比：' % (':'.join(start_l), ':'.join(end_l)))
    print('-' * 30)
    print(u'总交易日：%s' % all_count)
    print(u'后面的时间价和更高交易日：%s' % higer_count)
    print(u'后面的时间价和更低交易日：%s' % lower_count)
    print(u'价和跌最低：%.2f' % (min * 10000))
    print(u'价和涨最高：%.2f' % (max * 10000))
    print(u'价和涨跌平均值：%.2f' % (mean * 10000))
    print(u'价和涨跌中位数：%.2f' % (median * 10000))
    # print(u'近一年成交量平均值：%.2f' % volume_mean)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(u'%s-%s' % (':'.join(start_l), ':'.join(end_l)))
    ax.set_ylabel('trade_days')
    df.abs_change.hist(bins=100, ax=ax)
    filename = '{}_open.png'.format(u'%s-%s' % (start_time.strftime('%H%M'), end_time.strftime('%H%M')))
    fig.savefig(getTestPath(filename))


def analyze_atm():
    filename = 'strategyAtmReturnNextTradedayOpen.csv'
    fp = getTestPath(filename)
    df = pd.read_csv(fp, index_col=0, parse_dates=True)

    # df = df.groupby('tradeDay').apply(analyze_single_atm)
    df = df.groupby('tradeDay').apply(analyze_single_atm_open)

    grouped = df.groupby('time_flag')
    groupedDict = dict(list(grouped))

    keys = groupedDict.keys()
    keys.sort()

    print(u'昨日平值期权今日开盘n分钟后，价和与昨日收盘对比统计情况统计结果(单位：元)：')
    for key in keys:
        print('' * 30)
        df = groupedDict[key]
        # if key != '0-minute':
        if key == '1-minute':
            all_count = len(df)
            higer_count = len(df[df.abs_change > 0])
            lower_count = len(df[df.abs_change <= 0])
            min = df.abs_change.min()
            max = df.abs_change.max()
            mean = df.abs_change.mean()
            median = df.abs_change.median()

            df2 = df.iloc[-240:]
            volume_mean = df2.volume.mean()

            print('-' * 30)
            print(u'开盘%s分钟后：' % key[0])
            print('-' * 30)
            print(u'总交易日：%s' % all_count)
            print(u'价和比昨收盘更高交易日：%s' % higer_count)
            print(u'价和比昨收盘更低交易日：%s' % lower_count)
            print(u'价和跌最低：%.2f' % (min * 10000))
            print(u'价和涨最高：%.2f' % (max * 10000))
            print(u'价和涨跌平均值：%.2f' % (mean * 10000))
            print(u'价和涨跌中位数：%.2f' % (median * 10000))
            print(u'近一年成交量平均值：%.2f' % volume_mean)

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(key)
            ax.set_ylabel('trade_days')
            df.abs_change.hist(bins=200, ax=ax)
            filename = '{}_open.png'.format(key)
            fig.savefig(getTestPath(filename))


if __name__ == '__main__':
    # analyze_atm()
    # analyze_atm_range('09:31', '10:00')
    # analyze_atm_range('14:30', '14:55')
    # analyze_atm_range('14:00', '14:55')
    # analyze_atm_range('09:31', '11:30')
    # analyze_atm_range('13:01', '14:55')
    analyze_atm_range('09:01', '14:55')

