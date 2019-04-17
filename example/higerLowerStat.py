# coding:utf-8


import tushare as ts
import os
import pandas as pd
from datetime import datetime
from easeData.functions import getDataDir
from easeData.const import *

DATETIME = u'日期'
WEEKDAY = u'星期几'
S_OPEN_CHANGE = u'标的开盘涨跌幅'
VIX_OPEN_CHANGE = u'波值开盘涨跌幅'
VIX_OPEN = u'波值开盘价'
VIX_CLOSE = u'波值收盘价'
VIX_HIGH = u'波值最高价'
VIX_LOW = u'波值最低价'
S_OPEN = u'标的开盘价'
S_CLOSE = u'标的收盘价'
S_HIGH = u'标的最高价'
S_LOW = u'标的最低价'
S_CLOSE_CHANGE = u'标的收盘涨跌幅'
VIX_CLOSE_CHANGE = u'波值收盘涨跌幅'


def analyze_qvix():
    filename = 'qvix_daily.csv'
    fp = os.path.join(getDataDir(), RESEARCH, OPTION, 'qvix', filename)
    df = pd.read_csv(fp, index_col=0, parse_dates=True)

    df['prev_close'] = df.close.shift(1)
    df['vix_open_change'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['vix_close_change'] = (df['close'] - df['open']) / df['open']

    column_name_dict = {
        'open': VIX_OPEN,
        'high': VIX_HIGH,
        'low': VIX_LOW,
        'close': VIX_CLOSE,
        'vix_open_change': VIX_OPEN_CHANGE,
        'vix_close_change': VIX_CLOSE_CHANGE
    }

    df.rename(columns=column_name_dict, inplace=True)
    df.index.name = DATETIME
    df.to_csv('vix.csv', encoding='utf-8-sig')
    return df


def analyze_50etf(start='2015-02-09', end=None, ratio=0.01):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    df = ts.get_k_data('510050', start=start, end=end)
    df['date'] = df['date'].map(lambda date_str: datetime.strptime(date_str, '%Y-%m-%d'))
    df['weekday'] = df['date'].map(lambda date: date.weekday() + 1)
    df.set_index('date', inplace=True)

    df['prev_close'] = df.close.shift(1)
    df['s_open_change'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['s_close_change'] = (df['close'] - df['open']) / df['open']

    column_name_dict = {
        'weekday': WEEKDAY,
        's_open_change': S_OPEN_CHANGE,
        's_close_change': S_CLOSE_CHANGE,
        'open': S_OPEN,
        'close': S_CLOSE,
        'high': S_HIGH,
        'low': S_LOW
    }
    df.rename(
        columns=column_name_dict, inplace=True)
    df.index.name = DATETIME

    df_res = df[abs(df[S_OPEN_CHANGE]) > ratio]

    newItem = [WEEKDAY, S_OPEN_CHANGE, S_CLOSE_CHANGE, S_OPEN, S_HIGH, S_LOW, S_CLOSE]
    df_res = df_res[newItem]
    df_res.to_csv('50etf.csv', encoding='utf-8-sig')
    # print(df_res)
    return df_res


def merge_s_and_vix(s_df, vix_df):
    df = pd.concat([s_df, vix_df], axis=1, join='inner')

    newItem = [WEEKDAY, S_OPEN_CHANGE, VIX_OPEN_CHANGE, VIX_OPEN, VIX_HIGH, VIX_LOW, VIX_CLOSE, S_CLOSE_CHANGE,
               VIX_CLOSE_CHANGE]
    df = df[newItem]
    df.to_csv('50etf_and_vix.csv', encoding='utf-8-sig')


if __name__ == '__main__':
    vix_df = analyze_qvix()
    s_df = analyze_50etf()
    merge_s_and_vix(s_df, vix_df)
