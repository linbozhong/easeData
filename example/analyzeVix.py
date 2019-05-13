# coding:utf-8


import os
import pandas as pd
from datetime import datetime
from easeData.functions import getDataDir
from easeData.const import *


def analyze_qvix():
    filename = 'vixBar.csv'
    fp = os.path.join(getDataDir(), RESEARCH, OPTION, 'qvix', filename)
    df = pd.read_csv(fp)

    df.datetime = df.datetime.map(lambda timestamp: datetime.fromtimestamp(timestamp / 1000))
    df['pre_close'] = df['close'].shift(1)
    df['open_change'] = (df['open'] - df['pre_close']) / df['pre_close']
    df['open_change'].hist(bins=100)
