# coding:utf-8

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from functions import initDataDir
from collections import OrderedDict


def analyzeCorr(folder):
    allPrice = OrderedDict()
    filenames = os.listdir(folder)
    varieties = [filename.replace('0000.csv', '') for filename in filenames]
    paths = [os.path.join(folder, filename) for filename in filenames if filename.endswith('.csv')]
    print(varieties)
    for variety, path in zip(varieties, paths):
        df = pd.read_csv(path, index_col=0)
        allPrice[variety] = df['close']

    priceDf = pd.DataFrame(allPrice)
    returnDf = priceDf.pct_change()
    returnDf = returnDf.corr()
    returnDf.to_csv('corrtest.csv')
    return returnDf


def heat_map(data, scale=1):
    nums = len(data)
    scale = scale if scale <= 2 else 2
    sqrSize = int(nums * scale)

    fig = plt.figure(figsize=(sqrSize, sqrSize))
    ax = fig.add_subplot(1, 1, 1)

    sns.heatmap(data, ax=ax, cmap='RdBu', annot=True, fmt='.2f', vmax=1, vmin=-1, square=True, cbar=False)
    ax.xaxis.set_ticks_position('top')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize='xx-large')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize='xx-large', rotation='horizontal')

    fig.savefig('test2.png', dpi=200)


def main():
    dp = initDataDir()
    folder = os.path.join(dp, 'jaqsPriceData', 'future', 'daily', 'continuousMain')

    data = analyzeCorr(folder)
    heat_map(data)


if __name__ == '__main__':
    main()
