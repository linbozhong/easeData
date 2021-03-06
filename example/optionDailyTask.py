# coding:utf-8

def updateData():
    print(u"更新数据..")
    today = datetime.today()
    start = today - timedelta(days=15)
    collector = JQDataCollector()
    collector.downloadOptionData('basic')
    collector.downloadAllOptionDaily(start)
    collector.downloadAllOptionGreece(start)


def plotPosition():
    print(u"绘制持仓量走势..")
    mission = ['1904', '1905', '1906', '1909']
    plotter = PositionDiffPlotter()
    for m in mission:
        plotter.setQueryMonth(m)
        plotter.plotData(plotter.getPosition)
        plotter.cleanCache()


def plotRatio():
    print(u"绘制沽购比走势..")
    plotter = SellBuyRatioPlotter()

    plotter.isOnlyNearby = True
    plotter.plotRatio()

    plotter.isOnlyNearby = False
    plotter.plotRatio()


def plotQvix():
    print(u"绘制波动率指数..")
    plotter = QvixPlotter()
    plotter.plotAll()
    print(u"绘制50etf波动率差")
    plotter.plotVolDiff()
    print(u'绘制IFIHIC价差')
    plotter.plotICIHIF()


def plotAtm():
    plotter = SellBuyRatioPlotter()

    # print(u"绘制平值期权走势..")
    # plotter.plotAtmCombinePrice()

    print(u"绘制上周五跨式组合期权本周走势图..")
    close = plotter.getOneWeekMergedPrice(plotter.getAtmContract)
    plotter.plotMergedPrice('atmOneWeek', close)

    close = plotter.getOneWeekMergedPrice(plotter.getStraddleContract, level=1)
    plotter.plotMergedPrice('straddle1OneWeek', close)

    close = plotter.getOneWeekMergedPrice(plotter.getStraddleContract, level=2)
    plotter.plotMergedPrice('straddle2OneWeek', close)

    # print(u"绘制上周五跨式组合期权本周k线图")
    # plotter.plotMergePriceKline(plotter.getAtmContract, 'atmOneWeekKline.html')
    # plotter.plotMergePriceKline(plotter.getStraddleContract, 'straddle1OneWeekKline.html', level=1)
    # plotter.plotMergePriceKline(plotter.getStraddleContract, 'straddle2OneWeekKline.html', level=2)

    # print(u"绘制平值期权Alpha..")
    # plotter.plotAtmAlpha()

    # print(u"绘制50etf资金流向图..")
    # plotter.updateMoneyFlowOf50Etf()
    # plotter.plotMoneyFlowOf50Etf()

    # print(u'update atm continuous bar')
    # plotter.updateNeutralNextTradeDayBar()


def plotEtfRatio():
    print(u"绘制50etf成交量占比走势..")
    plotter = SellBuyRatioPlotter()
    plotter.plotEtfMarketRatio()


def vixHist():
    vixAnalyzer.updateVixData()
    underlyingAnalyzer.updatePrice(start='2005-02-23')
    vixAnalyzer.plotPercentile([0, 120, 250, 500])
    vixAnalyzer.plotHvPercentile([0, 250])
    vixAnalyzer.plotVolatilityDiff([20, 30])
    vixAnalyzer.getAllVixPercentileDist([0, 120, 250, 500])
    vixAnalyzer.plotPercentileDist([0, 120, 250, 500])
    vixAnalyzer.render('qvix', 'qvix_research.html')


def main():
    updateData()
    # plotPosition()
    plotRatio()

    plotAtm()
    # plotEtfRatio()

    analyzeVix.plot_atm_ohlc_daily()

    vixHist()

    plotQvix()


if __name__ == '__main__':
    import os
    import sys

    # 添加当前目录到py查找环境目录
    p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(p)

    import analyzeVix

    from datetime import datetime, timedelta
    from easeData.collector import JQDataCollector
    from easeData.analyze import (PositionDiffPlotter, SellBuyRatioPlotter, QvixPlotter,
                                  VixAnalyzer, OptionUnderlyingAnalyzer)
    from easeData.analyze import get_qvix_data
    from easeData.functions import dateToStr

    collector = JQDataCollector()

    vixAnalyzer = VixAnalyzer(collector, '510050.XSHG')
    underlyingAnalyzer = OptionUnderlyingAnalyzer(collector, '510050.XSHG')

    main()
