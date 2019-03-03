# coding:utf-8


def updateData():
    print(u"更新数据..")
    today = datetime.today()
    start = today - timedelta(days=15)
    collector = JQDataCollector()
    collector.downloadOptionData('basic')
    collector.downloadAllOptionDaily(start)


def plotPosition():
    print(u"绘制持仓量走势..")
    mission = ['1903', '1904', '1906', '1909']
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


def plotAtm():
    print(u"绘制平值期权走势..")
    plotter = SellBuyRatioPlotter()
    plotter.plotAtmCombinePrice()
    print(u"绘制平值期权Alpha..")
    plotter.plotAtmAlpha()

    print(u"绘制50etf资金流向图..")
    plotter.updateMoneyFlowOf50Etf()
    plotter.plotMoneyFlowOf50Etf()


def plotEtfRatio():
    print(u"绘制50etf成交量占比走势..")
    plotter = SellBuyRatioPlotter()
    plotter.plotEtfMarketRatio()


def main():
    updateData()
    plotPosition()
    plotRatio()
    plotQvix()
    plotAtm()
    # plotEtfRatio()


if __name__ == '__main__':
    import os
    import sys

    # 添加当前目录到py查找环境目录
    p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(p)

    from datetime import datetime, timedelta
    from easeData.collector import JQDataCollector
    from easeData.analyze import PositionDiffPlotter, SellBuyRatioPlotter, QvixPlotter

    main()
