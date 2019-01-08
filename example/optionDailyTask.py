# coding:utf-8


def updateData():
    print(u"更新数据..")
    today = datetime.today()
    start = today - timedelta(days=7)
    collector = JQDataCollector()
    collector.downloadOptionData('basic')
    collector.downloadAllOptionDaily(start)


def plotPosition():
    print(u"绘制持仓量走势..")
    mission = ['1901', '1902', '1903', '1906']
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


def main():
    updateData()
    plotPosition()
    plotRatio()
    plotQvix()


if __name__ == '__main__':
    import os
    import sys

    p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(p)

    from datetime import datetime, timedelta
    from easeData.collector import JQDataCollector
    from easeData.analyze import PositionDiffPlotter, SellBuyRatioPlotter, QvixPlotter

    main()
