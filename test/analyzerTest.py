# coding:utf-8

import unittest
import os
from datetime import datetime
from easeData.functions import getTestPath, dateToStr
from easeData.analyze import (CorrelationAnalyzer, PositionDiffPlotter, SellBuyRatioPlotter, QvixPlotter,
                              NeutralContractAnalyzer, OptionUnderlyingAnalyzer, VixAnalyzer, MonteCarlo)
from easeData.collector import JQDataCollector

corrAnalyzer = CorrelationAnalyzer()
positionDiffPlotter = PositionDiffPlotter()
sellBuyRatioPlotter = SellBuyRatioPlotter()
qvixPlotter = QvixPlotter()

jqsdk = JQDataCollector()

underlyingAnalyzer = OptionUnderlyingAnalyzer(jqsdk, '510050.XSHG')


class TestMonteCarlo(unittest.TestCase):
    def setUp(self):
        self.obj = MonteCarlo(underlyingAnalyzer)

    def testGetPrice(self):
        df = self.obj.getPrice(start='2019-07-01')
        print(df)

    def testMonteCarlo(self):
        df = self.obj.monteCarloSim(100, 21)
        df.to_csv(getTestPath('montecarlo.csv'))

    def testGetExpiredDays(self):
        date = '2019-08-28'
        print(self.obj.getExpiredDays(date))

    def testGetProbability(self):
        up = 3.2
        down = 2.8
        date = '2019-08-28'
        self.obj.getPrice('2018-07-28')
        print(self.obj.getProbability(up, down, date))


class TestVixAnalyzer(unittest.TestCase):
    def setUp(self):
        self.obj = VixAnalyzer(jqsdk, '510050.XSHG')

    def testUpdateVixData(self):
        self.obj.updateVixData()

    def testGetVix(self):
        df = self.obj.getVix()
        print(df.index)
        print(df)

    def testGetHistVol(self):
        self.obj.getHisVolatility(20)
        df = self.obj.getVix()
        df.to_csv(getTestPath('vix_hist_vol_test.csv'))

    def testGetHVPercentile(self):
        self.obj.getHVPercentile(20)
        df = self.obj.getVix()
        df.to_csv(getTestPath('HV20_percentile.csv'))

    def testGetPercentile(self):
        print(self.obj.getVixPercentile())
        self.obj.vix.to_csv(getTestPath('vix_test.csv'))

    def testGetPercentileDist(self):
        # self.obj.getVixPercentileDist()
        self.obj.getAllVixPercentileDist([0, 120, 250, 500])

    def testPlotPercentile(self):
        # self.obj.getVixPercentile()
        self.obj.plotPercentile([0, 120, 250, 500])
        self.obj.plotHvPercentile([0, 250])
        self.obj.plotVolatilityDiff([20, 30])
        self.obj.plotPercentileDist([0, 120, 250, 500])
        self.obj.render('qvix', 'qvix_research.html')

    def testAnalyzerVixAndUnderlying(self):
        self.obj.analyzeVixAndUnderlying()
        # self.obj.analyzeVixAndUnderlying(start='2019-01-01')
        # self.obj.analyzeVixAndUnderlying(start='2018-06-10')
        # self.obj.analyzeVixAndUnderlying(start='2017-06-10')


class TestUnderlyingAnalyzer(unittest.TestCase):
    def setUp(self):
        self.obj = OptionUnderlyingAnalyzer(jqsdk, '510050.XSHG')

    def testUpdatePrice(self):
        self.obj.updatePrice(start='2005-02-23')

    def testGetVolatility(self):
        stats_df = []
        for n in [10, 20, 30, 60, 90]:
            self.obj.getHistVolatility(n)
            name = 'HV_{}'.format(n)
            stats_df.append(self.obj.price[name])
            # print(self.obj.price[name].describe())

        self.obj.price.to_csv(getTestPath('510050_hist_volatility.csv'))

    def testPlotVolatiltiy(self):
        self.obj.plotVolatilityBox()

    def testPlotChange(self):
        self.obj.plotChange()
        # self.obj.analyzerChange(start='2018-06-11')
        # self.obj.analyzerChange(start='2016-06-11')
        # self.obj.analyzerChange(start='2019-01-01')

    def testAnalyzeChange(self):
        self.obj.analyzeChange()


class TestCorrAnalyzer(unittest.TestCase):
    def setUp(self):
        self.obj = corrAnalyzer
        # self.obj.setNum(3)

    def testIsTradeDay(self):
        print(self.obj.isTradeDay('2018-10-01'))
        print(self.obj.isTradeDay('2018-12-09'))
        print(self.obj.isTradeDay('2018-12-07'))

    def testGetDateRange(self):
        self.obj.setFreq('daily')
        print(self.obj.getDateRange())

        self.obj.setFreq('bar')
        print(self.obj.getDateRange())

    def testGetPrice(self):
        dt1 = datetime(2018, 1, 1)
        dt2 = datetime(2018, 11, 1)
        dt3 = datetime(2018, 11, 29, 9)
        dt4 = datetime(2018, 11, 29, 15)

        self.obj.setFreq('daily')
        self.obj.getPrice(dt1, dt2)

        self.obj.setFreq('bar')
        self.obj.setExclude(self.obj.cffex)
        # self.obj.getPrice(dt3, dt4, exclude=self.obj.cffex)
        self.obj.getPrice(dt3, dt4)

    def testGetCorrelationArray(self):
        # self.obj.setFreq('daily')
        # self.obj.getCorrelationArray()

        self.obj.setFreq('bar')
        self.obj.setEndDay('2018-11-30')
        self.obj.setExclude(self.obj.cffex)
        self.obj.getCorrelationArray()

    def testPlotCorrelationArray(self):
        self.obj.setFreq('bar')
        self.obj.setNum(30)
        self.obj.setEndDay('2018-12-10')
        self.obj.setExclude(self.obj.cffex)
        df = self.obj.getCorrelationArray()
        self.obj.plotCorrelationArray(df)

        # self.obj.setFreq('daily')
        # self.obj.setNum(12)
        # self.obj.setEndDay('2018-12-10')
        # self.obj.setExclude(['AP', 'fu', 'sc'])
        # df = self.obj.getCorrelationArray()
        # self.obj.plotCorrelationArray(df)


class TestPositionDiffTrend(unittest.TestCase):
    def setUp(self):
        self.obj = positionDiffPlotter

    def testGetTradingContractInfo(self):
        df = self.obj.getTradingContract()
        df.to_csv(getTestPath('optionTradingContract.csv'), encoding='utf-8-sig')

    def testGetDisplayContract(self):
        df = self.obj.getMainContract()
        df.to_csv(getTestPath('optionMainContract.csv'), encoding='utf-8-sig')

    def testGetCompContract(self):
        df = self.obj.getCompContract()
        df.to_csv(getTestPath('optionDisplayContract.csv'), encoding='utf-8-sig')

    def testGetDailyPrice(self):
        contract = '10001562.XSHG'
        df = self.obj.getDailyPrice(contract)
        df.to_csv(getTestPath('optionPrice.csv'), encoding='utf-8-sig')

    def testGetContractName(self):
        contracts = ['10001562.XSHG', '10001542.XSHG']
        print([self.obj.getContractName(c) for c in contracts])

    def testGetGroupedCode(self):
        codes = self.obj.getGoupedCode()
        print(codes)

    def testGetMainContractDailyPrice(self):
        price = self.obj.getAllContractDailyPrice()
        for code, df in price.items():
            fp = getTestPath('{}.csv'.format(code))
            df.to_csv(fp, encoding='utf-8-sig')

    def testGetPosDiff(self):
        df = self.obj.getPositionDiff()
        df.to_csv(getTestPath('posDiff.csv'), encoding='utf-8-sig')

    def testGetPosition(self):
        df = self.obj.getPosition()
        df.to_csv(getTestPath('position.csv'), encoding='utf-8-sig')

    def testPlotPosDiff(self):
        self.obj.setQueryMonth('1901')
        self.obj.plotData(self.obj.getPositionDiff)

    def testPlotPosition(self):
        # self.obj.setQueryMonth('1902')
        # self.obj.plotData(self.obj.getPosition)

        mission = ['1901', '1902', '1903', '1906']
        for m in mission:
            self.obj.setQueryMonth(m)
            self.obj.plotData(self.obj.getPosition)
            self.obj.cleanCache()


class TestNeutralAnalyzer(unittest.TestCase):
    def setUp(self):
        self.obj = NeutralContractAnalyzer(jqsdk, '510050.XSHG')

        filename = 'option_daily_2018-02-09.csv'
        self.fp = os.path.join(self.obj.jqsdk.getPricePath('option', 'daily'), filename)

    def testGetNearbyContract(self):
        df = self.obj.getNearbyContract(self.fp)
        df.to_csv(getTestPath('nearbyContract_20190225.csv'))

    def testGetUnderlying(self):
        self.obj.getUnderlyingPrice()

    def testGetAtmContract(self):
        df = self.obj.getAtmContract(self.fp, method='match')
        df.to_csv(getTestPath('atm_contract_match.csv'))

        df = self.obj.getAtmContract(self.fp, method='simple')
        df.to_csv(getTestPath('atm_contract_simple.csv'))

    def testGetStraddleContract(self):
        level = 1
        df = self.obj.getStraddleContract(self.fp, method='match', level=level)
        df.to_csv(getTestPath('straddle{}_contract_match.csv'.format(level)))

        # df = self.obj.getStraddleContract(self.fp, method='simple', level=level)
        # df.to_csv(getTestPath('straddle{}_contract_simple.csv'.format(level)))

    def testGetNeutralGroupInfo(self):
        # self.obj.getNeutralGroupInfo('2017-01-01', '2019-06-11')
        # self.obj.getNeutralGroupInfo('2017-01-01', '2019-06-11', method='simple')

        # self.obj.getNeutralGroupInfo('2017-01-01', '2019-06-12', group='straddle', level=1)
        # self.obj.getNeutralGroupInfo('2017-01-01', '2019-06-12', group='straddle', method='simple', level=1)

        # self.obj.getNeutralGroupInfo('2017-01-01', '2019-06-12', group='straddle', level=2)
        # self.obj.getNeutralGroupInfo('2017-01-01', '2019-06-12', group='straddle', method='simple', level=2)

        self.obj.getNeutralGroupInfo('2017-01-01', '2019-06-12', group='straddle', level=3)
        self.obj.getNeutralGroupInfo('2017-01-01', '2019-06-12', group='straddle', method='simple', level=3)

    def testGetNeutralNextTradeDayBar(self):
        # self.obj.getNeutralNextTradeDayBar('2019-05-20', '2019-05-24')
        self.obj.getNeutralNextTradeDayBar('2019-05-20', '2019-05-24', group='straddle', level=2)

    def testUpdateNeutralNextTradeDayBar(self):
        # end = '2019-05-28'
        # end = '2017-02-28'
        end = None
        method = 'match'
        # method = 'simple'

        self.obj.updateNeutralNextTradeDayBar(end=end, group='atm', method=method)
        self.obj.updateNeutralNextTradeDayBar(end=end, group='straddle', method=method, level=1)
        self.obj.updateNeutralNextTradeDayBar(end=end, group='straddle', method=method, level=2)
        self.obj.updateNeutralNextTradeDayBar(end=end, group='straddle', method=method, level=3)

    def testGetOHLC(self):
        method = 'match'
        self.obj.getOHLCdaily(method=method)
        self.obj.getOHLCdaily(group='straddle', level=1, method=method)
        self.obj.getOHLCdaily(group='straddle', level=2, method=method)
        self.obj.getOHLCdaily(group='straddle', level=3, method=method)

    def testGetLast5Days(self):
        self.obj.getLast5Days()

    def testDailyBackTest(self):
        # self.obj.dailyBackTest()
        # self.obj.dailyBackTest(start='open')

        # self.obj.dailyBackTest(method='simple')
        # self.obj.dailyBackTest(method='simple', start='open')

        # self.obj.setSlippage(1)
        # self.obj.setInterval(4)
        # self.obj.dailyBackTest(group='straddle', method='match', level=1, start='pre_close')
        # self.obj.dailyBackTest(group='straddle', method='simple', level=1, start='pre_close')
        # self.obj.dailyBackTest(group='straddle', method='match', level=1, start='open')
        # self.obj.dailyBackTest(group='straddle', method='simple', level=1, start='open')

        # self.obj.setSlippage(1)
        # self.obj.setInterval(4)
        # self.obj.dailyBackTest(group='straddle', method='match', level=2, start='pre_close')
        # self.obj.dailyBackTest(group='straddle', method='simple', level=2, start='pre_close')
        # self.obj.dailyBackTest(group='straddle', method='match', level=2, start='open')
        # self.obj.dailyBackTest(group='straddle', method='simple', level=2, start='open')

        # self.obj.setSlippage(1)
        # self.obj.setInterval(4)
        # self.obj.dailyBackTest(group='straddle', method='match', level=3, start='pre_close')
        # self.obj.dailyBackTest(group='straddle', method='simple', level=3, start='pre_close')
        # self.obj.dailyBackTest(group='straddle', method='match', level=3, start='open')
        # self.obj.dailyBackTest(group='straddle', method='simple', level=3, start='open')

        self.obj.dailyBackTest(group='atm', method='match', start='pre_close', isLast5=False)
        self.obj.setSlippage(1)
        self.obj.setInterval(4)
        self.obj.dailyBackTest(group='straddle', method='match', level=1, start='pre_close', isLast5=False)
        self.obj.dailyBackTest(group='straddle', method='match', level=2, start='pre_close', isLast5=False)
        self.obj.dailyBackTest(group='straddle', method='match', level=3, start='pre_close', isLast5=False)

    def testBacktestingCompare(self):
        self.obj.backTestingCompare(method='match', start='pre_close')
        self.obj.backTestingPosition()
        # self.obj.analyzeStop()

    def testRemoveGap(self):
        df = self.obj.removeOHLCgap()
        df.to_csv(getTestPath('test_atm_ohlc_remove_gap.csv'))

        # df = self.obj.removeGapByMonth()
        # df.to_csv(getTestPath('test_atm_ohlc_remove_gap_by_month.csv'))

    def testPlotOHLC(self):
        # self.obj.plotAtmOHLC()
        # self.obj.plotStraddleOHLC()
        # self.obj.plotStraddleOHLC(level=2)
        # self.obj.plotStraddleOHLC(level=3)

        # method = 'match'
        method = 'simple'
        self.obj.plotAtmOHLC(method=method, isGap=True, isIncludePre=False)
        self.obj.plotAtmOHLC(method=method, isGap=True, isIncludePre=True)
        self.obj.plotAtmOHLC(method=method, isGap=False, isIncludePre=True, divideByMonth=False)
        self.obj.plotAtmOHLC(method=method, isGap=False, isIncludePre=False, divideByMonth=False)

        # self.obj.plotStraddleOHLC(method=method, isGap=True, isIncludePre=False, divideByMonth=False, level=1)
        # self.obj.plotStraddleOHLC(method=method, isGap=True, isIncludePre=True, divideByMonth=False, level=1)
        # self.obj.plotStraddleOHLC(method=method, isGap=False, isIncludePre=False, divideByMonth=False, level=1)
        # self.obj.plotStraddleOHLC(method=method, isGap=False, isIncludePre=True, divideByMonth=False, level=1)

        # self.obj.plotStraddleOHLC(method=method, isGap=True, isIncludePre=False, divideByMonth=False, level=2)
        # self.obj.plotStraddleOHLC(method=method, isGap=True, isIncludePre=True, divideByMonth=False, level=2)
        # self.obj.plotStraddleOHLC(method=method, isGap=False, isIncludePre=False, divideByMonth=False, level=2)
        # self.obj.plotStraddleOHLC(method=method, isGap=False, isIncludePre=True, divideByMonth=False, level=2)

        # self.obj.plotStraddleOHLC(method=method, isGap=True, isIncludePre=False, divideByMonth=False, level=3)
        # self.obj.plotStraddleOHLC(method=method, isGap=True, isIncludePre=True, divideByMonth=False, level=3)
        # self.obj.plotStraddleOHLC(method=method, isGap=False, isIncludePre=False, divideByMonth=False, level=3)
        # self.obj.plotStraddleOHLC(method=method, isGap=False, isIncludePre=True, divideByMonth=False, level=3)


class TestSellBuyRatioPlotter(unittest.TestCase):
    def setUp(self):
        self.obj = sellBuyRatioPlotter
        self.atmStart = '2019-01-07'
        self.atmEnd = '2019-01-29'

        filename = 'option_daily_2019-02-27.csv'
        self.fp = os.path.join(self.obj.jqsdk.getPricePath('option', 'daily'), filename)

    def testGetContractInfo(self):
        df = self.obj.getContractInfo()
        print(self.obj.contractInfo)
        df.to_csv(getTestPath('contractInfo.csv'), encoding='utf-8-sig')

    def testGetDataFromContractInfo(self):
        code = '10001641.XSHG'
        print(self.obj.getContractType(code))
        print(self.obj.contractTypeDict)

        print(self.obj.getTradingCode(code))
        print(self.obj.tradingCodeDict)

        print(self.obj.getLastTradeDate(code))
        print(self.obj.lastTradeDateDict)

    def testGetNearbyContract(self):
        # filename = 'option_daily_2018-12-07.csv'
        filename = 'option_daily_2018-12-25.csv'
        fp = os.path.join(self.obj.jqsdk.getPricePath('option', 'daily'), filename)
        df = self.obj.getNearbyContract(fp)
        df.to_csv(getTestPath('nearbyContract.csv'))

    def testGetAtmContract(self):
        filename = 'option_daily_2019-04-10.csv'
        fp = os.path.join(self.obj.jqsdk.getPricePath('option', 'daily'), filename)
        # df = self.obj.getAtmContract(fp)
        # df.to_csv(getTestPath('atmContract.csv'))

        # df2 = self.obj.getStraddleContract(fp)
        # df2.to_csv(getTestPath('straddle.csv'))

        df3 = self.obj.getAssignedContract(fp, 3300, 2700)
        df3.to_csv(getTestPath('assigned.csv'))

    def testGetMergedPrice(self):
        start = datetime(2019, 2, 18)
        end = datetime(2019, 2, 22)
        self.obj.getMergedPrice(start, end, self.obj.getAssignedContract, callStrikePrice=2550, putStrikePrice=2550)

    def testGetRecentDays(self):
        print(self.obj.getRecentDays(7))

    def testGetAtm(self):
        filename = 'option_daily_2019-01-09.csv'
        fp = os.path.join(self.obj.jqsdk.getPricePath('option', 'daily'), filename)
        res = self.obj.getAtmPriceCombineByDate(fp)
        print(res)

    def testGetAtmAlphaByDate(self):
        res = self.obj.getAtmAlphaByDate(self.fp)
        print(res)

    def testGetAtmAlphaByRange(self):
        df = self.obj.getAtmAlphaByRange('2019-02-20', '2019-02-20')
        # df.to_csv(getTestPath('atmAlphaByRange.csv'))
        print(df)

    def testGetAtmReturnByRange(self):
        res = self.obj.getAtmReturnByRange('2019-01-18', '2019-01-23')
        print(res)

    def testStrategyAtmLastTradeDays(self):
        df = self.obj.strategyAtmLastTradeDays()
        df.to_csv(getTestPath('strategyAtmLastTradeDays.csv'))

    def testStrategyAtmReturnNextTradedayOpen(self):
        df = self.obj.strategyAtmReturnNextTradedayOpen('2017-01-01', '2019-05-10')
        df.to_csv(getTestPath('strategyAtmReturnNextTradedayOpen.csv'))

    def testGetNeutralNextTradeDayBar(self):
        # self.obj.getNeutralNextTradeDayBar(start='2017-01-01', end='2017-02-01', group='straddle', level=1)
        # self.obj.getNeutralNextTradeDayBar(start='2017-01-01', end='2017-02-01', group='straddle', level=2)
        self.obj.getNeutralNextTradeDayBar(start='2017-01-15', end='2017-02-15', group='straddle', level=3)

    def testUpdateNeutralNextTradeDayBar(self):
        self.obj.updateNeutralNextTradeDayBar()
        self.obj.updateNeutralNextTradeDayBar(group='straddle', level=1)
        self.obj.updateNeutralNextTradeDayBar(group='straddle', level=2)
        self.obj.updateNeutralNextTradeDayBar(group='straddle', level=3)

    def testGetAtmAlpha(self):
        df = self.obj.getAtmAlpha()
        print(df)

    def testPlotAtmAlpha(self):
        self.obj.plotAtmAlpha()

    def testGetAtmByRange(self):
        start = '2019-01-01'
        end = '2019-01-09'
        reslist = self.obj.getAtmPriceCombineByRange(start, end)
        print(reslist)

    def testGetOneWeekAtm(self):
        self.obj.getOneWeekMergedPrice(self.obj.getAtmContract)

    def testPlotOneWeekAtm(self):
        close = self.obj.getOneWeekMergedPrice(self.obj.getAtmContract)
        self.obj.plotMergedPrice('atmOneWeek-test', close)

        close = self.obj.getOneWeekMergedPrice(self.obj.getStraddleContract, level=2)
        self.obj.plotMergedPrice('straddle2-test', close)

    def testPlotOneWeekAtmKline(self):
        self.obj.plotMergePriceKline(self.obj.getAtmContract, 'atmOneWeek-kline-test.html')
        self.obj.plotMergePriceKline(self.obj.getStraddleContract, 'straddle-kline-test.html', level=1)

    def testPlotAssignedMergedPrice(self):
        start = datetime(2019, 2, 18)
        end = datetime(2019, 2, 22)
        call = 2550
        put = 2550
        close = self.obj.getMergedPrice(start, end, self.obj.getAssignedContract, callStrikePrice=call,
                                        putStrikePrice=put)
        self.obj.plotMergedPrice('{}-to-{}-straddle'.format(dateToStr(start), dateToStr(end)), close)

        call = 2750
        put = 2750
        start = datetime(2019, 2, 25)
        end = datetime(2019, 3, 1)
        close = self.obj.getMergedPrice(start, end, self.obj.getAssignedContract, callStrikePrice=call,
                                        putStrikePrice=put)
        # close.to_csv(getTestPath('bug.csv'))
        self.obj.plotMergedPrice('{}-to-{}-straddle'.format(dateToStr(start), dateToStr(end)), close)

    def testMergeAtm(self):
        # self.obj.setAtmStart(self.atmStart)
        # self.obj.setAtmEnd(self.atmEnd)
        # df = self.obj.mergeAtmPriceCombine()
        # df.to_csv(getTestPath('atm.csv'))

        self.obj.plotAtmCombinePrice()

    def testPlotMinute(self):
        self.obj.plotMinute()

    def testGetEtfRatio(self):
        df = self.obj.getEtfMarketRatio()
        df.to_csv(getTestPath('etfRatio.csv'))

    def testPlotEtfRatio(self):
        self.obj.plotEtfMarketRatio()

    def testCalcRatioByDate(self):
        filename = 'option_daily_2019-01-03.csv'
        fp = os.path.join(self.obj.jqsdk.getPricePath('option', 'daily'), filename)

        # 近月
        d = self.obj.calcRatioByDate(fp)
        print(d)

        # 全月份
        self.obj.isOnlyNearby = False
        d = self.obj.calcRatioByDate(fp)
        print(d)

    def testGetRatio(self):
        df = self.obj.getRatio()
        df.to_csv(getTestPath('ratio.csv'), encoding='utf-8-sig')

    def testPlotRatio(self):
        self.obj.isOnlyNearby = True
        self.obj.plotRatio()

        self.obj.isOnlyNearby = False
        self.obj.plotRatio()

    def testGet50etfAdjustDayList(self):
        d1 = datetime(2017, 1, 1)
        d2 = datetime.today()
        d3 = datetime(2019, 1, 1)
        days = self.obj.get50etfAdjustDayList(d1, d3)
        print(days)

    def testGetMoneyFlowOf50EtfInPeriod(self):
        d1 = datetime(2019, 2, 25)
        d2 = datetime.today()
        d3 = datetime(2019, 3, 1)
        df = self.obj.getMoneyFlowOf50Etf(d1, d3)
        df.to_csv(getTestPath('50etf_money_flow.csv'))

    def testGetMoneyFlowOf50Etf(self):
        d1 = datetime(2017, 1, 1)
        d2 = datetime.today()
        # d3 = datetime(2019, 3, 1)
        df = self.obj.getMoneyFlowOf50Etf(d1, d2)
        # df.to_csv(getTestPath('50etf_money_flow.csv'))

    def testUpdateMoneyFlowOf50Etf(self):
        self.obj.updateMoneyFlowOf50Etf()

    def testPlotMoneyFlowOf50Etf(self):
        self.obj.plotMoneyFlowOf50Etf()


class TestQvixPlotter(unittest.TestCase):
    def setUp(self):
        self.obj = qvixPlotter

    def testGetCsvData(self):
        self.obj.getCsvData()

    def testAdd50etf(self):
        df = self.obj.add50etf()
        print(df)
        df.to_csv(getTestPath('50etf-qvix.csv'))

    def testAdd50etfParkinson(self):
        s = self.obj.add50etfParkinsonNumber(20)
        print(s)

    def testAddHisVol(self):
        s = self.obj.addHistVolotility(20)
        print(s)
        self.obj.dailyData.to_csv(getTestPath('50etf-hisvol.csv'))

    def testPlotKline(self):
        self.obj.plotKline()

    def testPlotMa(self):
        self.obj.plotMa()

    def testPlotBoll(self):
        self.obj.plotBollChanel()

    def testPlotAll(self):
        self.obj.plotAll()

    def testPlotVolDiff(self):
        self.obj.plotVolDiff()

    def testPlotICIHIFDiff(self):
        self.obj.plotICIHIF()


if __name__ == '__main__':
    unittest.main()
