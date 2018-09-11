# coding:utf-8

import pandas as pd
import numpy as np
import os
from const import *
from functions import (loadSetting, saveSetting,
                       getUnderlyingSymbol, getParentDirectory,
                       addExchange, rmDateDash, strToDate, dateToStr,
                       mkFileDir, mkFilename, parseFilename,
                       getCurrentMainContract, getMainContract)
from jaqs.data import DataApi
from os.path import abspath, dirname
from datetime import datetime


class DataCollector(object):
    def __init__(self):
        pass


class JaqsDataCollector(DataCollector):
    """
    A collector to get various of finance Data via Jaqs Data API.
    The Data can save to csv or database.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Define category of securities
    VIEW_INSTRUMENT_INFO = 'jz.instrumentInfo'
    VIEW_TRADING_DAY_INFO = 'jz.secTradeCal'
    VIEW_INDEX_INFO = 'lb.indexInfo'
    VIEW_CONSTITUENT_OF_INDEX = 'lb.indexCons'
    VIEW_INDUSTRY_INFO = 'lb.secIndustry'
    VIEW_SUSPEND_STOCK = 'lb.secSusp'

    # Define inst_type of jaqs
    INST_TYPE_STOCK = (1,)
    INST_TYPE_FUND = (2, 3, 4, 5)
    INST_TYPE_FUTURE_BASIC = (6, 7)
    INST_TYPE_BOND = (8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20)
    INST_TYPE_ABS = (16,)
    INST_TYPE_INDEX = (100,)
    INST_TYPE_FUTURE = (101, 102, 103)
    INST_TYPE_OPTION = (201, 202, 203)

    # Define all output filed of get basic info except the default field.
    ALL_FIELD_INSTRUMENT = ('inst_type', 'delist_date', 'status', 'currency',
                            'buylot', 'selllot', 'pricetick',
                            'underlying', 'product', 'market', 'multiplier')
    ALL_FIELD_TRADING_DAY = ('isweekday', 'isweekend', 'isholiday')

    TRADE_BEGIN_TIME = ('090100', '091600', '093100', '210100')
    TRADE_END_TIME = ('150000', '151500')

    def __init__(self, logger, dataPath):
        super(JaqsDataCollector, self).__init__()
        self.logger = logger
        self.dataPath = dataPath
        self.api = None
        self.dbAdaptor = dict()

        self.tradeCalArray = None
        self.instTypeNameMap = None
        self.futureExchangeMap = None
        self.futureCurrentMainContractMap = None
        self.futureHistoryMainContractMap = None
        self.basicDataMap = dict()
        self.symbolMap = dict()

        self.createFolder()

    def createFolder(self):
        """
        Create the essential data folder when collector is instantiated.
        ----------------------------------------------------------------------------------------------------------------
        :return: None
        """
        basicDataPath = os.path.join(self.dataPath, DIR_JAQS_BASIC_DATA)
        priceDataPath = os.path.join(self.dataPath, DIR_JAQS_PRICE_DATA)
        if not os.path.exists(basicDataPath):
            os.mkdir(basicDataPath)
        if not os.path.exists(priceDataPath):
            os.mkdir(priceDataPath)

    def connectApi(self, address=None, user=None, token=None):
        """
        Initialize jaqs dataApi.
        You can save login setting into config.json to connect jaqs API without passing parameter.
        ----------------------------------------------------------------------------------------------------------------
        :param address: string. check jaqs document.
        :param user: string. get from jaqs.
        :param token: string. get form jaqs.
        """
        if self.api is None:
            settingFile = os.path.join(dirname(abspath(__file__)), FILE_SETTING)
            setting = loadSetting(settingFile)
            if address is None:
                address = setting['jaqs_address']
            if user is None:
                user = setting['jaqs_user']
            if token is None:
                token = setting['jaqs_token']
            try:
                self.api = DataApi(address)
                self.api.login(user, token)
                self.logger.info("Jaqs API Connected.")
            except Exception as e:
                msg = "Unknown Error: {}".format(e)
                self.logger.error(msg)

    def setDbAdaptor(self, dbAdaptor):
        """
        :param dbAdaptor: instance.
        :return:
        """
        adaptorName = dbAdaptor.name
        self.dbAdaptor[adaptorName] = dbAdaptor

    def getDbAdaptor(self, adaptorName):
        """
        :param adaptorName: string.
        :return: instance.
        """
        return self.dbAdaptor.get(adaptorName)

    def getInstTypeToNameMap(self):
        """
        Make a inst_type to type name map.
        ----------------------------------------------------------------------------------------------------------------
        :return:
        dict{tuple(int): string}
            key: A tuple of jaqs inst_type pre-defined in class attribute.
            value: type name. eg.'stock', 'future'
        """
        if self.instTypeNameMap is None:
            typeMap = {key: value for key, value in JaqsDataCollector.__dict__.items() if 'INST_TYPE' in key}
            self.instTypeNameMap = {value: key.replace('INST_TYPE_', '').lower() for key, value in typeMap.items()}
        return self.instTypeNameMap

    def getBasicDataFilePath(self, inst_types):
        """
        Make basic Data file name by inst_types.
        ----------------------------------------------------------------------------------------------------------------
        :param inst_types: tuple(int). You'd better pass with pre-defined class attributes directly .
        :return: string. filename path.
        """
        instTypeNameMap = self.getInstTypeToNameMap()
        filename = '{}_basic.csv'.format(instTypeNameMap.get(inst_types))
        return os.path.join(self.dataPath, DIR_JAQS_BASIC_DATA, filename)

    def getFutureCurrentMainContract(self):
        """
        Get current main contract dict from json file. The file is from Ricequant(use rqData.py to get it).
        ----------------------------------------------------------------------------------------------------------------
        """
        if self.futureCurrentMainContractMap is None:
            filePath = os.path.join(getParentDirectory(), DIR_EXTERNAL_DATA, DIR_RQ_DATA, FILE_CURRENT_MAIN_CONTRACT)
            self.futureCurrentMainContractMap = getCurrentMainContract(filePath)
        return self.futureCurrentMainContractMap

    def setFutureCurrentMainContract(self, newMainDict, updateFile=True):
        """
        Modify current main contract dict if necessary.
        ----------------------------------------------------------------------------------------------------------------
        :param newMainDict: dict{string: string}. key: underlying symbol. value: main contract.
        :param updateFile: bool. if True. it will update the json file.
        """
        for key, newValue in newMainDict.items():
            if key in self.futureCurrentMainContractMap:
                self.futureCurrentMainContractMap[key] = newValue.decode()
        if updateFile:
            filePath = os.path.join(getParentDirectory(), DIR_EXTERNAL_DATA, DIR_RQ_DATA, FILE_CURRENT_MAIN_CONTRACT)
            saveSetting(self.futureCurrentMainContractMap, filePath)

    def getFutureExchangeMap(self):
        """
        Get exchange map from basic externalData file.
        ----------------------------------------------------------------------------------------------------------------
        :return:
        dict{string: string}
            key: underlying symbol. eg.'rb'
            value: exchange symbol. eg. 'CZE'
        """
        if self.futureExchangeMap is None:
            df = self.getBasicData(self.INST_TYPE_FUTURE)
            self.futureExchangeMap = {}
            for symbol in df.symbol:
                baseSymbol = getUnderlyingSymbol(symbol)
                if baseSymbol not in self.futureExchangeMap:
                    market = symbol.split('.')[-1]
                    self.futureExchangeMap[baseSymbol] = market
        return self.futureExchangeMap

    def addFutureExchangeToSymbol(self, symbol):
        """
        Add exchange symbol to a contract symbol and cache it to accelerate.
        Format of symbolMap:
        eg. {'rb1801': 'rb1801.SHF'}
        ----------------------------------------------------------------------------------------------------------------
        :param symbol: string. Tradable symbol.
        :return: string. Symbol added with '.MARKET_SYMBOL'.
        """
        if symbol not in self.symbolMap:
            exchangeMap = self.getFutureExchangeMap()
            if getUnderlyingSymbol(symbol) in exchangeMap:
                self.symbolMap[symbol] = addExchange(symbol, exchangeMap[getUnderlyingSymbol(symbol)])
        return self.symbolMap.get(symbol)

    def isCZC(self, symbol):
        map_ = self.getFutureExchangeMap()
        underlying = getUnderlyingSymbol(symbol)
        return map_.get(underlying) == 'CZC'

    @staticmethod
    def adjustSymbolOfCZC(symbol):
        """
        Adjust symbol of CZC exchange (from rqData, end with 4-digts) to trade-able symbol(end with 3-digts)
        :param symbol: string. eg. 'SR1909'
        :return: string. eg. 'SR909'
        """
        return '{}{}'.format(getUnderlyingSymbol(symbol), symbol[-3:])

    def getFutureHistoryMainContractMap(self):
        """
        Get a underlying symbol to history main contracts map.
        If source data file is not existed, you can get it from rqData.
        ----------------------------------------------------------------------------------------------------------------
        :return:
        dict{string: [(string, string, string), ...]}
            key: underlying symbol. eg. 'rb'
            value: list of tuple(symbol, beginDate, endDate).
        """
        if self.futureHistoryMainContractMap is None:
            path = os.path.join(getParentDirectory(), DIR_EXTERNAL_DATA, DIR_RQ_DATA, FILE_MAIN_CONTRACT_HISTORY)
            self.futureHistoryMainContractMap = getMainContract(path)
        return self.futureHistoryMainContractMap

    def getFutureContractLifespan(self, symbol):
        """
        Get the life span of a future contract.
        ----------------------------------------------------------------------------------------------------------------
        :param symbol: string. contract symbol or underlying symbol. eg. 'rb1801' or 'rb'
        :return:
        tuple(string, string). The listing date and de-listing date. format: %Y%m%d
        """
        # If symbol is underlying symbol.
        map_ = self.getFutureHistoryMainContractMap()
        if symbol in map_:
            mainContracts = map_.get(symbol)
            begin = mainContracts[0][1]
            end = mainContracts[-1][2] if mainContracts[-1][2] != '' else dateToStr(datetime.today())
            jaqsStart = datetime(2012, 1, 1)
            if strToDate(end) <= jaqsStart:
                self.logger.info("Result of lifespan is None.")
                return None
            else:
                begin = begin if strToDate(begin) > jaqsStart else dateToStr(jaqsStart)
                return rmDateDash(begin).encode(), rmDateDash(end)  # uniform return format.

        # If symbol is trade-able contract symbol.
        df = self.getBasicData(self.INST_TYPE_FUTURE)
        queryRes = df.loc[df.symbol == self.addFutureExchangeToSymbol(symbol)]
        if queryRes.empty:
            self.logger.info("Result of lifespan is None.")
            return None
        else:
            return str(queryRes.iloc[0].list_date), str(queryRes.iloc[0].delist_date)

    def getMainContractListByPeriod(self, underlyingSymbol, start=None, end=None):
        """
        Get main contract list of specified underlying symbol within some period(start and end are both contained).
        ----------------------------------------------------------------------------------------------------------------
        :param underlyingSymbol: string. eg.'rb'
        :param start: string. start date of window. format: '%Y-%m-%d' or '%Y%m%d'
        :param end: string. end date of window.
        :return: list[tuple(string, string, string)]
        """

        mainContracts = self.getFutureHistoryMainContractMap().get(underlyingSymbol)
        allBeginDt = strToDate(mainContracts[0][1])
        if mainContracts[-1][2] != '':
            allEndDt = strToDate(mainContracts[-1][2])
        else:
            allEndDt = datetime.today()

        start = allBeginDt if start is None else strToDate(self.getNextTradeDay(start))
        end = allEndDt if end is None else strToDate(self.getPreTradeDay(end))

        # If selective window and duration of main contracts list is only touch or non-intersect.
        if start == allEndDt:
            return mainContracts[-1]
        if end == allBeginDt:
            return mainContracts[0]
        if start > allEndDt or end < allBeginDt:
            return []

        # If selective window intersect with duration of main contracts list.
        if start > end:
            self.logger.error('The end date must be larger than start date.')
            return []
        startIdx = None
        endIdx = None
        for index, element in enumerate(mainContracts):
            symbol, beginDate, endDate = element
            beginDate = strToDate(beginDate)
            endDate = datetime.today() if endDate == '' else strToDate(endDate)
            if beginDate <= start and start <= endDate:
                startIdx = index
            if beginDate <= end and end <= endDate:
                endIdx = index + 1

        startIdx = 0 if startIdx is None else startIdx
        endIdx = len(mainContracts) if endIdx is None else endIdx
        return mainContracts[startIdx: endIdx]

    def getMainContractSymbolByDate(self, underlyingSymbol, date):
        """
        Get main contract symbol by specified date.
        ----------------------------------------------------------------------------------------------------------------
        :param underlyingSymbol: string. eg. 'rb'
        :param date: string. format: '%Y-%m-%d' or '%Y%m%d'
        :return: symbol: string. eg.'rb1801'
        """
        date = strToDate(date)
        mainContracts = self.getFutureHistoryMainContractMap().get(underlyingSymbol)
        if mainContracts:
            for symbol, start, end in mainContracts:
                start = strToDate(start)
                if end == '':
                    end = datetime.today()
                else:
                    end = strToDate(end)
                if start <= date and date <= end:
                    return symbol

    def queryBasicData(self, category, outputField, inputParameter):
        """
        Encapsulate jaqs.DataApi.query() method. Check the jaqs document for more details.
        ----------------------------------------------------------------------------------------------------------------
        :param category: string
        :param outputField: iterable container. list[string] or tuple(string).
        :param inputParameter: dict
        :return: DataFrame
        """
        if outputField is None:
            outputField = ''
        outputField = ','.join(outputField)
        generator = ('{}={}'.format(key, value) for key, value in inputParameter.items())
        inputParameter = '&'.join(generator)
        self.logger.debug(category, outputField, inputParameter)
        df, msg = self.api.query(view=category, fields=outputField, filter=inputParameter, data_format='pandas')
        return df

    def queryInstrumentInfo(self, outputFiled=None, **kwargs):
        """
        Encapsulate self.getBasicData() method. Use to get instrument info of securities.
        ----------------------------------------------------------------------------------------------------------------
        :param outputFiled: list[string] or tuple(string).
        :param kwargs: The parameters supported by jaqs's Api.
        :return: DataFrame
        """
        self.logger.debug(outputFiled, kwargs)
        return self.queryBasicData(self.VIEW_INSTRUMENT_INFO, outputFiled, kwargs)

    def queryInstrumentInfoByType(self, inst_types, outputFiled=None, refresh=False, outputPath=None, **kwargs):
        """
        Simply download basic Data just by input market type. eg.stock, future, fund and so on.
        ----------------------------------------------------------------------------------------------------------------
        :param inst_types: tuple. Pass this parameter with class attribute that is pre-defined.
        :param outputFiled: iterable container. list[string] or tuple(string).
        :param refresh: bool. Whether to refresh the Data file which is existed.
        :param outputPath: string. The output file path.
        :param kwargs:
        :return: DataFrame
        """
        df_list = [self.queryInstrumentInfo(outputFiled=outputFiled, inst_type=i, **kwargs) for i in inst_types]
        df = pd.concat(df_list)
        if outputPath is None:
            outputPath = self.getBasicDataFilePath(inst_types)
        if not os.path.exists(outputPath) or refresh:
            df.to_csv(outputPath, encoding='utf-8-sig', index=False)
        return df

    def querySecTradeCal(self, outputFiled=None, **kwargs):
        """
        Get trade calendar from jaqs.
        ----------------------------------------------------------------------------------------------------------------
        :param outputFiled: iterable container.
        :param kwargs:
        :return: DataFrame
        """
        return self.queryBasicData(self.VIEW_TRADING_DAY_INFO, outputFiled, kwargs)

    def queryIndexInfo(self, outputFiled=None, **kwargs):
        """
        Get Index info from jaqs.
        ----------------------------------------------------------------------------------------------------------------
        :param outputFiled: iterable container.
        :param kwargs:
        :return: DataFrame
        """
        return self.queryBasicData(self.VIEW_INDEX_INFO, outputFiled, kwargs)

    def queryBar(self, symbol, **kwargs):
        """
        Encapsulate jaqs.DataApi.bar() method. Check the jaqs document for more details.
        ----------------------------------------------------------------------------------------------------------------
        :param symbol: string. For the symbol is tradable. eg.'rb1810'. So it need to be converted to 'rb1810.SHF'.
        :param kwargs: The parameters supported by jaqs's Api.
        :return: DataFrame
        """
        df, msg = self.api.bar(symbol=self.addFutureExchangeToSymbol(symbol), **kwargs)
        tday = kwargs.get('trade_date')
        if df is None:
            self.logger.warn('{}@{} Result is None. Please check parameter.'.format(symbol, tday))
        elif df.empty:
            self.logger.warn('{}@{} DataFrame is None. Please check parameter.'.format(symbol, tday))
        else:
            beginDate = df.iloc[0].date
            beginTime = df.iloc[0].time
            endDate = df.iloc[-1].date
            endTime = df.iloc[-1].time
            self.logger.info('%s Finished. Duration: %s %s - %s %s' % (symbol, beginDate, beginTime, endDate, endTime))
            return df, (beginDate, beginTime, endDate, endTime)

    def queryTick(self):
        pass

    def queryDaily(self):
        pass

    def subscribe(self):
        pass

    def getBasicData(self, inst_types):
        """
        Load basic Data from local file as DataFrame by pass inst_types and save Data to self.basicData(if not existed).
        If Data is already in self.basicData, Get it directly.
        If Data file dose not exist, it will download automatically.
        ----------------------------------------------------------------------------------------------------------------
        :param inst_types: tuple(int)
        :return:
        dict{string: DataFrame}
            key: inst_type name. eg. 'future'
            value: DataFrame.
        """
        typeName = self.getInstTypeToNameMap()[inst_types]
        if typeName not in self.basicDataMap:
            path = self.getBasicDataFilePath(inst_types)
            if os.path.exists(path):
                df = pd.read_csv(path)
            else:
                df = self.queryInstrumentInfoByType(inst_types, outputFiled=self.ALL_FIELD_INSTRUMENT)
            self.basicDataMap[typeName] = df
        return self.basicDataMap[typeName]

    def getTradeCal(self):
        """
        Load trading day calendar from jaqs or file(if existed) and save as self.tradeCalArray for caching it.
        ----------------------------------------------------------------------------------------------------------------
        :return: np.array(np.int64). Trading day np-array.
                 eg.[19900101, ..., 20180801]
        """
        if self.tradeCalArray is None:
            path = os.path.join(self.dataPath, DIR_JAQS_BASIC_DATA, FILE_TRADE_CAL)
            if os.path.exists(path):
                df = pd.read_csv(path)
            else:
                df = self.querySecTradeCal()
                df.to_csv(path, encoding='utf-8')
            self.tradeCalArray = df.trade_date.values.astype(np.int64)
        return self.tradeCalArray

    def getTradingDayArray(self, start, end=None):
        """
        Get a trading days range between start date and end date. The start and end day is contained.
        ----------------------------------------------------------------------------------------------------------------
        :param start: string. start date. format:'%-%m-%d' or '%Y%m%d'
        :param end: string. end date.
        :return: np.array(string). format:'%Y%m%d'
        """
        tradeCal = self.getTradeCal()
        if end is None:
            end = datetime.today().strftime('%Y%m%d')
        if '-' in start:
            start = rmDateDash(start)
        if '-' in end:
            end = rmDateDash(end)
        tradeDayRange = tradeCal[(tradeCal >= int(start)) & (tradeCal <= int(end))]
        return tradeDayRange.astype(np.string_)

    def getNextTradeDay(self, date, nDays=1):
        """
        Get next trading day of specify date.
        ----------------------------------------------------------------------------------------------------------------
        :param date: string. format %Y-%m-%d or %Y%m%d
        :param nDays: how many next trade-days from the date
        :return: string. format: %Y%m%d
        """
        if '-' in date:
            date = rmDateDash(date)
        tradeCal = self.getTradeCal()
        nextDays = tradeCal[tradeCal > int(date)]
        if len(nextDays) == 0:
            return date
        try:
            next_ = str(nextDays[nDays - 1])
        except IndexError:
            next_ = str(nextDays[-1])
        return next_

    def getPreTradeDay(self, date, nDays=1):
        """
        Get previous trading day of specify date.
        ----------------------------------------------------------------------------------------------------------------
        :param date: string. format %Y-%m-%d or %Y%m%d
        :param nDays: int. how many pre trade-days from the date.
        :return:
        """
        if '-' in date:
            date = rmDateDash(date)
        tradeCal = self.getTradeCal()
        preDays = tradeCal[tradeCal < int(date)]
        if len(preDays) == 0:
            return date
        try:
            pre = str(preDays[-nDays])
        except IndexError:
            pre = str(preDays[0])
        return pre

    def downloadBarByContract(self, symbol, start=None, end=None, refresh=False, saveToDb=False, needFull=True,
                              adaptorName=None, **kwargs):
        """
        Download bar to csv file by contract symbol or underlying symbol()
        If symbol is underlying symbol, it download the continuous main contract. Default date from 2012-01-01 to now.
        If symbol is trade-able contract symbol. Default start is list-date and default end is de-list date.
        ----------------------------------------------------------------------------------------------------------------
        :param symbol: string. contract symbol. eg. 'rb1805' or 'rb'
        :param start: string. start date. format:'%Y-%m-%d' or '%Y%m%d'.
        :param end: string. end date.
        :param refresh: bool. If true, it will rewrite the existed file.
        :param saveToDb: bool. If true. it will save to db.
        :param needFull: bool. If true. only full data will be save to csv file.
        :param adaptorName: string.
        """
        folder = mkFileDir(self.dataPath, symbol)

        # Get trade date of existed file and remove incomplete data file.
        existedDay = []
        files = os.listdir(folder)
        if files:
            for filename in os.listdir(folder):
                _symbol, beginDate, beginTime, endDate, endTime = parseFilename(filename)
                if not self.isCompleteFile(filename):
                    os.remove(os.path.join(folder, filename))
                else:
                    existedDay.append(endDate)

        # Set initial download mission by trading day list.s
        if start is None or end is None:
            # Check symbol is continuous main or invalid.
            lifespan = self.getFutureContractLifespan(symbol)
            if lifespan is None:
                self.logger.error("Invalid symbol: {}".format(symbol))
                return
                # if symbol in self.getFutureExchangeMap().keys():
                #     # 有的合约开始日期远远大于2012年，这里如果设为2012年，就会浪费很多时间去试错。需要更改。
                #     start = '2012-01-01'
                #     end = dateToStr(datetime.today())
                # else:
                #     self.logger.error("Invalid symbol: {}".format(symbol))
                #     return
            # Valid and trade-able symbol
            else:
                if start is None:
                    start = lifespan[0]
                if end is None:
                    end = lifespan[1]
                    if strToDate(end) > datetime.today():
                        end = dateToStr(datetime.today())
        self.logger.debug((symbol, start, end))
        tradingDay = self.getTradingDayArray(start, end)

        # Get mission depends on whether refresh or not
        if refresh:
            missionDay = tradingDay
        else:
            missionDay = [day for day in tradingDay if day not in existedDay]
        self.logger.debug("Mission: \n{}".format(missionDay))

        for trade_date in missionDay:
            filename = df = None

            # Attempt to get data
            result = self.queryBar(symbol, trade_date=int(trade_date), **kwargs)
            if result is not None:
                df, (beginDate, beginTime, endDate, endTime) = result
                filename = mkFilename(symbol, beginDate, beginTime, endDate, endTime, 'csv')
            else:
                # For maybe missing data within continuous main contract. So try to fix with main contract.
                if symbol in self.getFutureExchangeMap().keys():
                    replaceSymbol = self.getMainContractSymbolByDate(symbol, trade_date)
                    if replaceSymbol is None:
                        self.logger.info("Get Alternative data failed.")
                        continue
                    self.logger.info("Trying to get from {}".format(replaceSymbol))
                    result = self.queryBar(replaceSymbol, trade_date=int(trade_date), **kwargs)
                    if result is not None:
                        df, (beginDate, beginTime, endDate, endTime) = result
                        filename = mkFilename(symbol, beginDate, beginTime, endDate, endTime, 'csv')
                        df.code = symbol
                        df.symbol = self.addFutureExchangeToSymbol(symbol)
                        df.oi = np.nan
                        df.settle = np.nan
                    else:
                        self.logger.info("Alternative method failed.")

            # Save to file/db
            if filename and df is not None:
                df = df[df.volume != 0]
                if not needFull or self.isCompleteFile(filename):
                    path = os.path.join(folder, filename)
                    df.to_csv(path, encoding='utf-8-sig')
                if saveToDb:
                    if adaptorName is None:
                        adaptorName = 'vnpy'
                    self.saveBarToDb(adaptorName, df)

    def downloadMainContractBar(self, underlyingSymbol, start=None, end=None, **kwargs):
        """
        Download minute bar within specify period by uderlying symbol.
        ----------------------------------------------------------------------------------------------------------------
        :param underlyingSymbol: string. eg.'rb'
        :param start: start date. format: '%Y-%m-%d' or '%Y%m%d'.
        :param end: end date.
        :param kwargs:
        :return:
        """
        contracts = [item[0] for item in self.getMainContractListByPeriod(underlyingSymbol, start, end)]
        self.logger.debug(contracts)
        for contract in contracts:
            if self.isCZC(contract):
                contract = self.adjustSymbolOfCZC(contract)
            self.downloadBarByContract(contract, **kwargs)

    def downloadAllMainContractBar(self, start=None, end=None, skipSymbol=None, **kwargs):
        """
        Download all main contract minute bars within specify period .
        ----------------------------------------------------------------------------------------------------------------
        :param start: start date. format: '%Y-%m-%d' or '%Y%m%d'.
        :param end: end date
        :param skipSymbol: iterable container<string> . list or tuple.
        :param kwargs:
        :return:
        """
        if skipSymbol is None:
            skipSymbol = []
        symbols = self.getFutureExchangeMap().keys()
        for symbol in symbols:
            print(symbol)
            if symbol in self.getFutureCurrentMainContract().keys() and symbol not in skipSymbol:
                self.downloadMainContractBar(symbol, start, end, **kwargs)

    def downloadAllContinuousMainContract(self, start=None, end=None, skipSymbol=None, **kwargs):
        """
        Download all continuous main contract.
        ----------------------------------------------------------------------------------------------------------------
        :param start: start date. format: '%Y-%m-%d' or '%Y%m%d'.
        :param end: end date
        :param skipSymbol: iterable container<string> . list or tuple.
        :param kwargs:
        :return:
        """
        if skipSymbol is None:
            skipSymbol = []
        symbols = self.getFutureExchangeMap().keys()
        for symbol in symbols:
            if symbol in self.getFutureCurrentMainContract().keys() and symbol not in skipSymbol:
                self.downloadBarByContract(symbol, start, end, **kwargs)

    def downloadCurrentMainContractBar(self, start=None, skipSymbol=None, refresh=False):
        """
        Download current trade day minute bar of current main contract.
        ----------------------------------------------------------------------------------------------------------------
        :param start: start date.
        :param skipSymbol: iterable container<string> . list or tuple.
        :param refresh: bool. If true, it will rewrite the existed csv file and database.
        :return:
        """
        currentMain = self.getFutureCurrentMainContract()
        symbols = [value for key, value in currentMain.items() if key not in skipSymbol]
        self.logger.debug(symbols)
        if start is None:
            start = self.getPreTradeDay(dateToStr(datetime.today()), nDays=3)
        for symbol in symbols:
            self.downloadBarByContract(symbol, start=start, refresh=refresh, saveToDb=True)

    def saveBarToDb(self, adaptorName, df):
        """
        Save bars to db via dbAdaptor.
        ----------------------------------------------------------------------------------------------------------------
        :param adaptorName: string.
        :param df: DataFrame. Bars
        :return:
        """
        adaptor = self.getDbAdaptor(adaptorName)
        if adaptor is None:
            self.logger.error("dbAdaptor must be set first")
        else:
            for row in df.iterrows():
                index, barSeries = row
                doc = adaptor.convertBar(barSeries)
                adaptor.saveBarToDb(doc)

    def saveBarToCsv(self):
        pass

    def isCompleteFile(self, filename):
        """
        Judge whether the data of a file is complete or not by filename.
        ----------------------------------------------------------------------------------------------------------------
        :param filename: string.
        :return: bool
        """
        _symbol, beginDate, beginTime, endDate, endTime = parseFilename(filename)
        complete = beginTime in self.TRADE_BEGIN_TIME and endTime in self.TRADE_END_TIME
        return complete

    def cleanIncompleteFile(self):
        """
        Clean up the incomplete data file.
        ----------------------------------------------------------------------------------------------------------------
        :return:
        """
        path = os.path.join(self.dataPath, DIR_JAQS_PRICE_DATA, FUTURE)
        for root, dirs, files in os.walk(path):
            if files:
                files = (file_ for file_ in files if file_.endswith('.csv'))
                for f in files:
                    self.logger.debug(f)
                    if not self.isCompleteFile(f):
                        fp = os.path.join(root, f)
                        self.logger.info("Delete file: {}".format(fp))
                        os.remove(fp)
