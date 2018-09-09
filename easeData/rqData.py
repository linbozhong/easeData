# coding:utf-8

"""
Please note:
This file can not be run locally! Because it use APIs can only use in the ricequant/research environment.
You can use this file by following these steps:
1.Register in Ricequant and open notebook in research environment.
2.Paste code of this file and run. Then it will save externalData files on your ricequant folder.
3.Download those files and put into '/externalData' of this project

"""

import pandas as pd
import json
import codecs
from datetime import datetime
from collections import OrderedDict


def get_latest_tradingday():
    # From today to get the latest trading day
    today = datetime.today().strftime('%Y%m%d')
    latest_tradingday = get_previous_trading_date(today)
    return latest_tradingday.strftime('%Y%m%d')


def get_latset_main_contract(symbol):
    # From underlying symbol to get the latest trading main contract.
    # The parameter and return results of RQ Api are all capital. So it needs to be adjusted according to the exchange.
    latest_tradingday = get_latest_tradingday()
    series = get_dominant_future(symbol.upper(), latest_tradingday)
    result = series[0]
    if symbol.islower():
        result = result.lower()
    return result


def get_contracts(**kwargs):
    # Get All the instrument info. You can specify date use parameter date='20180727'
    df = all_instruments(type="Future", country='cn', **kwargs)
    df = df.drop_duplicates(['underlying_symbol'])
    df = df[['exchange', 'underlying_symbol']]
    return df


def get_current_main_contract():
    """
    Get the latest main contracts of all exchange.
    --------------------------------------------------------
    :return:
    Before Flattened
        dict{string, dict{string: string}}
            key: exchange symbol, eg.'DCE'
            value: a dict
                    key: underlying symbol, eg.'rb'
                    value: main contract, eg.'rb1810'
        eg.
        {
            'SHFEE': {'rb': 'rb1810', 'cu': 'cu1810', ..},
            ..
        }
    After flattened(The actual return value.
        dict{string, string}
            key: underlying symbol
            value: main contract
        eg.
        {'rb': 'rb1810', 'cu': 'cu1810', ...}
    """

    df = get_contracts(date=get_latest_tradingday())
    current_main_contract = {}
    for row in df.iterrows():
        key = row[1].exchange
        symbol = row[1].underlying_symbol
        if key not in current_main_contract:
            current_main_contract[key] = {}
        if key in ['DCE', 'INE', 'SHFE']:
            symbol = symbol.lower()

        main_contract_symbol = get_latset_main_contract(symbol)
        # CZCE's symbol is 3-digs and RQ's is 4-digs.
        if key == 'CZCE':
            main_contract_symbol = main_contract_symbol[:-4] + main_contract_symbol[-3:]
        main_contract = {symbol: main_contract_symbol}
        current_main_contract[key].update(main_contract)

    # flatten the dict
    flattenedDict = OrderedDict()
    for symbolDict in current_main_contract.values():
        flattenedDict.update(symbolDict)

    with codecs.open('current_main_contract.json', 'w', 'utf-8') as f:
        json.dump(flattenedDict, f, indent=4)
    print("current_main_contract.json is saved.")
    return flattenedDict


def get_contracts_dict():
    """
    Get a dict with exchange key and underlying symbol value.
    --------------------------------------------------------
    :return:
    dict:{string: [string]}
        key: exchange symbol
        value: list of underlying symbol
    eg.
    {'SHFE': ['rb', 'cu', ..], ..}
    """
    df = get_contracts()
    symbols = {}
    for i in range(len(df)):
        key = df.iloc[i].exchange
        value = df.iloc[i].underlying_symbol
        if key not in ['CZCE', 'CFFEX']:
            value = value.lower()
        symbols.setdefault(key, []).append(value)
    return symbols


def get_data():
    contracts_dict = get_contracts_dict()

    # DataFrame should be initialized with a symbol for the empty DataFrame can't use merge method.
    main_hist = pd.DataFrame()
    fg = get_dominant_future('FG')
    main_hist['FG'] = fg
    main_hist['date'] = main_hist.index
    main_hist['exchange'] = 'CZCE'
    main_hist = main_hist[['date', 'exchange', 'FG']]  # 调整顺序

    # Merge all the symbol externalData.
    for exchange, symbols in contracts_dict.items():
        for symbol in symbols:
            # skip the first symbol which is used to initialize the DataFrame.
            if symbol == 'FG':
                continue
            df_symbol = pd.DataFrame()
            series = get_dominant_future(symbol.upper())
            df_symbol[symbol] = series
            df_symbol['date'] = df_symbol.index
            df_symbol['exchange'] = exchange
            main_hist = pd.merge(main_hist, df_symbol, how='outer')

    main_hist.to_csv('main_contract_history.csv', index=False)
    print("main_contract_history.csv is saved.")


# ====== Running  =========
# get_contracts()
# get_contracts_dict()
get_data()
get_current_main_contract()