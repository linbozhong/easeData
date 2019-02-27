# coding:utf-8

import os
import codecs
import json
import pandas as pd
from datetime import datetime

test_file_name = 'zxjt_zmv.csv'
test_file = os.path.join(os.getcwd(), 'zxjt_zmv.csv')
config_file = os.path.join(os.getcwd(), 'config.json')

TA_DIR_SOURCE = 'source'
TA_DIR_OUTPUT = 'output'
TA_DIR_OUTPUT_DETAIL = 'detail'

TA_OPTION_SIZE = 10000
TA_CSV_CODING = 'GB2312'
TA_ACCOUNT_NAME = u'账户名称'

TA_DIRECTION = u'买卖'
TA_DIRECTION_LONG = u'买入'
TA_DIRECTION_LONG_SIM = u'买'
TA_DIRECTION_SHORT = u'卖出'
TA_DIRECTION_SHORT_SIM = u'卖'
TA_DIRECTION_OTHER = u'其他'
TA_DIRECTION_SHORT_OPEN_PROFIT = u'浮盈卖出'
TA_DIRECTION_LONG_OPEN_PROFIT = u'浮盈买入'

TA_OFFSET = u'开平'
TA_OFFSET_OPEN = u'开仓'
TA_OFFSET_CLOSE = u'平仓'
TA_OFFSET_OPEN_PROFIT = u'浮盈计算'

TA_COVERED = u'备兑'

TA_EXCHANGE_TODAY = u'市场名称'
TA_EXCHANGE_HIST = u'交易所'
TA_CONTRACT_CODE_TODAY = u'合约代码'
TA_CONTRACT_CODE_HIST = u'合约编码'
TA_CONTRACT_NAME = u'合约名称'
TA_TRADING_ID_TODAY = u'成交号'
TA_TRADING_ID_HIST = u'成交编码'
TA_TRADING_TIME = u'成交时间'
TA_TRADING_DATE = u'成交日期'
TA_TRADING_DATETIME = u'时间戳'
TA_TRADING_VOLUME = u'成交数量'
TA_TRADING_PRICE = u'成交价格'
TA_TRADING_TURNOVER = u'成交金额'
TA_TRADING_COMMISSION = u'成交手续费'
TA_PER_COMMISSION = u'单笔手续费'
TA_SETTLEMENT = u'清算金额'

TA_TRADING_FIRST_TIME = u'首笔交易时间'
TA_TRADING_LAST_TIME = u'末笔交易时间'
TA_TRADING_RETURN = u'交易盈亏'
TA_TRADING_OPEN_PROFIT = u''

direction_map = {
    TA_DIRECTION_LONG_SIM: TA_DIRECTION_LONG,
    TA_DIRECTION_SHORT_SIM: TA_DIRECTION_SHORT
}

change_name_map = {
    TA_CONTRACT_CODE_TODAY: TA_CONTRACT_CODE_HIST,
    TA_EXCHANGE_TODAY: TA_EXCHANGE_HIST,
    TA_TRADING_ID_TODAY: TA_TRADING_ID_HIST
}

source = os.path.join(os.getcwd(), TA_DIR_SOURCE)
output = os.path.join(os.getcwd(), TA_DIR_OUTPUT)
detail = os.path.join(os.getcwd(), TA_DIR_OUTPUT, TA_DIR_OUTPUT_DETAIL)


def mk_dirs():
    """
    创建文件夹。
    :return:
    """
    for fp in [source, output, detail]:
        if not os.path.exists(fp):
            os.makedirs(fp)


def load_setting(filename, encoding='utf-8', **kwargs):
    """
    读取配置文件
    :param filename:
    :param encoding:
    :param kwargs:
    :return:
    """
    with codecs.open(filename, 'r', encoding) as f:
        setting = json.load(f, **kwargs)
    return setting


def analyze_trade(filename):
    """
    分析单个账户成交记录。
    :param filename:
    :return:
    """
    setting = load_account_setting(filename.split('.')[0])
    df = read_csv(filename)
    add_commission(df, setting)
    calculate_settlement(df)
    # df.to_csv('test2.csv', encoding=TA_CSV_CODING)
    begin = df[TA_TRADING_DATETIME].iloc[0].strftime('%Y-%m-%d')
    end = df[TA_TRADING_DATETIME].iloc[-1].strftime('%Y-%m-%d')

    grouped = df.groupby(by=TA_CONTRACT_CODE_HIST)
    grouped_dict = dict(list(grouped))
    # print(grouped_dict.keys())

    res_list = []
    for contract_code in setting['contract_close'].keys():
        res_dict = calculate_contract_return(grouped_dict, contract_code, setting)
        res_list.append(res_dict)

    column_order = [TA_ACCOUNT_NAME, TA_TRADING_FIRST_TIME, TA_TRADING_LAST_TIME, TA_CONTRACT_CODE_HIST,
                    TA_CONTRACT_NAME, TA_TRADING_COMMISSION, TA_TRADING_RETURN]
    res_df = pd.DataFrame(res_list)
    res_df = res_df[column_order]
    fp = os.path.join(output, u'{}_盈亏汇总_{}-{}.csv'.format(setting['name'], begin, end))
    res_df.to_csv(fp, encoding=TA_CSV_CODING)


def read_csv(filename):
    """
    读取历史成交记录，如果存在当天成交记录则进行合并操作。
    :param filename: string
    :return:
    """
    output_columns = [TA_TRADING_DATETIME, TA_EXCHANGE_HIST, TA_CONTRACT_CODE_HIST, TA_CONTRACT_NAME, TA_DIRECTION,
                      TA_OFFSET, TA_TRADING_PRICE, TA_TRADING_VOLUME, TA_TRADING_TURNOVER, TA_COVERED,
                      TA_TRADING_ID_HIST]
    base_dir = os.getcwd()
    fp = os.path.join(base_dir, filename)
    df = pd.read_csv(fp, encoding=TA_CSV_CODING)

    # 读取当日成交文件
    today_trade_file = os.path.join(base_dir, '{}_today.csv'.format(filename.split('.')[0]))
    if os.path.exists(today_trade_file):
        df_today = pd.read_csv(today_trade_file, encoding=TA_CSV_CODING)
        df_today.rename(columns=change_name_map, inplace=True)  # 更改列名
        df_today[TA_TRADING_DATE] = datetime.today().strftime('%Y-%m-%d')  # 添加日期文本
        df_today[TA_DIRECTION] = df_today[TA_DIRECTION].map(lambda x: direction_map[x])  # 买卖文本转换
        df = pd.concat([df, df_today], sort=True)

    # df.dropna(axis=1, inplace=True, how='all')
    df[TA_CONTRACT_CODE_HIST] = df[TA_CONTRACT_CODE_HIST].map(str)
    df[TA_TRADING_DATETIME] = df[TA_TRADING_DATE] + df[TA_TRADING_TIME]
    df[TA_TRADING_DATETIME] = df[TA_TRADING_DATETIME].map(
        lambda dt_str: datetime.strptime(dt_str, '%Y-%m-%d%H:%M:%S'))
    df.sort_values(by=TA_TRADING_DATETIME, inplace=True)
    df = df[df[TA_TRADING_ID_HIST].notnull()]
    df = df[output_columns]
    df.reset_index(drop=True, inplace=True)
    # print(df)
    df.to_csv('test.csv', encoding='gb2312')
    return df


def load_account_setting(account_name):
    """
    载入账户的配置。
    :param account_name:
    :return:
    """
    setting = load_setting(config_file)
    contracts_list = setting[account_name].get('contracts')
    if not contracts_list:
        contracts_list = setting['general_contracts']
    contracts_dict = {contract: setting['contract_close'][contract] for contract in contracts_list}
    setting[account_name]['contract_close'] = contracts_dict
    return setting[account_name]


def add_commission(df, setting):
    """
    根据买卖方向添加单笔手续费。
    :param df: pandas.DataFrame
    :return: df
    """
    zero_comission = ((df[TA_DIRECTION] == TA_DIRECTION_SHORT) & (df[TA_OFFSET] == TA_OFFSET_OPEN)) | (
            df[TA_DIRECTION] == TA_DIRECTION_OTHER)
    df.loc[zero_comission, TA_PER_COMMISSION] = 0
    df[TA_PER_COMMISSION].fillna(setting['commission'], inplace=True)
    df[TA_TRADING_COMMISSION] = df[TA_PER_COMMISSION] * df[TA_TRADING_VOLUME]
    # print(df)
    return df


def calculate_settlement(df):
    """
    计算每笔成交记录的清算金额。
    :param df:
    :return:
    """
    long_ = df[TA_DIRECTION] == TA_DIRECTION_LONG
    short = df[TA_DIRECTION] == TA_DIRECTION_SHORT
    df.loc[short, TA_TRADING_VOLUME] = -(df[TA_TRADING_VOLUME])
    df.loc[long_, TA_SETTLEMENT] = -(df[TA_TRADING_TURNOVER] + df[TA_TRADING_COMMISSION])
    df.loc[short, TA_SETTLEMENT] = df[TA_TRADING_TURNOVER] - df[TA_TRADING_COMMISSION]
    # print(df)
    return df


def calculate_contract_return(grouped_dict, contract_code, setting):
    """
    统计单个合约的盈亏情况。
    :param setting:
    :param grouped_dict:
    :param contract_code:
    :return:
    """
    res_dict = {}
    df = grouped_dict[contract_code]
    open_volume = df[TA_TRADING_VOLUME].sum()
    if open_volume != 0:
        close_price = setting['contract_close'][contract_code]
        df = df.append(df.iloc[-1])
        if open_volume > 0:
            df[TA_DIRECTION].values[-1] = TA_DIRECTION_SHORT_OPEN_PROFIT
        elif open_volume < 0:
            df[TA_DIRECTION].values[-1] = TA_DIRECTION_LONG_OPEN_PROFIT
        df[TA_TRADING_VOLUME].values[-1] = -open_volume
        df[TA_TRADING_PRICE].values[-1] = close_price
        df[TA_TRADING_TURNOVER].values[-1] = abs(close_price * open_volume * TA_OPTION_SIZE)
        df[TA_TRADING_COMMISSION].values[-1] = 0
        df[TA_SETTLEMENT].values[-1] = close_price * open_volume * TA_OPTION_SIZE
        df[TA_TRADING_ID_HIST].values[-1] = 0
        df[TA_PER_COMMISSION].values[-1] = 0
        df[TA_TRADING_DATETIME].values[-1] = datetime.today().replace(hour=15, minute=0, second=0)
        df[TA_OFFSET].values[-1] = TA_OFFSET_OPEN_PROFIT

    begin = df[TA_TRADING_DATETIME].iloc[0].strftime('%Y-%m-%d')
    end = df[TA_TRADING_DATETIME].iloc[-1].strftime('%Y-%m-%d')
    fp = os.path.join(detail, u'{}_合约明细_{}_{}-{}.csv'.format(setting['name'], contract_code, begin, end))
    df.to_csv(fp, encoding=TA_CSV_CODING)

    res_dict[TA_ACCOUNT_NAME] = setting['name']
    res_dict[TA_TRADING_FIRST_TIME] = df[TA_TRADING_DATETIME].values[0]
    res_dict[TA_TRADING_LAST_TIME] = df[TA_TRADING_DATETIME].values[-1]
    res_dict[TA_CONTRACT_CODE_HIST] = df[TA_CONTRACT_CODE_HIST].values[0]
    res_dict[TA_CONTRACT_NAME] = df[TA_CONTRACT_NAME].values[0]
    res_dict[TA_TRADING_COMMISSION] = df[TA_TRADING_COMMISSION].sum()
    res_dict[TA_TRADING_RETURN] = df[TA_SETTLEMENT].sum()
    return res_dict


def read_all_csv():
    """
    分析source文件夹下的所有交易记录。
    :return:
    """
    mk_dirs()
    for csv_filename in os.listdir(source):
        if csv_filename.endswith('.csv') and not ('today' in csv_filename):
            print(csv_filename)
            analyze_trade(csv_filename)


def main():
    pass


if __name__ == '__main__':
    print(os.getcwd())
    s = load_setting(config_file)
    print(s)

    print(load_account_setting('zx_gwyh'))
    print(load_account_setting('dbzq_cs'))

    read_all_csv()
    # read_csv(test_file_name)
    # d = {}
    # analyze_trade(test_file_name)
