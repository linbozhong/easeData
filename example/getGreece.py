# coding:utf-8

import pandas as pd

from easeData.analyze import NeutralContractAnalyzer
from easeData.functions import getTestPath
from easeData.collector import JQDataCollector
from jqdatasdk import opt, query
from copy import copy
from greece_input import target

underlyingSymbol = '510050.XSHG'
exchange = 'XSHG'
output_list = ['date', 'name', 'delta', 'theta', 'gamma', 'vega', 'rho']

# target = {
#     '2019-04-19': ['10001777', '10001805', '10001795', '10001813', '10001756',
#                    '10001786', '10001821', '10001768', '10001758', '10001769'],
#
#     '2019-04-23': ['10001777', '10001805', '10001756', '10001758', '10001769',
#                    '10001768', '10001767', '10001786', '10001795', '10001813',
#                    '10001821'],
#
#     '2019-04-04': ['10001769', '10001771', '10001751', '10001752', '10001759',
#                    '10001761', '10001795', '10001762', '10001803'],
#
#     '2019-04-15': ['10001751', '10001752', '10001777', '10001784', '10001785',
#                    '10001769', '10001805', '10001764', '10001765', '10001766',
#                    '10001756', '10001795'],
#
#     '2019-03-25': ['10001585', '10001586', '10001771', '10001760', '10001620',
#                    '10001713', '10001727', '10001772', '10001596'],
#
#     '2019-03-28': ['10001771', '10001769', '10001772', '10001759', '10001760'],
#
#     '2019-03-01': ['10001582', '10001583', '10001592', '10001593', '10001729',
#                    '10001730', '10001594', '10001585'],
#
#     '2019-03-29': ['10001771', '10001772', '10001759', '10001760', '10001769'],
#
#     '2019-05-08': ['10001786', '10001787', '10001793', '10001785', '10001791',
#                    '10001792', '10001781', '10001777'],
# }

jq = JQDataCollector()
neutralAnalyzer = NeutralContractAnalyzer(underlyingSymbol)
basic_df = neutralAnalyzer.getContractInfo()


def get_greece(date, contract_list):
    risk = opt.OPT_RISK_INDICATOR
    df = jq.run_query(query(risk).filter(risk.date == date, risk.exchange_code == exchange))
    contract_list = map(lambda x: x + '.XSHG', contract_list)

    df = df[df.code.isin(contract_list)]
    df = copy(df)
    df.set_index('code', inplace=True)

    name = basic_df[basic_df.index.isin(contract_list)].name
    df = pd.concat([df, name], axis=1, sort=True)
    df = df[output_list]
    return df


def batch_get_greece():
    for date, contracts in target.items():
        fn = 'greece_{}.csv'.format(date)
        df = get_greece(date, contracts)
        df.to_csv(getTestPath(fn), encoding='gb2312')


if __name__ == '__main__':
    batch_get_greece()





