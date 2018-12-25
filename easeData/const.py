# coding:utf-8

# 基础数据
BASIC = 'basic'

# 数据周期
DAILY = 'daily'
BAR = 'bar'
TICK = 'tick'

# 市场定义
FUTURE = 'future'
STOCK = 'stock'
FUND = 'fund'
OPTION = 'option'

# 数据供应商
VENDOR_BASE = 'BaseVendor'
VENDOR_JQ = 'JQData'
VENDOR_JAQS = 'JaqsData'
VENDOR_RQ = 'RQData'
VENDOR_TQ = 'TQData'

# 交易所代码
# 可以用作不同数据供应商转换数据的中间代码，采取vnpy的代码
EXCHANGE_SSE = 'SSE'  # 上交所
EXCHANGE_SZSE = 'SZSE'  # 深交所
EXCHANGE_CFFEX = 'CFFEX'  # 中金所
EXCHANGE_SHFE = 'SHFE'  # 上期所
EXCHANGE_CZCE = 'CZCE'  # 郑商所
EXCHANGE_DCE = 'DCE'  # 大商所
EXCHANGE_SGE = 'SGE'  # 上金所
EXCHANGE_INE = 'INE'  # 上海国际能源交易中心
EXCHANGE_HKEX = 'HKEX'  # 港交所
EXCHANGE_HKFE = 'HKFE'  # 香港期货交易所
EXCHANGE_UNKNOWN = 'UNKNOWN'  # 未知交易所
EXCHANGE_NONE = ''  # 空交易所

# 文件目录
DIR_EXTERNAL_DATA = 'externalData'  # 外部依赖数据目录
DIR_RQ_DATA = 'rqData'

DIR_BASIC_DATA = 'BasicData'
DIR_PRICE_DATA = 'PriceData'

DIR_DATA_ROOT = 'edData'
DIR_DATA_LOG = 'log'
DIR_DATA_TEST = 'test'


# DIR_JAQS_BASIC_DATA = 'jaqsBasicData'
# DIR_JAQS_PRICE_DATA = 'jaqsPriceData'
# DIR_RQ_PRICE_DATA = 'rqPriceData'
# DIR_JQDATA_BASIC_DATA = 'jqdataBasicData'
# DIR_JQDATA_PRICE_DATA = 'jqdataPriceData'


# 文件名
FILE_SETTING = 'config.json'

# File name of external data
FILE_CURRENT_MAIN_CONTRACT = 'current_main_contract.json'
FILE_MAIN_CONTRACT_HISTORY = 'main_contract_history.csv'

# File of basic Data.
FILE_FUTURE_BASIC = 'future_baisc.csv'
FILE_TRADE_CAL = 'trade_cal.csv'

# 数据库名
VNPY_TICK_DB_NAME = 'VnTrader_Tick_Db'
VNPY_DAILY_DB_NAME = 'VnTrader_Daily_Db'
VNPY_MINUTE_DB_NAME = 'VnTrader_1Min_Db'

# 转换器名称
ADAPTOR_VNPY = 'vnpyAdaptor'

CONVERTER_JQDATA = 'jqdataConverter'
