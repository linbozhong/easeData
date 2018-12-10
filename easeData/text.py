# coding:utf-8

# 数据采集器API
API_IS_CONNECTED = u'数据采集器的API已经连接，不需要重复连接'
API_NOT_CONNECTED = u'数据采集器的API尚未连接，请先连接'
API_SUCCEED = u'数据采集器API连接成功'
API_FAILED = u'数据采集器API连接失败，请检查账号设置'

# 错误信息
ERROR_UNKNOWN = u'未知错误'
ERROR_NOT_TARGET_PATH = u'尚未设置要转换的文件目录'
ERROR_NOT_ACTIVE_CONVERTER = u'尚未激活数据转换器'
ERROR_NOT_FREQ = u'尚未设置要保存的数据的时间周期'
ERROR_INVALID_SYMBOL = u'不是合法的代码'

# 提示信息
TIME_CONSUMPTION = u'时间消耗'
DATA_IS_NONE = u'没有获取到数据'
DB_UPDATE_COMPLETE = u'数据库价格序列更新完成'

# 文件操作
FILE_IS_EXISTED = u'文件已存在'
FILE_DOWNLOAD_SUCCEED = u'文件下载完成'
FILE_UPDATE_SUCCEED = u'文件增量更新完成'
FILE_IS_NEWEST = u'文件是最新的，不需要增量更新'
FILE_LOADING = u'正在读取文件'
FILE_DATA_EXISTED_IN_DB = u'文件数据已保存在数据库内'
FILE_INVALID_NAME = u'不符合规范的文件名'

# 专有提示信息
JQ_SYNC_SUCCEED = u'数据库同步成功，当日总使用量'
JQ_THIS_COUNT = u'本次数据使用量'
JQ_TODAY_COUNT = u'当日总使用量'
JQ_IGNORE_THIS_MONTH = u'已设置为忽略当前月份的数据'

JAQS_END_LT_START = u'结束日期必须大于或等于开始日期'