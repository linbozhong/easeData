# coding:utf-8

import random
import time

month = [4, 5]

option_type = [u'购', u'沽']

direction = [u'买入', u'卖出']

offset = [u'开仓', u'平仓']

open_type = [u'保证金', u'delta']

open_quantity = ['5%', '10%']

close_quantity = ['33%', '50%', '100%']

strike_price_a = range(2700, 3050, 50)
strike_price_b = range(3100, 3400, 100)
strike_price_a.extend(strike_price_b)

if __name__ == '__main__':
    print(u'模拟指令开始：')
    print('-' * 100)
    while True:
        time_interval = random.randint(1, 5)
        time.sleep(time_interval)

        order_offset = random.choice(offset)

        if order_offset == u'开仓':
            order_type = random.choice(open_type)
            order_quantity = random.choice(open_quantity)
            order_direction = random.choice(direction)
        else:
            order_type = ''
            order_direction = ''
            order_quantity = random.choice(close_quantity)

        order_m = random.choice(month)
        order_option_type = random.choice(option_type)
        order_strike_price = random.choice(strike_price_a)

        command = u'{offset} {direction} {month}月 {strike}{option_type} {order_type} {quantity}'.format(
            offset=order_offset,
            direction=order_direction,
            month=order_m,
            strike=order_strike_price,
            option_type=order_option_type,
            order_type=order_type,
            quantity=order_quantity)

        print(command)
        print('-' * 100)
