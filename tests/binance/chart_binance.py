import sys, os

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)
from common.global_variables import *

from common import asciichart
from rl_common import *

import ccxt  # noqa: E402

# -----------------------------------------------------------------------------

binance = ccxt.binance({
    'apiKey': API_KEY_BINANCE,
    'secret': SECRET_KEY_BINANCE,
    'enableRateLimit': True,
})
timeframe = '5m'

symbol = 'ETH/BTC'

# each ohlcv candle is a list of [ timestamp, open, high, low, close, volume ]
index = 4  # use close price from each ohlcv candle

height = 15
length = 80


def print_chart(exchange, symbol, timeframe):
    print("\n" + exchange.name + ' ' + symbol + ' ' + timeframe + ' chart:')

    # get a list of ohlcv candles
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)

    # get the ohlCv (closing price, index == 4)
    series = [x[index] for x in ohlcv]

    # print the chart
    print("\n" + asciichart.plot(series[-length:], {'height': height}))  # print the chart

    last = ohlcv[len(ohlcv) - 1][index]  # last closing price
    return last

last = print_chart(binance, symbol, timeframe)
print("\n" + binance.name + " ₿ = $" + str(last) + "\n")  # print last closing price