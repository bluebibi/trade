import ccxt
import pprint
#print(ccxt.exchanges)

upbit = ccxt.upbit()
upbit_markets = upbit.load_markets()
pprint.pprint(upbit_markets)

pprint.pprint([(key, value) for key, value in upbit_markets.items() if '/KRW' in key])
