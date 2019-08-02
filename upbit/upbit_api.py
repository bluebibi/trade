import time
import jwt
from urllib.parse import urlencode
import re
import pprint
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import datetime

pp = pprint.PrettyPrinter(indent=2)

getframe_expr = 'sys._getframe({}).f_code.co_name'


def _send_post_request(url, headers=None, data=None):
    try:
        resp = requests_retry_session().post(url, headers=headers, data=data)
        remaining_req_dict = {}
        remaining_req = resp.headers.get('Remaining-Req')
        if remaining_req is not None:
            group, min, sec = _parse_remaining_req(remaining_req)
            remaining_req_dict['group'] = group
            remaining_req_dict['min'] = min
            remaining_req_dict['sec'] = sec
        contents = resp.json()
        return contents, remaining_req_dict
    except Exception as x:
        print("send post request failed", x.__class__.__name__)
        print("caller: ", eval(getframe_expr.format(2)))
        return None


def _parse_remaining_req(remaining_req):
    try:
        p = re.compile("group=([a-z]+); min=([0-9]+); sec=([0-9]+)")
        m = p.search(remaining_req)
        return m.group(1), int(m.group(2)), int(m.group(3))
    except:
        return None, None, None


def _send_get_request(url, headers=None):
    try:
        resp = requests_retry_session().get(url, headers=headers)
        remaining_req_dict = {}
        remaining_req = resp.headers.get('Remaining-Req')
        if remaining_req is not None:
            group, min, sec = _parse_remaining_req(remaining_req)
            remaining_req_dict['group'] = group
            remaining_req_dict['min'] = min
            remaining_req_dict['sec'] = sec
        contents = resp.json()
        return contents, remaining_req_dict
    except Exception as x:
        print("send get request failed", x.__class__.__name__)
        print("caller: ", eval(getframe_expr.format(2)))
        return None


def _send_delete_request(url, headers=None, data=None):
    try:
        resp = requests_retry_session().delete(url, headers=headers, data=data)
        remaining_req_dict = {}
        remaining_req = resp.headers.get('Remaining-Req')
        if remaining_req is not None:
            group, min, sec = _parse_remaining_req(remaining_req)
            remaining_req_dict['group'] = group
            remaining_req_dict['min'] = min
            remaining_req_dict['sec'] = sec
        contents = resp.json()
        return contents, remaining_req_dict
    except Exception as x:
        print("send post request failed", x.__class__.__name__)
        print("caller: ", eval(getframe_expr.format(2)))
        return None


def get_tick_size(price):
    if price >= 2000000:
        order_unit = get_order_unit(2000000)
        tick_size = round(price / order_unit) * order_unit
    elif price >= 1000000:
        order_unit = get_order_unit(1000000)
        tick_size = round(price / order_unit) * order_unit
    elif price >= 500000:
        order_unit = get_order_unit(500000)
        tick_size = round(price / order_unit) * order_unit
    elif price >= 100000:
        order_unit = get_order_unit(100000)
        tick_size = round(price / order_unit) * order_unit
    elif price >= 10000:
        order_unit = get_order_unit(10000)
        tick_size = round(price / order_unit) * order_unit
    elif price >= 1000:
        order_unit = get_order_unit(1000)
        tick_size = round(price / order_unit) * order_unit
    elif price >= 100:
        order_unit = get_order_unit(100)
        tick_size = round(price / order_unit) * order_unit
    elif price >= 10:
        order_unit = get_order_unit(10)
        tick_size = round(price / order_unit) * order_unit
    else:
        order_unit = get_order_unit(1)
        tick_size = round(price / order_unit) * order_unit
    return tick_size


def get_order_unit(price):
    if price >= 2000000:
        return 1000
    elif price >= 1000000:
        return 500
    elif price >= 500000:
        return 100
    elif price >= 100000:
        return 50
    elif price >= 10000:
        return 10
    elif price >= 1000:
        return 5
    elif price >= 100:
        return 1
    elif price >= 10:
        return 0.1
    else:
        return 0.01


def requests_retry_session(retries=5, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None):
    s = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist)
    adapter = HTTPAdapter(max_retries=retry)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    return s


def _call_public_api(url, **kwargs):
    try:
        while True:
            time.sleep(0.05)
            resp = requests_retry_session().get(url, params=kwargs)
            contents = resp.json()

            if contents and 'error' in contents and contents['error']['message'] == 'Too many API requests.':
                time.sleep(0.05)
            else:
                break

        return contents
    except Exception as x:
        print("It failed", x.__class__.__name__)
        return None


class Upbit:
    def __init__(self, access, secret, fmt):
        self.access = access
        self.secret = secret
        self.fmt = fmt

    def _request_headers(self, data=None):
        payload = {
            "access_key": self.access,
            "nonce": int(time.time() * 1000)
        }
        if data is not None:
            payload['query'] = urlencode(data)
        jwt_token = jwt.encode(payload, self.secret, algorithm="HS256").decode('utf-8')
        authorization_token = 'Bearer {}'.format(jwt_token)
        headers = {"Authorization": authorization_token}
        return headers

    def get_balance(self, ticker="KRW"):
        """
        특정 코인/원화의 잔고 조회
        :param ticker:
        :return:
        """
        try:
            # KRW-BTC
            if '-' in ticker:
                ticker = ticker.split('-')[1]

            balances = self.get_balances()[0]
            balance = None

            for x in balances:
                if x['currency'] == ticker:
                    balance = float(x['balance'])
                    break
            return balance

        except Exception as x:
            print(x.__class__.__name__)
            return None


    def get_balances(self):
        '''
        전체 계좌 조회
        :return:
        '''
        url = "https://api.upbit.com/v1/accounts"
        headers = self._request_headers()
        return _send_get_request(url, headers=headers)

    def buy_limit_order(self, ticker, price, volume):
        '''
        지정가 매수
        :param ticker: 마켓 티커
        :param price: 주문 가격
        :param volume: 주문 수량
        :return:
        '''
        try:
            url = "https://api.upbit.com/v1/orders"
            data = {"market": ticker,
                    "side": "bid",
                    "volume": str(volume),
                    "price": str(price),
                    "ord_type": "limit"}
            headers = self._request_headers(data)
            return _send_post_request(url, headers=headers, data=data)
        except Exception as x:
            print(x.__class__.__name__)
            return None

    def buy_market_order(self, ticker, price):
        '''
        시장가 매수
        :param ticker: 마켓 티커
        :param price: 주문 가격
        :return:
        '''
        try:
            url = "https://api.upbit.com/v1/orders"
            data = {"market": ticker,
                    "side": "bid",
                    "price": str(price),
                    "ord_type": "price"}
            headers = self._request_headers(data)
            return _send_post_request(url, headers=headers, data=data)
        except Exception as x:
            print(x.__class__.__name__)
            return None

    def sell_market_order(self, ticker, volume):
        '''
        시장가 매도
        :param ticker: 마켓 티커
        :param volume: 주문 수량
        :return:
        '''
        try:
            url = "https://api.upbit.com/v1/orders"
            data = {"market": ticker,
                    "side": "bid",
                    "volume": str(volume),
                    "ord_type": "market"}
            headers = self._request_headers(data)
            return _send_post_request(url, headers=headers, data=data)
        except Exception as x:
            print(x.__class__.__name__)
            return None

    def buy_market_order_old(self, ticker, price, margin=0.01):
        """
        시장가 매수 (호가 조회 후 최우선 매도호가로 주문)
        :param ticker:  티커
        :param price:  매수금액
        :param margin:  매수 수량 계산에 사용되는 margin
        :return:
        """
        try:
            orderbooks = self.get_orderbook(ticker)
            orderbooks = orderbooks[0]['orderbook_units']
            total_ask_size = 0

            for orderbook in orderbooks:
                ask_price = orderbook['ask_price']
                ask_size = orderbook['ask_size']

                bid_price = ask_price                                   # 매수가
                available_bid_size = (price / ask_price) * (1 - margin)   # 매수 가능 수량 (마진 고려)
                bid_size = min(available_bid_size, ask_size)            # 현재 호가에 대한 매수 수량
                self.buy_limit_order(ticker, bid_price, bid_size)
                total_ask_size += bid_size

                # 현재 호가에 수량이 부족한 경우
                if available_bid_size > ask_size:
                    price -= (bid_price * bid_size)
                else:
                    break

            return total_ask_size
        except Exception as x:
            print(x.__class__.__name__)
            return None

    def sell_market_order_old(self, ticker, size):
        """
        시장가 매도 (호가 조회 후 최우선 매수 호가로 주문)
        :param ticker:  티커
        :param size:  수량
        :return:
        """
        try:
            orderbooks = self.get_orderbook(ticker)
            orderbooks = orderbooks[0]['orderbook_units']

            for orderbook in orderbooks:
                # 매수호가
                bid_price = orderbook['bid_price']
                bid_size = orderbook['bid_size']

                ask_price = bid_price                                   # 매도가 = 최우선 매수가
                ask_size = min(size, bid_size)                          # 현재 호가에 대한 매수 수량
                self.sell_limit_order(ticker, ask_price, ask_size)

                # 현재 호가에 수량이 부족한 경우
                if bid_size < size:
                    size -= bid_size
                else:
                    break
        except Exception as x:
            print(x.__class__.__name__)
            return None


    def sell_limit_order(self, ticker, price, volume):
        '''
        지정가 매도
        :param ticker: 마켓 티커
        :param price: 주문 가격
        :param volume: 주문 수량
        :return:
        '''
        try:
            url = "https://api.upbit.com/v1/orders"
            data = {"market": ticker,
                    "side": "ask",
                    "volume": str(volume),
                    "price": str(price),
                    "ord_type": "limit"}
            headers = self._request_headers(data)
            return _send_post_request(url, headers=headers, data=data)
        except Exception as x:
            print(x.__class__.__name__)
            return None

    def cancel_order(self, uuid):
        '''
        주문 취소
        :param uuid: 주문 함수의 리턴 값중 uuid
        :return:
        '''
        try:
            url = "https://api.upbit.com/v1/order"
            data = {"uuid": uuid}
            headers = self._request_headers(data)
            return _send_delete_request(url, headers=headers, data=data)
        except Exception as x:
            print(x.__class__.__name__)
            return None

    ##########################
    ############ 조회
    ##########################
    def get_tickers(self, fiat="ALL"):
        """
        마켓 코드 조회 (업비트에서 거래 가능한 마켓 목록 조회)
        :return:
        """
        try:
            url = "https://api.upbit.com/v1/market/all"
            contents = _call_public_api(url)

            if isinstance(contents, list):
                markets = [x['market'] for x in contents]

                if fiat is "KRW":
                    return [x for x in markets if x.startswith("KRW")]
                elif fiat is "BTC":
                    return [x for x in markets if x.startswith("BTC")]
                elif fiat is "ETH":
                    return [x for x in markets if x.startswith("ETH")]
                elif fiat is "USDT":
                    return [x for x in markets if x.startswith("USDT")]
                else:
                    return markets
            else:
                return None
        except Exception as x:
            print(x.__class__.__name__)
            return None

    def get_ohlcv(self, ticker="KRW-BTC", interval="day", count=200):
        """
        일 캔들 조회
        :return:
        """
        try:
            if interval is "day":
                url = "https://api.upbit.com/v1/candles/days"
            elif interval is "minute1":
                url = "https://api.upbit.com/v1/candles/minutes/1"
            elif interval is "minute3":
                url = "https://api.upbit.com/v1/candles/minutes/3"
            elif interval is "minute5":
                url = "https://api.upbit.com/v1/candles/minutes/5"
            elif interval is "minute10":
                url = "https://api.upbit.com/v1/candles/minutes/10"
            elif interval is "minute15":
                url = "https://api.upbit.com/v1/candles/minutes/15"
            elif interval is "minute30":
                url = "https://api.upbit.com/v1/candles/minutes/30"
            elif interval is "minute60":
                url = "https://api.upbit.com/v1/candles/minutes/60"
            elif interval is "minute240":
                url = "https://api.upbit.com/v1/candles/minutes/240"
            elif interval is "week":
                url = "https://api.upbit.com/v1/candles/weeks"
            elif interval is "month":
                url = "https://api.upbit.com/v1/candles/months"
            else:
                url = "https://api.upbit.com/v1/candles/days"

            contents = _call_public_api(url, market=ticker, count=count)
            dt_list = [datetime.datetime.strptime(x['candle_date_time_kst'], self.fmt) for x in contents]
            df = pd.DataFrame(contents, columns=['candle_date_time_kst', 'opening_price', 'high_price', 'low_price', 'trade_price',
                                                 'candle_acc_trade_volume'],
                              index=dt_list)
            df = df.rename(
                columns={'candle_date_time_kst': "datetime", "opening_price": "open", "high_price": "high", "low_price": "low", "trade_price": "close",
                         "candle_acc_trade_volume": "volume"})
            return df.iloc[::-1]
        except Exception as x:
            print(x.__class__.__name__)
            return None

    def get_daily_ohlcv_from_base(self, ticker="KRW-BTC", base=0):
        try:
            df = self.get_ohlcv(ticker, interval="minute60")
            df = df.resample('24H', base=base).agg(
                {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            )
            return df
        except Exception as x:
            print(x.__class__.__name__)
            return None

    def get_current_price(self, ticker="KRW-BTC"):
        '''
        최종 체결 가격 조회 (현재가)
        :param ticker:
        :return:
        '''
        try:
            url = "https://api.upbit.com/v1/ticker"
            contents = _call_public_api(url, markets=ticker)

            if contents is not None:
                # 여러 마케을 동시에 조회
                if isinstance(ticker, list):
                    ret = {}
                    for content in contents:
                        market = content['market']
                        price = content['trade_price']
                        ret[market] = price
                    return ret
                else:
                    return contents[0]['trade_price']
            else:
                return None
        except Exception as x:
            print(x.__class__.__name__)
            return None

    def get_orderbook(self, tickers="KRW-BTC"):
        '''
        호가 정보 조회
        :param tickers: 티커 목록을 문자열
        :return:
        '''
        try:
            url = "https://api.upbit.com/v1/orderbook"
            contents = _call_public_api(url, markets=tickers)
            return contents
        except Exception as x:
            print(x.__class__.__name__)
            return None

    def get_market_index(self):
        try:
            url = "https://crix-api-cdn.upbit.com/v1/crix/index/recents?codes=IDX.UPBIT.UBMI&codes=IDX.UPBIT.UBAI"
            contents = _call_public_api(url)

            idx_dict = {
                'data': {
                    'btai': {'market_index': None, 'rate': None, 'width': None},
                    'btmi': {'market_index': None, 'rate': None, 'width': None},
                    'date': None
                },
                'status': '0000'
            }

            for idx in contents:
                if idx['code'] == 'IDX.UPBIT.UBMI':
                    idx_dict['data']['btmi']['market_index'] = "{0:.2f}".format(idx['tradePrice'])
                    idx_dict['data']['btmi']['rate'] = "{0:.2f}".format(idx['signedChangeRate'] * 100)
                if idx['code'] == 'IDX.UPBIT.UBAI':
                    idx_dict['data']['btai']['market_index'] = "{0:.2f}".format(idx['tradePrice'])
                    idx_dict['data']['btai']['rate'] = "{0:.2f}".format(idx['signedChangeRate'] * 100)

            return idx_dict
        except Exception as x:
            print(x.__class__.__name__)
            return None

    def get_all_coin_names(self):
        url = "https://api.upbit.com/v1/market/all"

        contents = _call_public_api(url)
        coin_names = []
        for m in contents:
            if m['market'].startswith('KRW-'):
                coin_names.append(m['market'].split('-')[1])
        return coin_names

    def get_expected_buy_coin_price_for_krw(self, ticker, krw, transaction_fee_rate):
        orderbook = self.get_orderbook(tickers=ticker)[0]
        orderbook_units = orderbook["orderbook_units"]
        ask_price_lst = []
        ask_size_lst = []
        for item in orderbook_units:
            ask_price_lst.append(item["ask_price"])
            ask_size_lst.append(item["ask_size"])

        # print(ask_price_lst)
        # print(ask_size_lst)

        original_krw = krw

        fee = krw * transaction_fee_rate
        krw = krw - fee

        calc_size_sum = 0.0

        #print(0, krw, calc_size_sum, 0)
        for i, ask_size in enumerate(ask_size_lst):
            calc_krw_sum = ask_price_lst[i] * ask_size
            if calc_krw_sum > krw:
                calc_size_sum += krw / ask_price_lst[i]
                krw = krw - krw
                #print(i+1, krw, calc_size_sum)
                break
            else:
                calc_size_sum += ask_size
                krw = krw - calc_krw_sum
                #print(i+1, krw, calc_size_sum)

        calc_price = (original_krw - fee) / calc_size_sum

        # 매수원금: 1000000, 수수료: 500.0, 매수단가: 1823.7691975619496, 확보한 코인수량: 548.0408383561644
        return original_krw, fee, calc_price, calc_size_sum

    def get_expected_sell_coin_price_for_volume(self, ticker, volume, transaction_fee_rate):
        orderbook = self.get_orderbook(tickers=ticker)[0]
        orderbook_units = orderbook["orderbook_units"]
        bid_price_lst = []
        bid_size_lst = []
        for item in orderbook_units:
            bid_price_lst.append(item["bid_price"])
            bid_size_lst.append(item["bid_size"])

        # print(bid_price_lst)
        # print(bid_size_lst)

        calc_krw_sum = 0.0
        original_volume = volume

        #print(0, volume, calc_krw_sum)
        for i, bid_size in enumerate(bid_size_lst):
            if bid_size > volume:
                calc_krw_sum += bid_price_lst[i] * volume
                volume = volume - volume
                #print(i+1, volume, calc_krw_sum)
                break
            else:
                calc_krw_sum += bid_price_lst[i] * bid_size
                volume = volume - bid_size
                #print(i+1, volume, calc_krw_sum)

        calc_price = calc_krw_sum / original_volume

        fee = calc_krw_sum * transaction_fee_rate

        calc_krw_sum = calc_krw_sum - fee

        # 매도 코인수량: 548.0408383561644, 매도단가: 1805.0, 수수료: 494.79924644171336, 매도결과금:989103.693636985
        return original_volume, calc_price, fee, calc_krw_sum






