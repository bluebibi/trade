import time
from pytz import timezone
import sys, os

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(os.path.join(PROJECT_HOME, "/"))

from web.db.database import BuySell, buy_sell_session
from common.utils import *
from common.logger import get_logger
from codes.upbit.upbit_api import Upbit
from common.global_variables import *

logger = get_logger("sell")
upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

if os.getcwd().endswith("predict"):
    os.chdir("..")


class Seller:
    @staticmethod
    def select_all_bought_coin_names():

        trades_bought_or_trailed = buy_sell_session.query(BuySell).filter(BuySell.status.in_(
            [CoinStatus.bought.value, CoinStatus.trailed.value, CoinStatus.up_trailed.value]
        )).all()

        coin_trail_info = {}
        for trade in trades_bought_or_trailed:
            coin_ticker_name = trade.coin_ticker_name
            buy_datetime = dt.datetime.strptime(trade.buy_datetime, fmt.replace("T", " "))
            if trade.trail_datetime:
                trail_datetime = dt.datetime.strptime(trade.trail_datetime, fmt.replace("T", " "))
            else:
                trail_datetime = None

            coin_trail_info[coin_ticker_name] = {
                "buy_datetime_str": trade.buy_datetime,
                "buy_datetime": buy_datetime,
                "lstm_prob": trade.lstm_prob,
                "gb_prob": trade.gb_prob,
                "xgboost_prob": trade.xgboost_prob,
                "buy_base_price": trade.buy_base_price,
                "buy_krw": trade.buy_krw,
                "buy_fee": trade.buy_fee,
                "buy_price": trade.buy_price,
                "buy_coin_volume": trade.buy_coin_volume,
                "trail_datetime_str": trade.trail_datetime,
                "trail_datetime": trail_datetime,
                "trail_price": trade.trail_price,
                "trail_rate": trade.trail_rate,
                "total_krw": trade.total_krw,
                "trail_up_count": trade.trail_up_count,
                "status": trade.status
            }
        return coin_trail_info

    @staticmethod
    def update_coin_info(trail_datetime, trail_price, sell_fee, sell_krw, trail_rate, total_krw, trail_up_count, status,
                         coin_ticker_name, buy_datetime):

        q = buy_sell_session.query(BuySell).filter_by(coin_ticker_name=coin_ticker_name, buy_datetime=buy_datetime)
        trade = q.first()
        trade.trail_datetime = trail_datetime
        trade.trail_price = trail_price
        trade.sell_fee = sell_fee
        trade.sell_krw = sell_krw
        trade.trail_rate = trail_rate
        trade.total_krw = total_krw
        trade.trail_up_count = trail_up_count
        trade.status = status
        buy_sell_session.commit()

    def trail(self, coin_trail_info):
        now = dt.datetime.now(timezone('Asia/Seoul'))
        now_str = now.strftime(fmt)
        current_time_str = now_str.replace("T", " ")
        now_datetime = dt.datetime.strptime(now_str, fmt)

        msg_str = ""

        for coin_ticker_name in coin_trail_info:

            _, new_trail_price, sell_fee, sell_krw = upbit.get_expected_sell_coin_price_for_volume(
                coin_ticker_name,
                coin_trail_info[coin_ticker_name]["buy_coin_volume"],
                TRANSACTION_FEE_RATE
            )

            buy_krw = coin_trail_info[coin_ticker_name]["buy_krw"]
            trail_rate = (sell_krw - buy_krw) / buy_krw

            buy_datetime = coin_trail_info[coin_ticker_name]["buy_datetime"]
            time_diff = now_datetime - buy_datetime
            time_diff_minutes = time_diff.seconds / 60

            if trail_rate > SELL_RATE:
                trail_up_count, coin_status = self.sell_action_for_up_trail(
                    coin_ticker_name,
                    coin_trail_info[coin_ticker_name],
                    new_trail_price,
                    trail_rate
                )
            elif trail_rate < -DOWN_FORCE_SELL_RATE:
                trail_up_count = 0
                coin_status = CoinStatus.loss_sold.value
            else:
                trail_up_count = 0
                if time_diff_minutes > FUTURE_TARGET_SIZE * 10:
                    if trail_rate > TRANSACTION_FEE_RATE:
                        coin_status = CoinStatus.gain_sold.value
                    else:
                        coin_status = CoinStatus.loss_sold.value
                else:
                    coin_status = CoinStatus.trailed.value

            self.update_coin_info(
                trail_datetime=current_time_str,
                trail_price=new_trail_price,
                sell_fee=sell_fee,
                sell_krw=sell_krw,
                trail_rate=trail_rate,
                total_krw=coin_trail_info[coin_ticker_name]["total_krw"] + sell_krw,
                trail_up_count=trail_up_count,
                status=coin_status,
                coin_ticker_name=coin_ticker_name,
                buy_datetime=coin_trail_info[coin_ticker_name]["buy_datetime_str"]
            )

            if coin_status == CoinStatus.success_sold.value or coin_status == CoinStatus.gain_sold.value:
                msg_str += "[{0}, new_trail_price: {1}, sell_krw: {2}, trail_rate: {3}%, status: {4}]\n".format(
                    coin_ticker_name,
                    new_trail_price,
                    sell_krw,
                    convert_unit_2(trail_rate * 100),
                    coin_status_to_hangul(coin_status)
                )
        return msg_str

    def sell_action_for_up_trail(self, coin_ticker_name, one_coin_trail_info, new_trail_price, trail_rate):
        coin_status = one_coin_trail_info["status"]
        trail_price = one_coin_trail_info["trail_price"]
        trail_up_count = one_coin_trail_info["trail_up_count"]

        # 현재 가격이 처음으로 SELL_RATE 보다 올랐을 때
        if coin_status == CoinStatus.trailed.value:
            trail_up_count = 1
            new_coin_status = CoinStatus.up_trailed.value
            logger.info("{0}: trailed --> up_trailed, trail_price: {1:.2f}, trail_rate: {2:.2f}, trail_up_count: {3}/{4}".format(
                coin_ticker_name, trail_price, trail_rate, trail_up_count, UP_TRAIL_COUNT_BOUND
            ))
        # 현재 가격이 지속적으로 SELL_RATE 보다 올라와 있을 때
        elif coin_status == CoinStatus.up_trailed.value:
            # 현재 가격이 직전 trail_price 보다 올랐을 때
            if trail_price <= new_trail_price:
                new_coin_status = CoinStatus.up_trailed.value
                logger.info("{0}: up_trailed --> up_trailed, trail_price: {1:.2f}, trail_rate: {2:.2f}, trail_up_count: {3}/{4}".format(
                    coin_ticker_name, trail_price, trail_rate, trail_up_count, UP_TRAIL_COUNT_BOUND
                ))
            # 현재 가격이 직전 trail_price 보다 내렸을 때
            else:
                trail_up_count += 1
                # 업데이트된 trail_up_count가 BOUND에 도달 --> 전액 판매
                if trail_up_count >= UP_TRAIL_COUNT_BOUND:
                    new_coin_status = CoinStatus.success_sold.value
                    logger.info("{0}: The coin is sold now with the price {1} and the trail_rate {2} since the trail_up_count {3} reaches "
                                "UP_TRAIL_COUNT_BOUND {4}".format(
                        coin_ticker_name, new_trail_price, trail_rate, trail_up_count, UP_TRAIL_COUNT_BOUND
                    ))
                # 업데이트된 trail_up_count가 BOUND보다 작음 --> 다음 기회까지 관망
                else:
                    new_coin_status = CoinStatus.up_trailed.value
                    logger.info("{0}: Price is slightly down from {1} to {2} and the trail rate is now {3}, but the trail_up_count {4} is lower than "
                                "UP_TRAIL_COUNT_BOUND {5}".format(
                        coin_ticker_name, trail_price, new_trail_price, trail_rate, trail_up_count, UP_TRAIL_COUNT_BOUND
                    ))
        else:
            raise ValueError("sell_action_for_up_trail - coin_status: {0}".format(coin_status))

        return trail_up_count, new_coin_status

    def try_to_sell(self):
        coin_trail_info = self.select_all_bought_coin_names()

        if len(coin_trail_info) > 0:
            msg_str = self.trail(coin_trail_info)

            if msg_str:
                msg_str = "### SELL\n" + msg_str + " @ " + SOURCE

                if PUSH_SLACK_MESSAGE: SLACK.send_message("me", msg_str)

                logger.info("{0}".format(msg_str))


if __name__ == "__main__":
    seller = Seller()
    while True:
        seller.try_to_sell()
        time.sleep(SELL_PERIOD)
