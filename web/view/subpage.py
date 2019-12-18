import locale
from flask import Blueprint, render_template, request, jsonify
import sys, os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(os.path.join(PROJECT_HOME, "/"))

from codes.upbit.upbit_api import Upbit
from codes.upbit.recorder.upbit_info import UpbitInfo
from web.db.database import BuySell, get_order_book_class, upbit_info_session, naver_order_book_session, \
    buy_sell_session, model_session, Model
from common.global_variables import *
from common.utils import *

subpage_blueprint = Blueprint('subpage', __name__)
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)


@subpage_blueprint.route('/markets')
def _markets():
    return render_template('markets.html', menu="markets")


@subpage_blueprint.route('/market_data', methods=["POST"])
def _market_data():
    upbit_info_results = upbit_info_session.query(UpbitInfo).all()

    market_lst = []
    for upbit_info in upbit_info_results:
        market_lst.append(upbit_info.to_dict())

    return jsonify(market_lst)


@subpage_blueprint.route('/models')
def _models():
    model_dates = {}
    xgboost_model_files = glob.glob(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE,  'XGBOOST.pkl'))

    for xgboost_file in xgboost_model_files:
        time_diff = dt.datetime.fromtimestamp(os.stat(xgboost_file).st_mtime).strftime(fmt.replace("T", " "))

        model_dates['xgboost'] = {
            "last_modified": time_diff
        }

    gb_model_files = glob.glob(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, 'GB.pkl'))
    for gb_file in gb_model_files:
        time_diff = dt.datetime.fromtimestamp(os.stat(gb_file).st_mtime).strftime(fmt.replace("T", " "))

        model_dates["gb"] = {
            "last_modified": time_diff
        }

    return render_template("subpage/models.html", menu="models", model_dates=model_dates)


@subpage_blueprint.route('/model_data', methods=["POST"])
def _model_data(return_type="json"):
    models = model_session.query(Model).all()

    model_lst = []
    for model in models:
        model_lst.append(model.to_json())

    return jsonify(model_lst)


def get_KRW_BTC_info():
    q = naver_order_book_session.query(get_order_book_class("BTC")).order_by(get_order_book_class("BTC").collect_timestamp.desc()).limit(1)
    last_krw_btc_datetime = q.first().base_datetime

    num_krw_btc_records = naver_order_book_session.query(get_order_book_class("BTC")).count()

    return last_krw_btc_datetime, num_krw_btc_records


@subpage_blueprint.route('/data_collects')
def news_main():

    last_krw_btc_datetime, num_krw_btc_records = get_KRW_BTC_info()
    num_krw_btc_records = locale.format_string("%.0f", num_krw_btc_records, grouping=True)

    return render_template("subpage/data_collects.html", menu="data_collects",
                           last_krw_btc_datetime=last_krw_btc_datetime, num_krw_btc_records=num_krw_btc_records)


@subpage_blueprint.route('/trail')
def _trail():
    q = buy_sell_session.query(BuySell).filter_by(id=request.args.get("trade_id"))
    trade = q.first()

    coin_name = trade.coin_ticker_name.split('-')[1]

    return render_template("subpage/trail.html", menu="trade",
                           trade_id=trade.id, coin_name=coin_name, base_datetime=trade.buy_datetime)


@subpage_blueprint.route('/trail_completed')
def _trail_completed():
    q = buy_sell_session.query(BuySell).filter_by(id=request.args.get("trade_id"))
    trade = q.first()

    coin_name = trade.coin_ticker_name.split('-')[1]

    order_book_windows_data = _price_info_json(
        trade_id=trade.id, coin_name=coin_name, base_datetime_str=trade.buy_datetime, return_type="dict"
    )

    return render_template(
        "subpage/trail_completed.html", menu="trade",
        trade_id=trade.id, coin_name=coin_name, base_datetime=trade.buy_datetime,
        chart_labels=order_book_windows_data[coin_name]['base_datetime'],
        chart_ask_prices=order_book_windows_data[coin_name]['ask_price_lst'],
        chart_bid_prices=order_book_windows_data[coin_name]['bid_price_lst'],
        buy_price=order_book_windows_data[coin_name]['buy_price'],
        target_price=order_book_windows_data[coin_name]['target_price'],
        trail_price=order_book_windows_data[coin_name]['trail_price'],
        buy_datetime=order_book_windows_data[coin_name]['buy_datetime'],
        gb_prob=order_book_windows_data[coin_name]['gb_prob'],
        xgboost_prob=order_book_windows_data[coin_name]['xgboost_prob'],
        trail_rate=order_book_windows_data[coin_name]['trail_rate']
    )


@subpage_blueprint.route('/price_info_json', methods=['POST'])
def _price_info_json(trade_id=None, coin_name=None, base_datetime_str=None, return_type="json"):
    if coin_name is None and return_type == 'dict':
        coin_name = request.args.get("coin_name")
        base_datetime_str = request.args.get("base_datetime")
        q = buy_sell_session.query(BuySell).filter_by(id=request.args.get("trade_id"))
        trade = q.first()
    else:
        q = buy_sell_session.query(BuySell).filter_by(id=trade_id)
        trade = q.first()

    datetime_lst = []
    if isinstance(base_datetime_str, type(str())):
        base_datetime = dt.datetime.strptime(base_datetime_str, fmt.replace("T", " "))
    else:
        base_datetime = base_datetime_str

    for cursor in range(-WINDOW_SIZE, FUTURE_TARGET_SIZE + 1):
        c_base_datetime = base_datetime + dt.timedelta(minutes=cursor * 10)
        c_base_datetime_str = dt.datetime.strftime(c_base_datetime, fmt.replace("T", " "))
        datetime_lst.append(c_base_datetime_str)

    q = naver_order_book_session.query(get_order_book_class(coin_name)).filter(
        get_order_book_class(coin_name).base_datetime.in_(datetime_lst))
    order_book_lst = q.all()

    order_book_windows_data = dict()
    order_book_windows_data[coin_name] = dict()
    order_book_windows_data[coin_name]['base_datetime'] = []
    order_book_windows_data[coin_name]['ask_price_lst'] = []
    order_book_windows_data[coin_name]['bid_price_lst'] = []
    order_book_windows_data[coin_name]['buy_price'] = trade.buy_price
    order_book_windows_data[coin_name]['target_price'] = trade.buy_price * (1.0 + SELL_RATE)
    order_book_windows_data[coin_name]['trail_price'] = trade.trail_price
    order_book_windows_data[coin_name]['buy_datetime'] = trade.buy_datetime
    order_book_windows_data[coin_name]['gb_prob'] = convert_unit_2(trade.gb_prob)
    order_book_windows_data[coin_name]['xgboost_prob'] = convert_unit_2(trade.xgboost_prob)
    order_book_windows_data[coin_name]['trail_rate'] = locale.format_string("%.2f", trade.trail_rate * 100, grouping=True)

    for order_book in order_book_lst:
        order_book_windows_data[coin_name]['base_datetime'].append(order_book.base_datetime)
        order_book_windows_data[coin_name]['ask_price_lst'].append(order_book.ask_price_0)
        order_book_windows_data[coin_name]['bid_price_lst'].append(order_book.bid_price_0)

    if return_type == "dict":
        return order_book_windows_data
    else:
        return jsonify(order_book_windows_data)
