import locale
from flask import Blueprint, render_template, request, jsonify
import sys, os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from codes.upbit.upbit_api import Upbit, get_markets

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(os.path.join(PROJECT_HOME, "/"))

from web.db.database import BuySell, get_order_book_class
from web.db.database import db
from common.global_variables import *
from common.utils import *

subpage_blueprint = Blueprint('subpage', __name__)
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

engine_model = create_engine('sqlite:///{0}/web/db/model.db'.format(PROJECT_HOME))
Session = sessionmaker(bind=engine_model)
session = Session()

upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)


@subpage_blueprint.route('/markets')
def _markets():
    return render_template('markets.html', menu="markets")


@subpage_blueprint.route('/market_data', methods=["POST"])
def _market_data():
    return jsonify(get_markets(quote='KRW'))


@subpage_blueprint.route('/models')
def _models():
    coin_names = upbit.get_all_coin_names()

    xgboost_model_files = glob.glob(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE,  'XGBOOST', '*.pkl'))
    xgboost_models = {}
    for xgboost_file in xgboost_model_files:
        xgboost_file_name = xgboost_file.split("/")[-1].split(".pkl")
        coin_name = xgboost_file_name[0]

        time_diff = dt.datetime.fromtimestamp(os.stat(xgboost_file).st_mtime).strftime(fmt.replace("T", " "))

        xgboost_models[coin_name] = {
            "last_modified": time_diff
        }

    gb_model_files = glob.glob(os.path.join(PROJECT_HOME, LOCAL_MODEL_SOURCE, 'GB', '*.pkl'))
    gb_models = {}
    for gb_file in gb_model_files:
        gb_file_name = gb_file.split("/")[-1].split(".pkl")
        coin_name = gb_file_name[0]

        time_diff = dt.datetime.fromtimestamp(os.stat(gb_file).st_mtime).strftime(fmt.replace("T", " "))

        gb_models[coin_name] = {
            "last_modified": time_diff
        }

    txt = "<tr><th>코인 이름</th><th>XGBOOST 모델 정보</th><th>GB 모델 정보</th></tr>"
    num_xgboost_models = 0
    num_gb_models = 0
    for coin_name in coin_names:
        txt += "<tr>"

        if coin_name in xgboost_models:
            xgboost_model_last_modified = xgboost_models[coin_name]["last_modified"]
            num_xgboost_models += 1
        else:
            xgboost_model_last_modified = "-"

        if coin_name in gb_models:
            gb_model_last_modified = gb_models[coin_name]["last_modified"]
            num_gb_models += 1
        else:
            gb_model_last_modified = "-"

        if xgboost_model_last_modified != "-" and gb_model_last_modified != "-":
            coin_name = "<span style='color:#FF0000'><strong>{0}</strong></span>".format(coin_name)

        txt += "<td>{0}</td><td>{1}</td><td>{2}</td>".format(
            coin_name,
            xgboost_model_last_modified,
            gb_model_last_modified
        )

    return render_template("subpage/models.html", menu="models",
                           num_xgboost_models=num_xgboost_models, num_gb_models=num_gb_models)


def get_KRW_BTC_info():
    q = db.session.query(get_order_book_class("BTC")).order_by(get_order_book_class("BTC").collect_timestamp.desc()).limit(1)
    last_krw_btc_datetime = q.first().base_datetime

    num_krw_btc_records = db.session.query(get_order_book_class("BTC")).count()

    return last_krw_btc_datetime, num_krw_btc_records


@subpage_blueprint.route('/data_collects')
def news_main():

    last_krw_btc_datetime, num_krw_btc_records = get_KRW_BTC_info()
    num_krw_btc_records = locale.format_string("%.0f", num_krw_btc_records, grouping=True)

    return render_template("subpage/data_collects.html", menu="data_collects",
                           last_krw_btc_datetime=last_krw_btc_datetime, num_krw_btc_records=num_krw_btc_records)


@subpage_blueprint.route('/trail')
def _trail():
    q = db.session.query(BuySell).filter_by(id=request.args.get("trade_id"))
    trade = q.first()

    coin_name = trade.coin_ticker_name.split('-')[1]

    return render_template("subpage/trail.html", menu="trade",
                           trade_id=trade.id, coin_name=coin_name, base_datetime=trade.buy_datetime)


@subpage_blueprint.route('/trail_completed')
def _trail_completed():
    q = db.session.query(BuySell).filter_by(id=request.args.get("trade_id"))
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
        q = db.session.query(BuySell).filter_by(id=request.args.get("trade_id"))
        trade = q.first()
    else:
        q = db.session.query(BuySell).filter_by(id=trade_id)
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

    q = db.session.query(get_order_book_class(coin_name)).filter(
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
