import locale
import sys, os
from flask import Flask, render_template, redirect

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade/"
sys.path.append(PROJECT_HOME)

from common.global_variables import CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt, CoinStatus, WEB_DEBUG
from common.utils import convert_unit_2, coin_status_to_hangul, elapsed_time_str
from web.db.database import User, BuySell, buy_sell_session, trade_db_session
from web.login_manager import login_manager
# import logging
from web.view.subpage import subpage_blueprint
from flask import jsonify

from codes.upbit.upbit_api import Upbit
import warnings
warnings.filterwarnings('ignore')

application = Flask(__name__)

# logging.basicConfig(
#     filename='logging.log',
#     level=logging.INFO
# )

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


def create_application():
    application.debug = WEB_DEBUG
    application.config['DEBUG'] = WEB_DEBUG
    #application.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'
    application.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    application.config['WTF_CSRF_SECRET_KEY'] = os.urandom(24)
    application.config['SECRET_KEY'] = os.urandom(24)

    application.register_blueprint(subpage_blueprint, url_prefix='/subpage')
    login_manager.init_app(application)

    upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

    # create a user
    with application.app_context():
        email = "yh21.han@gmail.com"
        q = trade_db_session.query(User).filter(User.email == email)
        user = q.first()
        if user is None:
            add_user(
                name="한연희", email=email, password="1234"
            )

    return application


def add_user(name, email, password):
    admin = User()
    admin.name = name
    admin.email = email
    admin.set_password(password)
    admin.is_admin = True
    trade_db_session.add(admin)
    trade_db_session.commit()


@application.route('/')
def hello_html():
    return render_template('index.html', menu="trade")


@application.route('/trade_data_summary', methods=["POST"])
def _trade_data_summary():
    trade_data = _trade_data(return_type="dict")
    del trade_data['trades']

    return jsonify(trade_data)


@application.route('/trade_data', methods=["POST"])
def _trade_data(return_type="json"):
    num = 0
    num_success = 0
    num_trail_bought = 0
    num_gain = 0
    num_loss = 0
    total_gain = 0.0

    q = buy_sell_session.query(BuySell).order_by(BuySell.id.desc())
    trades = q.all()

    for trade in trades:
        total_gain += trade.sell_krw - trade.buy_krw

        trade.coin_ticker_name = "<a href='https://upbit.com/exchange?code=CRIX.UPBIT.{0}' target='_blank'>{0}</a>".format(
            trade.coin_ticker_name
        )
        trade.elapsed_time = elapsed_time_str(trade.buy_datetime, trade.trail_datetime)
        trade.gb_prob = convert_unit_2(trade.gb_prob)
        trade.xgboost_prob = convert_unit_2(trade.xgboost_prob)
        trade.buy_base_price = locale.format_string("%.2f", trade.buy_base_price, grouping=True)
        trade.buy_price = locale.format_string("%.2f", trade.buy_price, grouping=True)
        if trade.status == CoinStatus.trailed or trade.status == CoinStatus.up_trailed:
            trade.trail_price = "<a href='/subpage/trail?trade_id={0}'>{1}</a>".format(
                trade.id, locale.format_string("%.2f", trade.trail_price, grouping=True)
            )
        else:
            trade.trail_price = "<a href='/subpage/trail_completed?trade_id={0}'>{1}</a>".format(
                trade.id, locale.format_string("%.2f", trade.trail_price, grouping=True)
            )
        trade.buy_krw = locale.format_string("%.0f", trade.buy_krw, grouping=True)
        trade.sell_krw = locale.format_string("%.2f", trade.sell_krw, grouping=True)
        trade.trail_rate = locale.format_string("%.2f", trade.trail_rate * 100, grouping=True)
        trade.trail_datetime = None
        trade.query = None
        trade.query_class = None
        trade.metadata = None

        num += 1
        coin_status = coin_status_to_hangul(trade.status)
        if trade.status == CoinStatus.success_sold.value:
            num_success += 1
            trade.coin_status = "<span style='color:#FF0000'><strong>{0}</strong></span>".format(coin_status)
        elif trade.status == CoinStatus.gain_sold.value:
            num_gain += 1
            trade.coin_status = "<span style='color:#FF8868'><strong>{0}</strong></span>".format(coin_status)
        elif trade.status == CoinStatus.loss_sold.value:
            num_loss += 1
            trade.coin_status = "<span style='color:#92B3B7'>{0}</span>".format(coin_status)
        elif trade.status == CoinStatus.trailed.value:
            num_trail_bought += 1
            trade.coin_status = "<span style='color:#000000'>{0}</span>".format(coin_status)
        elif trade.status == CoinStatus.bought.value:
            num_trail_bought += 1

    total_gain = locale.format_string("%.2f", total_gain, grouping=True)

    trade_lst = []
    for trade in trades:
        trade_lst.append(trade.to_json())

    trade_data = {
        "trades": trade_lst,
        "num": num,
        "num_trail_bought": num_trail_bought,
        "num_total_success": num_gain + num_success,
        "num_gain": num_gain,
        "num_success": num_success,
        "num_loss": num_loss,
        "total_gain": total_gain
    }

    if return_type == "dict":
        return trade_data
    else:
        return jsonify(trade_data["trades"])


@application.errorhandler(401)
def unauthorized(e):
    return redirect('/auth/login')


if __name__ == "__main__":
    #logging.info("Flask Web Server Started!!!")
    application = create_application()
    application.run(host="0.0.0.0", port="8080")
    # application.run(host="localhost", port="8080", ssl_context=ssl_context)
