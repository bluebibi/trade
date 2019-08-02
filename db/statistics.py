import glob
import locale
import smtplib
import sqlite3
import jinja2

from email.mime.text import MIMEText

from pytz import timezone

import sys, os
idx = os.getcwd().index("upbit_auto_trade")
PROJECT_HOME = os.getcwd()[:idx] + "upbit_auto_trade/"
sys.path.append(PROJECT_HOME)

from common.global_variables import *
from common.utils import *

import datetime as dt

if os.getcwd().endswith("db"):
    os.chdir("..")

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

select_all_buy_sell_sql = "SELECT * FROM BUY_SELL ORDER BY id DESC;"

select_one_record_KRW_BTC_sql = "SELECT datetime FROM KRW_BTC ORDER BY id DESC LIMIT 1;"

count_rows_KRW_BTC_sql = "SELECT * FROM KRW_BTC ORDER BY id DESC LIMIT 1;"

now = dt.datetime.now(timezone('Asia/Seoul'))
now_str = now.strftime(fmt.replace("T", " "))


def render_template(**kwargs):
    templateLoader = jinja2.FileSystemLoader(searchpath="db")
    templateEnv = jinja2.Environment(loader=templateLoader)
    templ = templateEnv.get_template("email.html")
    return templ.render(**kwargs)


def get_model_status():
    coin_names = UPBIT.get_all_coin_names()

    cnn_model_files = glob.glob(PROJECT_HOME + 'models/CNN/*.pt')
    cnn_models = {}
    for cnn_file in cnn_model_files:
        cnn_file_name = cnn_file.split("/")[-1].split("_")
        coin_name = cnn_file_name[0]

        time_diff = elapsed_time_str(
            dt.datetime.fromtimestamp(os.stat(cnn_file).st_mtime).strftime(fmt.replace("T", " ")),
            now_str
        )

        cnn_models[coin_name] = {
            "saved_epoch": int(cnn_file_name[1]),
            "valid_loss_min": float(cnn_file_name[2]),
            "valid_accuracy": float(cnn_file_name[3]),
            "valid_data_size": int(cnn_file_name[4]),
            "valid_one_data_rate": float(cnn_file_name[5].replace(".pt", "")),
            "last_modified": time_diff
        }

    lstm_model_files = glob.glob(PROJECT_HOME + 'models/LSTM/*.pt')
    lstm_models = {}
    for lstm_file in lstm_model_files:
        lstm_file_name = lstm_file.split("/")[-1].split("_")
        coin_name = lstm_file_name[0]

        time_diff = elapsed_time_str(
            dt.datetime.fromtimestamp(os.stat(lstm_file).st_mtime).strftime(fmt.replace("T", " ")),
            now_str
        )

        lstm_models[coin_name] = {
            "saved_epoch": int(lstm_file_name[1]),
            "valid_loss_min": float(lstm_file_name[2]),
            "valid_accuracy": float(lstm_file_name[3]),
            "valid_data_size": int(lstm_file_name[4]),
            "valid_one_data_rate": float(lstm_file_name[5].replace(".pt", "")),
            "last_modified": time_diff
        }

    txt = "<tr><th>코인 이름</th><th>CNN 모델 정보</th><th>모델 구성</th><th>LSTM 모델 정보</th><th>모델 구성</th></tr>"
    num_both_models = 0
    for coin_name in coin_names:
        txt += "<tr>"

        if coin_name in cnn_models:
            cnn_info = "{0} : {1} : {2} : {3} : {4}".format(
                cnn_models[coin_name]["saved_epoch"],
                cnn_models[coin_name]["valid_loss_min"],
                cnn_models[coin_name]["valid_accuracy"],
                cnn_models[coin_name]["valid_data_size"],
                cnn_models[coin_name]["valid_one_data_rate"]
            )
            cnn_model_last_modified = cnn_models[coin_name]["last_modified"]
        else:
            cnn_info = "-"
            cnn_model_last_modified = "-"

        if coin_name in lstm_models:
            lstm_info = "{0} : {1} : {2} : {3} : {4}".format(
                lstm_models[coin_name]["saved_epoch"],
                lstm_models[coin_name]["valid_loss_min"],
                lstm_models[coin_name]["valid_accuracy"],
                lstm_models[coin_name]["valid_data_size"],
                lstm_models[coin_name]["valid_one_data_rate"]
            )
            lstm_model_last_modified = lstm_models[coin_name]["last_modified"]
        else:
            lstm_info = "-"
            lstm_model_last_modified = "-"

        if coin_name in cnn_models and coin_name in lstm_models:
            coin_name = "<span style='color:#FF0000'><strong>{0}</strong></span>".format(coin_name)
            num_both_models += 1

        txt += "<td>{0}</td><td>{1}</td><td>{2}</td><td>{3}</td><td>{4}</td>".format(
            coin_name,
            cnn_info,
            cnn_model_last_modified,
            lstm_info,
            lstm_model_last_modified
        )
    return txt, num_both_models


def get_KRW_BTC_info():
    with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
        cursor = conn.cursor()

        rows = cursor.execute(select_one_record_KRW_BTC_sql)
        for row in rows:
            last_krw_btc_datetime = row[0]
        conn.commit()

        rows = cursor.execute(count_rows_KRW_BTC_sql)
        for row in rows:
            num_krw_btc_records = row[0]
        conn.commit()

    return last_krw_btc_datetime, locale.format_string("%d", num_krw_btc_records, grouping=True)


def buy_sell_tables():
    with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
        cursor = conn.cursor()

        rows = cursor.execute(select_all_buy_sell_sql)

        conn.commit()

    txt = "<tr><th>매수 기준 날짜/시각</th><th>구매 코인</th><th>모델 확신도<br/>(CNN | LSTM)</th><th>구매 기준 가격</th><th>구매 가격</th>"
    txt += "<th>투자 금액</th><th>현재 가격</th><th>현재 원화</th><th>경과 시간</th><th>등락 비율</th><th>상태</th></tr>"
    total_rate = 0.0
    num = 0
    num_success = 0
    num_trail_bought = 0
    num_gain = 0
    num_loss = 0

    for row in rows:
        coin_status = coin_status_to_hangul(row[16])

        num += 1
        if row[16] == CoinStatus.success_sold.value:
            num_success += 1
            coin_status = "<span style='color:#FF0000'><strong>{0}</strong></span>".format(coin_status)
        elif row[16] == CoinStatus.gain_sold.value:
            num_gain += 1
            coin_status = "<span style='color:#FF8868'><strong>{0}</strong></span>".format(coin_status)
        elif row[16] == CoinStatus.loss_sold.value:
            num_loss += 1
            coin_status = "<span style='color:#92B3B7'>{0}</span>".format(coin_status)
        elif row[16] == CoinStatus.trailed.value:
            num_trail_bought += 1
        elif row[16] == CoinStatus.bought.value:
            num_trail_bought += 1

        total_rate += float(row[14])
        txt += "<tr>"
        txt += "<td>{0}</td><td>{1}</td><td>{2} | {3}</td><td>{4}</td><td>{5}</td><td>{6}</td><td>{7}</td><td>{" \
               "8}</td><td>{9}</td><td>{10}%</td><td>{11}%</td>".format(
            row[2].replace(":00", ""), #buy_datetime - 매수 기준 날짜/시각
            "<a href='https://upbit.com/exchange?code=CRIX.UPBIT.{0}'>{0}</a>".format(row[1]), #coin_ticker_name - 구매
            # 코인
            convert_unit_2(row[3]), #cnn_prob
            convert_unit_2(row[4]), #lstm_prob
            locale.format_string("%.2f", row[5], grouping=True), #buy_base_price - 구매 기준 가격
            locale.format_string("%.2f", row[8], grouping=True), #buy_price - 구매 가격
            locale.format_string("%.2f", row[6], grouping=True), #buy_krw - 투자 금액
            locale.format_string("%.2f", row[11], grouping=True), #trail_price - 현재 금액
            locale.format_string("%.2f", row[13], grouping=True), #sell_krw - 현재 원화
            elapsed_time_str(row[2], row[10]), #경과 시간
            convert_unit_2(row[14] * 100), #trail_rate - 등락 비율
            coin_status # 상태
        )
        txt += "</tr>"

    return txt, convert_unit_2(total_rate * 100), num, num_trail_bought, num_success, num_gain, num_loss


def main():
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login('yh21.han@gmail.com', GOOGLE_APP_PASSWORD)

    buy_sell_text, total_rate, num, num_trail_bought, num_success, num_gain, num_loss = buy_sell_tables()

    last_krw_btc_datetime, num_krw_btc_records = get_KRW_BTC_info()

    model_status, num_both_models = get_model_status()

    html_data = render_template(
        buy_sell_text=buy_sell_text,
        total_rate=total_rate,
        num=num,
        num_trail_bought=num_trail_bought,
        num_success=num_success,
        num_gain=num_gain,
        num_loss=num_loss,
        last_krw_btc_datetime=last_krw_btc_datetime,
        num_krw_btc_records=num_krw_btc_records,
        model_status=model_status,
        num_both_models=num_both_models
    )

    msg = MIMEText(html_data, _subtype="html", _charset="utf-8")
    msg['Subject'] = 'Statistics from ' + SOURCE

    s.sendmail("yh21.han@gmail.com", "yh21.han@gmail.com", msg.as_string())

    s.quit()


if __name__ =="__main__":
    main()
