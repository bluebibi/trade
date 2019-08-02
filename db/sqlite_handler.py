import sqlite3

import sys, os
idx = os.getcwd().index("upbit_auto_trade")
PROJECT_HOME = os.getcwd()[:idx] + "upbit_auto_trade/"
sys.path.append(PROJECT_HOME)

from common.global_variables import *

class SqliteHandler:
    def __init__(self, sqlite3_price_info_db_filename):
        with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
            conn.execute("PRAGMA busy_timeout = 3000")

    def create_price_info_tables(self, coin_names):
        with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
            cursor = conn.cursor()

            for coin_name in coin_names:
                ticker = "KRW_" + coin_name
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS {0} (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    datetime TEXT, open_price FLOAT, high_price FLOAT, low_price FLOAT, close_price FLOAT, volume FLOAT,
                    total_ask_size FLOAT, total_bid_size FLOAT, btmi FLOAT, btmi_rate FLOAT, 
                    btai FLOAT, btai_rate FLOAT)""".format(ticker))

            conn.commit()

    def drop_price_info_tables(self, coin_names):
        with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
            cursor = conn.cursor()
            for coin_name in coin_names:
                ticker = "KRW_" + coin_name
                cursor.execute("DROP TABLE IF EXISTS {0}".format(ticker))

            conn.commit()

    def create_buy_sell_table(self):
        with sqlite3.connect(sqlite3_buy_sell_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS BUY_SELL (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                coin_ticker_name TEXT, buy_datetime DATETIME, cnn_prob FLOAT, lstm_prob FLOAT, buy_base_price FLOAT,
                buy_krw INT, buy_fee INT, buy_price FLOAT, buy_coin_volume FLOAT, 
                trail_datetime DATETIME, trail_price FLOAT, sell_fee INT, sell_krw INT, trail_rate FLOAT, 
                total_krw INT, status TINYINT
                )""")

            conn.commit()

    def create_order_book_table(self, coin_names):
        with sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
            cursor = conn.cursor()

            for coin_name in coin_names:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS KRW_{0}_ORDER_BOOK (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    base_datetime DATETIME, daily_base_timestamp INTEGER, collect_timestamp INTEGER,
                    ask_price_0 FLOAT, ask_size_0 FLOAT,
                    ask_price_1 FLOAT, ask_size_1 FLOAT,
                    ask_price_2 FLOAT, ask_size_2 FLOAT,
                    ask_price_3 FLOAT, ask_size_3 FLOAT,
                    ask_price_4 FLOAT, ask_size_4 FLOAT,
                    ask_price_5 FLOAT, ask_size_5 FLOAT,
                    ask_price_6 FLOAT, ask_size_6 FLOAT,
                    ask_price_7 FLOAT, ask_size_7 FLOAT,
                    ask_price_8 FLOAT, ask_size_8 FLOAT,
                    ask_price_9 FLOAT, ask_size_9 FLOAT,
                    ask_price_10 FLOAT, ask_size_10 FLOAT,
                    ask_price_11 FLOAT, ask_size_11 FLOAT,
                    ask_price_12 FLOAT, ask_size_12 FLOAT,
                    ask_price_13 FLOAT, ask_size_13 FLOAT,
                    ask_price_14 FLOAT, ask_size_14 FLOAT,
                    bid_price_0 FLOAT, bid_size_0 FLOAT,
                    bid_price_1 FLOAT, bid_size_1 FLOAT,
                    bid_price_2 FLOAT, bid_size_2 FLOAT,
                    bid_price_3 FLOAT, bid_size_3 FLOAT,
                    bid_price_4 FLOAT, bid_size_4 FLOAT,
                    bid_price_5 FLOAT, bid_size_5 FLOAT,
                    bid_price_6 FLOAT, bid_size_6 FLOAT,
                    bid_price_7 FLOAT, bid_size_7 FLOAT,
                    bid_price_8 FLOAT, bid_size_8 FLOAT,
                    bid_price_9 FLOAT, bid_size_9 FLOAT,
                    bid_price_10 FLOAT, bid_size_10 FLOAT,
                    bid_price_11 FLOAT, bid_size_11 FLOAT,
                    bid_price_12 FLOAT, bid_size_12 FLOAT,
                    bid_price_13 FLOAT, bid_size_13 FLOAT,
                    bid_price_14 FLOAT, bid_size_14 FLOAT,
                    total_ask_size FLOAT, total_bid_size FLOAT)
                    """.format(coin_name)
                )

            conn.commit()

    def drop_order_book_tables(self, coin_names):
        with sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
            cursor = conn.cursor()
            for coin_name in coin_names:
                table_name = "KRW_" + coin_name + "_ORDER_BOOK"
                cursor.execute("DROP TABLE IF EXISTS {0}".format(table_name))

            conn.commit()

if __name__ == "__main__":
    sql_handler = SqliteHandler(sqlite3_price_info_db_filename)
    #sql_handler.create_tables(UPBIT.get_all_coin_names())
    #sql_handler.create_buy_sell_table()
    sql_handler.create_order_book_table(UPBIT.get_all_coin_names())
    #sql_handler.drop_order_book_tables(UPBIT.get_all_coin_names())
