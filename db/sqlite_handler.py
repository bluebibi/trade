import sqlite3

import sys, os
idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade/"
sys.path.append(PROJECT_HOME)

from upbit.upbit_api import Upbit
from common.global_variables import *


### upbit_order_book_recorder.py
select_by_start_base_datetime = "SELECT base_datetime FROM 'KRW_{0}_ORDER_BOOK' ORDER BY collect_timestamp ASC, base_datetime ASC LIMIT 1;"
select_by_final_base_datetime = "SELECT base_datetime FROM 'KRW_{0}_ORDER_BOOK' ORDER BY collect_timestamp DESC, base_datetime DESC LIMIT 1;"

select_by_datetime = "SELECT base_datetime FROM 'KRW_{0}_ORDER_BOOK' WHERE base_datetime=? LIMIT 1;"


create_order_book_table = """
                    CREATE TABLE IF NOT EXISTS KRW_{0}_ORDER_BOOK (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        base_datetime DATETIME, 
                        daily_base_timestamp INTEGER, 
                        collect_timestamp INTEGER,
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
                        total_ask_size FLOAT, total_bid_size FLOAT
                    )
                    """

### upbit_order_book_based_data.py
order_book_insert_sql = """
    INSERT INTO 'KRW_{0}_ORDER_BOOK' VALUES(
    NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

select_order_book_by_datetime = """
    SELECT * FROM 'KRW_{0}_ORDER_BOOK' WHERE base_datetime=? LIMIT 1;
"""

order_book_for_one_coin = """
SELECT
    C.base_datetime as 'base_datetime',
    C.daily_base_timestamp as 'daily_base_timestamp',
    C.collect_timestamp as 'collect_timestamp',

    {0}
"""

sql_body = ""

one_sql_ask_body = """
    C.ask_price_{0} as 'ask_price_{0}',
    B.ask_price_{0} as 'ask_price_{0}_btc',
    C.ask_size_{0} as 'ask_size_{0}',
    B.ask_size_{0} as 'ask_size_{0}_btc',
"""

one_sql_bid_body = """
    C.bid_price_{0} as 'bid_price_{0}',
    B.bid_price_{0} as 'bid_price_{0}_btc',
    C.bid_size_{0} as 'bid_size_{0}',
    B.bid_size_{0} as 'bid_size_{0}_btc',
"""

for i in range(0, 15):
    sql_body += one_sql_ask_body.format(i)

for i in range(0, 15):
    sql_body += one_sql_bid_body.format(i)

sql_body += """
    C.total_ask_size as 'total_ask_size', 
    B.total_ask_size as 'total_ask_size_btc',
    C.total_bid_size as 'total_bid_size', 
    B.total_bid_size as 'total_bid_size_btc'    
    """

order_book_for_one_coin = order_book_for_one_coin.format(sql_body)

select_all_from_order_book_for_one_coin = order_book_for_one_coin + """
    FROM KRW_BTC_ORDER_BOOK as B INNER JOIN KRW_{0}_ORDER_BOOK as C ON B.base_datetime = C.base_datetime
    ORDER BY collect_timestamp ASC, base_datetime ASC;
"""

select_all_from_order_book_for_one_coin_limit = order_book_for_one_coin + """
    FROM KRW_BTC_ORDER_BOOK as B INNER JOIN KRW_{0}_ORDER_BOOK as C ON B.base_datetime = C.base_datetime
    ORDER BY collect_timestamp ASC, base_datetime ASC LIMIT {1};
"""

select_all_from_order_book_for_one_coin_recent_window = order_book_for_one_coin + """
    FROM KRW_BTC_ORDER_BOOK as B INNER JOIN KRW_{0}_ORDER_BOOK as C ON B.base_datetime = C.base_datetime
    ORDER BY collect_timestamp DESC, base_datetime DESC LIMIT {1};
"""

create_buy_sell_table = """
                CREATE TABLE IF NOT EXISTS BUY_SELL (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    coin_ticker_name TEXT,
                    buy_datetime DATETIME,
                    lstm_prob FLOAT, 
                    gb_prob FLOAT,
                    xgboost_prob FLOAT,
                    buy_base_price FLOAT,
                    buy_krw INT, 
                    buy_fee INT, 
                    buy_price FLOAT, 
                    buy_coin_volume FLOAT, 
                    trail_datetime DATETIME, 
                    trail_price FLOAT,
                    sell_fee INT, 
                    sell_krw INT, 
                    trail_rate FLOAT, 
                    total_krw INT, 
                    trail_up_count INT,                    
                    status TINYINT
                )"""

#print(select_all_from_order_book_for_one_coin)


### sell.py
select_all_bought_or_trailed_coin_names_sql = """
    SELECT * FROM BUY_SELL WHERE status=? or status=?;
"""

update_trail_coin_info_sql = """
    UPDATE BUY_SELL SET trail_datetime=?, trail_price=?, sell_fee=?, sell_krw=?, trail_rate=?, total_krw=?, trail_up_count=?, status=? 
    WHERE coin_ticker_name=? and buy_datetime=?;
"""

### buy.py
select_coin_ticker_name_by_status_sql = """
    SELECT coin_ticker_name FROM BUY_SELL WHERE status=0 or status=1;
"""

select_current_base_datetime_by_datetime = """
    SELECT base_datetime, ask_price_0 FROM 'KRW_{0}_ORDER_BOOK' WHERE base_datetime='{1}';
"""

select_total_krw = """
    SELECT total_krw FROM BUY_SELL ORDER BY id DESC LIMIT 1;
"""

insert_buy_try_coin_info = """
    INSERT INTO BUY_SELL (
        coin_ticker_name, 
        buy_datetime,
        lstm_prob,
        gb_prob,
        xgboost_prob,
        buy_base_price, 
        buy_krw, 
        buy_fee, 
        trail_datetime,
        buy_price, 
        buy_coin_volume,
        trail_price, 
        trail_rate, 
        total_krw, 
        trail_up_count,
        status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

select_buy_prohibited_coins_sql = """
    SELECT buy_base_price FROM BUY_SELL WHERE coin_ticker_name=? and DATETIME(buy_datetime) > DATETIME(?, '-1 day');
"""

### statistics
select_all_buy_sell_sql = "SELECT * FROM BUY_SELL ORDER BY id DESC;"

select_one_record_KRW_BTC_sql = """
SELECT base_datetime FROM KRW_BTC_ORDER_BOOK ORDER BY collect_timestamp DESC, base_datetime DESC LIMIT 1;
"""

count_rows_KRW_BTC_sql = "SELECT count(*) FROM KRW_BTC_ORDER_BOOK;"

###
# select_last_arrangement_base_datetime_for_coin_name = """
#     SELECT last_arrangement_base_datetime FROM ORDER_BOOK_ARRANGEMENT WHERE coin_ticker_name=?;
# """
#
# insert_last_arrangement_base_datetime_for_coin_name = """
#     INSERT INTO ORDER_BOOK_ARRANGEMENT (coin_ticker_name, last_arrangement_base_datetime) VALUES (?, ?);
# """
#
# update_last_arrangement_base_datetime_for_coin_name = """
#     UPDATE ORDER_BOOK_ARRANGEMENT SET last_arrangement_base_datetime=? WHERE coin_ticker_name=?;
# """

class SqliteHandler:
    def __init__(self):
        with sqlite3.connect(sqlite3_buy_sell_db_filename, timeout=10, check_same_thread=False) as conn:
            conn.execute("PRAGMA busy_timeout = 3000")

    def create_buy_sell_table(self, coin_names):
        with sqlite3.connect(sqlite3_buy_sell_db_filename, timeout=10, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute(create_buy_sell_table)
            conn.commit()

    # def create_order_book_arrangement_table(self):
    #     with sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, check_same_thread=False) as conn:
    #         cursor = conn.cursor()
    #         cursor.execute("""
    #             CREATE TABLE IF NOT EXISTS ORDER_BOOK_ARRANGEMENT (
    #                 id INTEGER PRIMARY KEY AUTOINCREMENT,
    #                 coin_ticker_name TEXT,
    #                 last_arrangement_base_datetime DATETIME
    #             )"""
    #         )
    #         conn.commit()

    def create_order_book_table(self, coin_names):
        with sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, check_same_thread=False) as conn:
            cursor = conn.cursor()

            for coin_name in coin_names:
                cursor.execute(create_order_book_table.format(coin_name)
                )

            conn.commit()

    def drop_buy_sell_tables(self):
        with sqlite3.connect(sqlite3_buy_sell_db_filename, timeout=10, check_same_thread=False) as conn:
            cursor = conn.cursor()
            table_name = "BUY_SELL"
            cursor.execute("DROP TABLE IF EXISTS {0}".format(table_name))

            conn.commit()

    def drop_order_book_tables(self, coin_names):
        with sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, check_same_thread=False) as conn:
            cursor = conn.cursor()
            for coin_name in coin_names:
                table_name = "KRW_" + coin_name + "_ORDER_BOOK"
                cursor.execute("DROP TABLE IF EXISTS {0}".format(table_name))

            conn.commit()

if __name__ == "__main__":
    upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

    sql_handler = SqliteHandler()
    sql_handler.create_buy_sell_table(upbit.get_all_coin_names())
    #sql_handler.create_order_book_table(upbit.get_all_coin_names())

    # sql_handler.drop_buy_sell_tables()
    # sql_handler.drop_order_book_tables(upbit.get_all_coin_names())

