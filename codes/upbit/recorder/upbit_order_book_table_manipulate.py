# with mysql_engine.connect() as con:
#     for c_idx, coin_name in enumerate(coin_names):
#         con.execute('ALTER TABLE KRW_{0}_ORDER_BOOK MODIFY `collect_timestamp` bigint;'.format(coin_name))
#         print(c_idx, coin_name)