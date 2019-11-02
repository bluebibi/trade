#!/bin/bash
#
cd $HOME/git/trade/db/
ssh -i ~/yh21-han-aws-key-seoul.pem ubuntu@15.164.143.230 "cp /home/ubuntu/git/trade/web/db/upbit_order_book_info.db /home/ubuntu/git/trade/web/db/upbit_order_book_info.db.new"
ssh -i ~/yh21-han-aws-key-seoul.pem ubuntu@15.164.143.230 "cp /home/ubuntu/git/trade/web/db/upbit_buy_sell.db /home/ubuntu/git/trade/web/db/upbit_buy_sell.db.new"

mv upbit_order_book_info.db upbit_order_book_info.db.old
scp -i ~/yh21-han-aws-key-seoul.pem ubuntu@15.164.143.230:/home/ubuntu/git/trade/web/db/upbit_order_book_info.db.new ../web/db/upbit_order_book_info.db
scp -i ~/yh21-han-aws-key-seoul.pem ubuntu@15.164.143.230:/home/ubuntu/git/trade/web/db/upbit_buy_sell.db.new ../web/db/upbit_buy_sell.db
