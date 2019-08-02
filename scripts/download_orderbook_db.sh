#!/bin/bash
#
cd $HOME/git/trade/db/
mv upbit_order_book_info.db upbit_order_book_info.db.old
scp -i ~/yh21-han-aws-key-seoul.pem ubuntu@13.125.56.68:/home/ubuntu/git/trade/db/upbit_order_book_info.db .