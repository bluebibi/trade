#!/bin/bash
#
cd $HOME/git/trade/db/
mv upbit_order_book_info.db upbit_order_book_info.db.old
scp -i ~/yh21-han-aws-key-seoul.pem ubuntu@52.78.22.166:/home/ubuntu/git/trade/db/upbit_order_book_info.db .
