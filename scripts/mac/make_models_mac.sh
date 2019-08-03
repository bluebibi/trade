#!/bin/bash
#
cd $HOME/git/trade/db/
mv upbit_order_book_info.db upbit_order_book_info.db.old
scp -i ~/yh21-han-aws-key-seoul.pem ubuntu@52.78.22.166:/home/ubuntu/git/trade/db/upbit_order_book_info.db .

cd $HOME/git/trade

$HOME/anaconda3/envs/trade/bin/python -m predict.make_models >> $HOME/git/trade/logs/error/make_models.log 2>&1

