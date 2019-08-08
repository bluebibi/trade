#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda3/envs/trade/bin/python -m upbit.upbit_order_book_recorder >> $HOME/git/trade/logs/error/upbit_order_book_recorder.log 2>&1
sudo sync && echo 3 > /proc/sys/vm/drop_caches
$HOME/anaconda3/envs/trade/bin/python -m predict.buy >> $HOME/git/trade/logs/error/buy.log 2>&1
