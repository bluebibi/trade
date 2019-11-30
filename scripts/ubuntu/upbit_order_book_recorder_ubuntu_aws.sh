#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda3/envs/trade/bin/python -m codes.upbit.upbit_order_book_recorder >> $HOME/git/trade/logs/error/upbit_order_book_recorder.log 2>&1
$HOME/anaconda3/envs/trade/bin/python -m codes.predict.buy >> $HOME/git/trade/logs/error/buy.log 2>&1
