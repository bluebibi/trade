#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda3/envs/trade/bin/python -m upbit.upbit_order_book_recorder >> $HOME/git/trade/logs/error/upbit_order_book_recorder.log 2>&1
$HOME/anaconda3/envs/trade/bin/python -m predict.buy >> $HOME/git/trade/logs/error/buy.log 2>&1
$HOME/anaconda3/envs/trade/bin/python -m db.statistics >> $HOME/git/trade/logs/error/statistics.log 2>&1
