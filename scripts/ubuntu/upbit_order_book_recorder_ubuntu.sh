#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda3/envs/trade/bin/python -m codes.upbit.recorder.upbit_order_book_recorder >> $HOME/git/trade/logs/error/upbit_order_book_recorder.log 2>&1