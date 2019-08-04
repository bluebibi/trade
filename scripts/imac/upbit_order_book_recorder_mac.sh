#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda/envs/trade/bin/python -m upbit.upbit_order_book_recorder >> $HOME/git/trade/logs/error/upbit_order_book_recorder.log 2>&1
