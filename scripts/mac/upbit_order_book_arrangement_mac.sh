#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda3/envs/trade/bin/python -m upbit.upbit_order_book_arrangement >> $HOME/git/trade/logs/error/upbit_order_book_arrangement.log 2>&1
