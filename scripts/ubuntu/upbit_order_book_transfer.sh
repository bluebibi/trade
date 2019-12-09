#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda3/envs/trade/bin/python -m codes.upbit.upbit_order_book_transfer >> $HOME/git/trade/logs/error/upbit_order_book_transfer.log 2>&1