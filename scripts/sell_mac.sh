#!/bin/bash
#
cd $HOME/git/upbit_auto_trade

$HOME/anaconda3/envs/upbit_auto_trade/bin/python -m predict.sell >> $HOME/git/upbit_auto_trade/logs/error/sell.log 2>&1

