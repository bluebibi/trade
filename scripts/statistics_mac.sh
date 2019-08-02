#!/bin/bash
#
cd $HOME/git/upbit_auto_trade

$HOME/anaconda3/envs/upbit_auto_trade/bin/python -m db.statistics >> $HOME/git/upbit_auto_trade/logs/error/statistics.log 2>&1

