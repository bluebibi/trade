#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda/envs/trade/bin/python -m upbit.upbit_orderbook_recorder >> $HOME/git/trade/logs/error/upbit_orderbook_recorder.log 2>&1
