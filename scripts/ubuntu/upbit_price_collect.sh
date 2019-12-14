#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda3/envs/trade/bin/python -m codes.upbit.recorder.upbit_price_collector >> $HOME/git/trade/logs/error/upbit_price_collector.log 2>&1