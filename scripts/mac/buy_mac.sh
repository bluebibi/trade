#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda3/envs/trade/bin/python -m codes.predict.buy >> $HOME/git/trade/logs/error/buy.log 2>&1

