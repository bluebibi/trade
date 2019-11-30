#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda/envs/trade/bin/python -m codes.predict.buy >> $HOME/git/trade/logs/error/buy.log 2>&1

