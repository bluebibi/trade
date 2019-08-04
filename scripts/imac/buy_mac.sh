#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda/envs/trade/bin/python -m predict.buy >> $HOME/git/trade/logs/error/buy.log 2>&1

