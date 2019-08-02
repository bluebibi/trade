#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda3/envs/trade/bin/python -m predict.sell >> $HOME/git/trade/logs/error/sell.log 2>&1

