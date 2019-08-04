#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda/envs/trade/bin/python -m predict.sell >> $HOME/git/trade/logs/error/sell.log 2>&1

