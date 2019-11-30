#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda/envs/trade/bin/python -m codes.predict.sell >> $HOME/git/trade/logs/error/sell.log 2>&1

