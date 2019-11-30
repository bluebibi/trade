#!/bin/bash
#
cd $HOME/git/trade

sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
$HOME/anaconda3/envs/trade/bin/python -m codes.predict.buy >> $HOME/git/trade/logs/error/buy.log 2>&1

