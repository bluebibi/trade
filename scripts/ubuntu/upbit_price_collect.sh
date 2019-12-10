#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda3/envs/trade/bin/python -m codes.upbit.recorder.price_collect >> $HOME/git/trade/logs/price_collect.log 2>&1