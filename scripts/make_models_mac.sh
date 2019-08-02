#!/bin/bash
#
cd $HOME/git/upbit_auto_trade

$HOME/anaconda3/envs/upbit_auto_trade/bin/python -m predict.make_models >> $HOME/git/upbit_auto_trade/logs/error/make_models.log 2>&1

