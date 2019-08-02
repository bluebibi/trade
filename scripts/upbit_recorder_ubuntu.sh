#!/bin/bash
#
cd $HOME/git/upbit_auto_trade

$HOME/anaconda3/envs/upbit_auto_trade/bin/python -m upbit.upbit_recorder >> $HOME/git/upbit_auto_trade/logs/error/upbit_recorder.log 2>&1