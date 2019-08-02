#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda3/envs/trade/bin/python -m upbit.upbit_recorder >> $HOME/git/trade/logs/error/upbit_recorder.log 2>&1
