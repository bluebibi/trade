#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda3/envs/trade/bin/python -m db.statistics >> $HOME/git/trade/logs/error/statistics.log 2>&1

