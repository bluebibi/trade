#!/bin/bash
#
cd $HOME/git/trade

$HOME/anaconda3/envs/trade/bin/python -m codes.predict.make_models >> $HOME/git/trade/logs/error/make_models.log 2>&1

