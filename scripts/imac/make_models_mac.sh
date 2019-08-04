#!/bin/bash

cd $HOME/git/trade

$HOME/anaconda/envs/trade/bin/python -m predict.make_models >> $HOME/git/trade/logs/error/make_models.log 2>&1

