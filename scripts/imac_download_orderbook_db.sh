#!/bin/bash
#
cd $HOME/git/trade/web/db/
ssh -i ~/yh21-han-aws-key-seoul.pem -o BindAddress=192.168.8.102 ubuntu@15.164.143.230 "cp /home/ubuntu/git/trade/web/db/upbit_order_book_info.db /home/ubuntu/git/trade/web/db/upbit_order_book_info.db.new"

scp -i ~/yh21-han-aws-key-seoul.pem -o BindAddress=192.168.8.102 ubuntu@15.164.143.230:/home/ubuntu/git/trade/web/db/upbit_order_book_info.db.new /Users/yhhan/git/trade/web/db/upbit_order_book_info.db
