#!/bin/bash
#
cd $HOME/git/trade/db/
ssh -i ~/yh21-han-aws-key-seoul.pem -o BindAddress=192.168.8.101 ubuntu@15.164.143.230 "cp /home/ubuntu/git/trade/db/upbit_order_book_info.db /home/ubuntu/git/trade/db/upbit_order_book_info.db.new"

mv upbit_order_book_info.db upbit_order_book_info.db.old
scp -i ~/yh21-han-aws-key-seoul.pem -o BindAddress=192.168.8.101 ubuntu@15.164.143.230:/home/ubuntu/git/trade/db/upbit_order_book_info.db.new ./upbit_order_book_info.db
