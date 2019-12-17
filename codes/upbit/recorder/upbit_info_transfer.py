import sys, os

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from codes.upbit.recorder.upbit_info import UpbitInfo
from codes.upbit.upbit_api import Upbit
from common.global_variables import *
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

import warnings
warnings.filterwarnings('ignore')

upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

upbit_info_engine = create_engine(
    'sqlite:///{0}/web/db/upbit_info.db'.format(PROJECT_HOME),
    echo=False, connect_args={'check_same_thread': False}
)
upbit_info_session = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=upbit_info_engine)
        )

mysql_engine = create_engine('mysql+mysqlconnector://{0}:{1}@{2}/record'.format(
            MYSQL_ID, MYSQL_PASSWORD, MYSQL_HOST
        ), encoding='utf-8')

mysql_db_session = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=mysql_engine)
        )

if __name__ == "__main__":
    upbit_infos = upbit_info_session.query(UpbitInfo).all()
    for upbit_info in upbit_infos:
        local_object = mysql_db_session.merge(upbit_info)
        mysql_db_session.add(local_object)
        mysql_db_session.commit()



