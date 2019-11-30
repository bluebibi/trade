import locale
import sys, os

import ccxt
from pytz import timezone
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
import warnings

warnings.filterwarnings("ignore")

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

engine = create_engine(
    'sqlite:///{0}/web/db/upbit_info.db'.format(PROJECT_HOME),
    echo=False, connect_args={'check_same_thread': False}
)
Base = declarative_base()

ccxt_upbit = ccxt.upbit()


class UpbitInfo(Base):
    __tablename__ = "UPBIT_INFO"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    market = Column(String)
    korean_name = Column(String)
    eng_name = Column(String)
    limit_amount_max = Column(Float)
    limit_amount_min = Column(Float)
    limit_cost_max = Column(Float)
    limit_cost_min = Column(Float)
    limit_price_max = Column(Float)
    limit_price_min = Column(Float)
    maker = Column(Float)
    taker = Column(Float)
    percentage = Column(Boolean)
    precision_amount = Column(Float)
    precision_price = Column(Float)
    tierBased = Column(Boolean)

    def __init__(self, *args, **kw):
        super(UpbitInfo, self).__init__(*args, **kw)

    def get_id(self):
        return self.id

Base.metadata.create_all(engine)
db_session = sessionmaker(bind=engine)
db_session = db_session()

def get_market_info(quote='KRW'):
    upbit_markets = ccxt_upbit.load_markets()
    markets_krw = []
    for symbol, market in upbit_markets.items():
        upbit_info = UpbitInfo()

        if '/{0}'.format(quote) in symbol:
            market = market['info']['market']
            q = db_session.query(UpbitInfo).filter(UpbitInfo.market == market)
            market = q.first()
            if market is None:
                upbit_info.market = market['info']['market']
                upbit_info.korean_name = market['info']['korean_name']
                upbit_info.eng_name = market['info']['english_name']
                upbit_info.limit_amount_max = market['limits']['amount']['max']
                upbit_info.limit_amount_min = market['limits']['amount']['min']
                upbit_info.limit_cost_max = market['limits']['cost']['max']
                upbit_info.limit_cost_min = market['limits']['cost']['min']
                upbit_info.limit_price_max = market['limits']['price']['max']
                upbit_info.limit_price_min = market['limits']['price']['min']
                upbit_info.maker = market['maker']
                upbit_info.percentage = market['percentage']
                upbit_info.precision_amount = market['precision']['amount']
                upbit_info.precision_price = market['precision']['price']
                upbit_info.taker = market['taker']
                upbit_info.tierBased = market['tierBased']

                db_session.add(upbit_info)
                db_session.commit()
            else:
                market.market = market['info']['market']
                market.korean_name = market['info']['korean_name']
                market.eng_name = market['info']['english_name']
                market.limit_amount_max = market['limits']['amount']['max']
                market.limit_amount_min = market['limits']['amount']['min']
                market.limit_cost_max = market['limits']['cost']['max']
                market.limit_cost_min = market['limits']['cost']['min']
                market.limit_price_max = market['limits']['price']['max']
                market.limit_price_min = market['limits']['price']['min']
                market.maker = market['maker']
                market.percentage = market['percentage']
                market.precision_amount = market['precision']['amount']
                market.precision_price = market['precision']['price']
                market.taker = market['taker']
                market.tierBased = market['tierBased']

                db_session.commit()

    return markets_krw


if __name__ == "__main__":
    get_market_info("KRW")