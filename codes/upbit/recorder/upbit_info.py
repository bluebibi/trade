import sys, os

import ccxt
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import warnings

from codes.upbit.recorder.upbit_selenium import get_coin_info

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

    birth = Column(String)
    total_markets = Column(String)
    num_exchanges = Column(String)
    period_block_creation = Column(String)
    mine_reward_unit = Column(String)
    total_limit = Column(String)
    consensus_protocol = Column(String)
    web_site = Column(String)
    whitepaper = Column(String)
    block_site = Column(String)
    twitter_url = Column(String)
    intro = Column(String)

    def __init__(self, *args, **kw):
        super(UpbitInfo, self).__init__(*args, **kw)

    def get_id(self):
        return self.id

    def to_dict(self):
        d = {}
        d["coin_name"] = self.market.replace("KRW-", "")
        d["market"] = self.market
        d["korean_name"] = self.korean_name
        d["eng_name"] = self.eng_name
        d["limit_amount_max"] = self.limit_amount_max
        d["limit_amount_min"] = self.limit_amount_min
        d["limit_cost_max"] = self.limit_cost_max
        d["limit_cost_min"] = self.limit_cost_min
        d["limit_price_max"] = self.limit_price_max
        d["limit_price_min"] = self.limit_price_min
        d["maker"] = self.maker
        d["taker"] = self.taker
        d["percentage"] = self.percentage
        d["precision_amount"] = self.precision_amount
        d["precision_price"] = self.precision_price
        d["tierBased"] = self.tierBased

        d["birth"] = self.birth
        d["total_markets"] = self.total_markets
        d["num_exchanges"] = self.num_exchanges
        d["period_block_creation"] = self.period_block_creation
        d["mine_reward_unit"] = self.mine_reward_unit
        d["total_limit"] = self.total_limit
        d["consensus_protocol"] = self.consensus_protocol
        d["web_site"] = self.web_site
        d["whitepaper"] = self.whitepaper
        d["block_site"] = self.block_site
        d["twitter_url"] = self.twitter_url

        d["intro"] = self.intro

        return d

Base.metadata.create_all(engine)
db_session = sessionmaker(bind=engine)
db_session = db_session()

def get_market_info(quote='KRW'):
    upbit_markets = ccxt_upbit.load_markets()
    markets_krw = []
    for symbol, market in upbit_markets.items():
        if '/{0}'.format(quote) in symbol:
            print(symbol)
            market_name = market['info']['market']
            q = db_session.query(UpbitInfo).filter(UpbitInfo.market == market_name)
            m = q.first()

            if m is None:
                upbit_info = UpbitInfo()
                upbit_info.market = market['info']['market']
                coin_name = market['info']['market'].replace("KRW-", "")

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

                info = get_coin_info(coin_name)

                upbit_info.birth = info["birth"] if "birth" in info else None
                upbit_info.total_markets = info["total_markets"] if "total_markets" in info else None
                upbit_info.num_exchanges = info["num_exchanges"] if "num_exchanges" in info else None
                upbit_info.period_block_creation = info["period_block_creation"] if "period_block_creation" in info else None
                upbit_info.mine_reward_unit = info["mine_reward_unit"] if "mine_reward_unit" in info else None
                upbit_info.total_limit = info["total_limit"] if "total_limit" in info else None
                upbit_info.consensus_protocol = info["consensus_protocol"] if "consensus_protocol" in info else None
                upbit_info.web_site = info["web_site"] if "web_site" in info else None
                upbit_info.whitepaper = info["whitepaper"] if "whitepaper" in info else None
                upbit_info.block_site = info["block_site"] if "block_site" in info else None
                upbit_info.twitter_url = info["twitter_url"] if "twitter_url" in info else None

                upbit_info.intro = info["intro"] if "intro" in info else None

                db_session.add(upbit_info)
                db_session.commit()
            else:
                m.market = market['info']['market']
                coin_name = market['info']['market'].replace("KRW-", "")

                m.korean_name = market['info']['korean_name']
                m.eng_name = market['info']['english_name']
                m.limit_amount_max = market['limits']['amount']['max']
                m.limit_amount_min = market['limits']['amount']['min']
                m.limit_cost_max = market['limits']['cost']['max']
                m.limit_cost_min = market['limits']['cost']['min']
                m.limit_price_max = market['limits']['price']['max']
                m.limit_price_min = market['limits']['price']['min']
                m.maker = market['maker']
                m.percentage = market['percentage']
                m.precision_amount = market['precision']['amount']
                m.precision_price = market['precision']['price']
                m.taker = market['taker']
                m.tierBased = market['tierBased']

                info = get_coin_info(coin_name)

                m.birth = info["birth"] if "birth" in info else None
                m.total_markets = info["total_markets"] if "total_markets" in info else None
                m.num_exchanges = info["num_exchanges"] if "num_exchanges" in info else None
                m.period_block_creation = info["period_block_creation"] if "period_block_creation" in info else None
                m.mine_reward_unit = info["mine_reward_unit"] if "mine_reward_unit" in info else None
                m.total_limit = info["total_limit"] if "total_limit" in info else None
                m.consensus_protocol = info["consensus_protocol"] if "consensus_protocol" in info else None
                m.web_site = info["web_site"] if "web_site" in info else None
                m.whitepaper = info["whitepaper"] if "whitepaper" in info else None
                m.block_site = info["block_site"] if "block_site" in info else None
                m.twitter_url = info["twitter_url"] if "twitter_url" in info else None
                m.intro = info["intro"] if "intro" in info else None

                db_session.commit()

    return markets_krw


if __name__ == "__main__":
    get_market_info("KRW")