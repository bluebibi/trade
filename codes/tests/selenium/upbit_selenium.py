from bs4 import BeautifulSoup
from selenium import webdriver
import sys, os

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from common.global_variables import CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt
from codes.upbit.upbit_api import Upbit
import platform
print(platform.system())

options = webdriver.ChromeOptions()
options.add_argument("headless")

driver = webdriver.Chrome('./chromedriver', options=options)
driver.implicitly_wait(3)
driver.get('https://upbit.com')
driver.implicitly_wait(5)

upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)


def get_info(coin_name):
    driver.get('https://upbit.com/exchange?code=CRIX.UPBIT.KRW-{0}'.format(coin_name))
    driver.find_element_by_css_selector('article > span.titB > div.inforTab > dl > dd > a').click()
    html = driver.page_source

    soup = BeautifulSoup(html, 'html.parser') # BeautifulSoup사용하기

    info_table_css_selector = 'article > div > div.scrollB > div > div > span.inforB > div.tableLayout > table'

    table = soup.select(info_table_css_selector)[0]
    rows = table.findChildren(['th', 'tr'])

    info = {}
    for row in rows:
        cell_titles = row.findChildren('th')
        cells = row.findChildren('td')
        for idx, cell_title in enumerate(cell_titles):
            if cell_title.string == "최초발행":
                info['birth'] = cells[idx].string
            elif cell_title.string == '시가총액':
                info['total_markets'] = cells[idx].string
            elif cell_title.string == '상장거래소':
                info['num_exchanges'] = cells[idx].string
            elif cell_title.string == '블록 생성주기':
                info['period_block_creation'] = cells[idx].string
            elif cell_title.string == '채굴 보상량':
                info['mine_reward_unit'] = cells[idx].string
            elif cell_title.string == '총 발행한도':
                info['total_limit'] = cells[idx].string
            elif cell_title.string == '합의 프로토콜':
                info['consensus_protocol'] = cells[idx].string
            elif cell_title.string == '발행량':
                info
            else:
                print("!!!!!!!!!!!!!!!!!!!!", cell_title.string)

    return info


if __name__ == "__main__":
    coin_names = upbit.get_all_coin_names()

    coin_info = {}
    for coin_name in coin_names:
        info = get_info(coin_name)
        coin_info[coin_name] = info
        print(coin_name, info)
