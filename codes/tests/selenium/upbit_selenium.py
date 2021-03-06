from bs4 import BeautifulSoup
from selenium import webdriver
import sys, os

from selenium.webdriver.firefox.options import Options

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from common.global_variables import CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt
from codes.upbit.upbit_api import Upbit
import platform


if platform.system() == "Darwin":
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    driver = webdriver.Chrome('./chromedriver_mac', options=options)
    driver.implicitly_wait(3)
    driver.get('https://upbit.com')
    driver.implicitly_wait(5)

elif platform.system() == "Linux":
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--start-maximized')
    driver = webdriver.Firefox(executable_path='./geckodriver', options=options)
    driver.implicitly_wait(3)
    driver.get('https://upbit.com')
    driver.implicitly_wait(10)

else:
    driver = None


upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)


def get_info(coin_name):
    driver.get('https://upbit.com/exchange?code=CRIX.UPBIT.KRW-{0}'.format(coin_name))
    #driver.save_screenshot('picture.png')

    driver.find_element_by_css_selector('article > span.titB > div.inforTab > dl > dd > a').click()
    html = driver.page_source

    soup = BeautifulSoup(html, 'html.parser')

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
            else:
                print("!!!!!!!!!!!!!!!!!!!!", cell_title.string)

    website_css_selector = 'div.mainB > section.ty01 > article > div > div.scrollB > div > div > span.inforB > div.title > div.linkWrap > a'

    website_tags = soup.select(website_css_selector)
    for tag in website_tags:
        if "웹사이트" in tag.string:
            info["web_site"] = tag.get_attribute_list("href")[0]
        if "백서" in tag.string:
            info["whitepaper"] = tag.get_attribute_list("href")[0]
        if "블록조회" in tag.string:
            info["block_site"] = tag.get_attribute_list("href")[0]

    driver.switch_to.frame(driver.find_element_by_tag_name("iframe#twitter-widget-0"))
    twitter_css_selector = 'html > body > div.timeline-Widget > footer.timeline-Footer.u-cf > a.u-floatRight'
    driver.find_element_by_css_selector(twitter_css_selector)
    html = driver.page_source

    soup = BeautifulSoup(html, 'html.parser')
    tag = soup.select("a.u-floatRight")[0]

    info["twitter_url"] = tag.get_attribute_list("href")[0]

    return info

if __name__ == "__main__":
    coin_names = upbit.get_all_coin_names()

    coin_info = {}
    try:
        for coin_name in coin_names:
            info = get_info(coin_name)
            coin_info[coin_name] = info
            print(coin_name, info)
    except Exception as ex:
        print(ex)
        driver.close()