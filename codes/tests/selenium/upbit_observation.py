from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
from datetime import datetime
'''
    File name: main.py
    Author: DevHyung
    Date created: 2017-11-27
    Date last modified: 2017-11-27
    Python Version: 3.6
'''
dir = './chromedriver'  # 드라이브가 있는 경로
driver = webdriver.Chrome(dir)
driver.implicitly_wait(3)

def getReadyIdx():
    try:
        driver.set_window_size(1000, 1000)
        driver.get("https://upbit.com/exchange?")
        delay = 10  # seconds
        myElem = WebDriverWait(driver, delay).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="root"]/div/div/div[2]/section[2]/article[1]/span[2]/div/div/div/div[1]/table/tbody/tr[2]/td[3]')))
        print(">>> Page is ready ! ")
        bs = BeautifulSoup(driver.page_source, "lxml")
        div = bs.find("section", class_="ty02").find("div", class_="scrollB").find("table")
        trnum = len(div.find_all("tr"))  # 총 cnt
        readyidx = []
        namelist = []
        print(">>> Ready Page Search Start !")
        for idx in range(1, trnum + 1):
            data = driver.find_element_by_xpath('//*[@id="root"]/div/div/div[2]/section[2]/article[1]/span[2]/div/div/div/div[1]/table/tbody/tr[' + str(idx) + ']/td[3]')
            data.click()
            bs = BeautifulSoup(driver.page_source, "lxml")
            try:
                div = bs.find("section", class_="ty01").find("div", class_="txt").find("strong")
                readyidx.append(idx)
                namelist.append(div.get_text())
                print(div.get_text(), end=",")
            except:
                pass
            time.sleep(0.1)
        print("\n")
        print("Total " + str(len(readyidx)) + " detected !")
        print(">>> Ready Page Search Complete !")
    except TimeoutException:
        print("ERROR : Loading took too much time!")
    return readyidx,namelist

if __name__ == "__main__":
    while True:
        readyidx, namelist = getReadyIdx()
        print(">>> Start Detecting page change ")
        cycle = 0
        while True:
            cnt = 0
            if cycle == 1:
                print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + " : " + namelist[cnt] + " -> 변경되었음 ")
                print("_"*20)
                driver.close()
                driver = webdriver.Chrome(dir)
                break;
            for test in readyidx:
                data = driver.find_element_by_xpath('//*[@id="root"]/div/div/div[2]/section[2]/article[1]/span[2]/div/div/div/div[1]/table/tbody/tr[' + str(test) + ']/td[3]')
                data.click()
                bs = BeautifulSoup(driver.page_source, "lxml")
                try:
                    div = bs.find("section", class_="ty01").find("div", class_="txt").find("strong")
                except:# 변경감지
                    print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + " : " + namelist[cnt] + " -> 변경되었음 ")
                    print("_" * 20)
                    driver.close()
                    driver = webdriver.Chrome(dir)
                    break;
                # time.sleep(0.1) # 도는 시간 delay 0.1 => 0.1초
                cnt += 1
            print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + " : not detected...")
            cycle += 1