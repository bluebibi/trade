import requests
page = requests.get("https://upbit.com/exchange?code=CRIX.UPBIT.KRW-BTC")
print(page.content)