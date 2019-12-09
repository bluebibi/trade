from time import time
import requests
import asyncio

urls = ["https://www.naver.com", "http://www.google.com", "https://www.nytimes.com", "https://www.mlb.com", "https://www.kakaocorp.com"]


async def fetch(url):
    response = await loop.run_in_executor(None, requests.get, url)  # run_in_executor 사용
    page = response.text
    return "{0} Bytes".format(len(page))


async def main():
    futures = [asyncio.ensure_future(fetch(url)) for url in urls]
    # 태스크(퓨처) 객체를 리스트로 만듦
    result = await asyncio.gather(*futures)  # 결과를 한꺼번에 가져옴
    print(result)


begin = time()
loop = asyncio.get_event_loop()  # 이벤트 루프를 얻음
loop.run_until_complete(main())  # main이 끝날 때까지 기다림
loop.close()  # 이벤트 루프를 닫음
end = time()
print('실행 시간: {0:.3f}초'.format(end - begin))