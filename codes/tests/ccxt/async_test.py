from time import time
import requests

urls = ["https://www.naver.com", "http://www.google.com", "https://www.nytimes.com", "https://www.mlb.com", "https://www.kakaocorp.com"]

begin = time()
result = []
for url in urls:
    response = requests.get(url)
    page = response.text
    result.append("{0} Bytes".format(len(page)))

print(result)
end = time()
print('실행 시간: {0:.3f}초'.format(end - begin))