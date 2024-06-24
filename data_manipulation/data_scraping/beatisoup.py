from bs4 import BeautifulSoup
import requests
from urllib.request import Request, urlopen


URL = "https://www.tripadvisor.com/Attraction_Review-g293916-d450970-Reviews-BTS_Skytrain-Bangkok.html"


req = Request(
    url=URL, 
    headers={'User-Agent': 'Mozilla/5.0'}
)
webpage = urlopen(req).read()

# def scraping():
#     page = requests.get(URL)
#     soup = BeautifulSoup(page.content, "lxml")
#     # item = soup.find_all("div", class_="_c")
#     # print(item)

# scraping()