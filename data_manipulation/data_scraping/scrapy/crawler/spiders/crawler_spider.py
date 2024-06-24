from typing import Any, Iterable
import scrapy
from scrapy.http import Response, Request
from items import CrawlerItem


class DataSpider(scrapy.Spider):
    name = "web_data"
    start_urls = ["https://www.tripadvisor.com/"]
    allowed_domains = ["https://www.tripadvisor.com/"]
    
    def start_requests(self):
        url = "https://www.tripadvisor.com/Attraction_Review-g293916-d450970-Reviews-BTS_Skytrain-Bangkok.html"
        yield scrapy.Request(url, callback=self.parse)

    def parse(self, response: Response):
        comment = response.css("div._c")

        comment_item = CrawlerItem()
        comment_item["feedback"] = comment.css("")
