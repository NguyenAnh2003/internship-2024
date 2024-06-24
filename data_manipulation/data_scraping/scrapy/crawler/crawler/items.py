# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.item import Item, Field
import uuid

class CrawlerItem(Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    id = uuid.uuid4() # generate unique id
    feedback = Field()
    title = Field() # title of comment
    
