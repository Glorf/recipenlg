# -*- coding: utf-8 -*-
import scrapy
import os

from recipes_spider.items import RecipesItem

#
# File-follow spider
#

class TasteofhomeSpider(scrapy.Spider):

    collection_name = 'scrapy_items_tasteofhome'

    name = 'tasteofhome'
    allowed_domains = ['www.tasteofhome.com']

    start_urls = ['https://www.tasteofhome.com/recipes/country-style-pork-loin/']

    custom_settings = {
        #'ITEM_PIPELINES' : {},
        'LOG_LEVEL' : 'INFO',
        'JOBDIR' : 'jobs/tasteofhome-1'
    }

    file_path = './files/tasteofhomelinks.txt'

    if(os.path.exists(file_path)):
        with open(file_path) as f:
            start_urls = list(map(lambda x: x.strip(), f.readlines()))
    else:
        # self.logger.warn("No file at:\t" + file_path)
        start_urls = ['https://www.tasteofhome.com/recipes/country-style-pork-loin/']

    def parse(self, response):
        result = RecipesItem()
        try:
            result['title'] = response.css('h1::text').get()
            result['ingredients'] = response.css('ul.recipe-ingredients__list li::text').getall()
            result['directions'] = response.css('ul.recipe-directions__list li span::text').getall()
            result['link'] = response.url
            if result['ingredients'] and result['directions']:
                yield result
        except:
            self.logger.warning("Unable to get recipe from: " + response.url)
        self.logger.info("Parsed: "+response.url)
