# -*- coding: utf-8 -*-
import scrapy
import os

from recipes_spider.items import RecipesItem

#
# File-follow spider
#


class EpicuriousSpider(scrapy.Spider):

    collection_name = 'scrapy_items_epicurious'

    name = 'epicurious'
    allowed_domains = ['www.epicurious.com']
    start_urls = ['https://www.epicurious.com/recipes/member/views/chicken-divine-51789231']

    custom_settings = {
        #'ITEM_PIPELINES' : {},
        'LOG_LEVEL' : 'INFO',
        'JOBDIR' : 'jobs/epicurious-1'
    }

    file_path = './files/epicuriouslinks.txt'

    if(os.path.exists(file_path)):
        with open(file_path) as f:
            start_urls = list(map(lambda x: x.strip(), f.readlines()))
    else:
        # self.logger.warn("No file at:\t" + file_path)
        start_urls = ['https://www.epicurious.com/recipes/member/views/chicken-divine-51789231']

    def parse(self, response):
        result = RecipesItem()
        try:
            result['title'] = response.css('h1::text').get()
            result['ingredients'] = response.css('ul.ingredients li.ingredient::text').getall()
            result['directions'] = list(map(lambda x: x.strip(), response.css('ol.preparation-steps li.preparation-step::text').getall()))
            result['link'] = response.url
            if result['ingredients'] and result['directions']:
                yield result
        except:
            self.logger.warning("Unable to get recipe from: " + response.url)
        self.logger.info("Parsed: "+response.url)
