# -*- coding: utf-8 -*-
import scrapy
import os
import re

from recipes_spider.items import RecipesItem

#
# File-follow spider
#

class Food52Spider(scrapy.Spider):

    collection_name = 'scrapy_items_food52'

    name = 'food52'
    allowed_domains = ['food52.com']
    
    custom_settings = {
        # 'ITEM_PIPELINES' : {},
        'LOG_LEVEL' : 'INFO',
        'JOBDIR' : 'jobs/food52-1'
    }

    file_path = './files/food52links.txt'

    if(os.path.exists(file_path)):
        with open(file_path) as f:
            start_urls = list(map(lambda x: x.strip(), f.readlines()))
    else:
        # self.logger.warn("No file at:\t" + file_path)
        start_urls = ['https://food52.com/recipes/5795-grilled-salmon-with-green-asian-marinade']


    def parse(self, response):
        result = RecipesItem()
        try:
            result['title'] = response.css('h1.recipe__title::text').get().strip().replace(u'\xa0',' ')
            # result['ingredients'] = list(map(lambda x: x.replace('<br>','').replace('</span>','').replace('\n',' ').replace('  ',' ').replace('<span>','').replace('</li>','')[36:].strip(), response.css('div.recipe__list--ingredients').css('li').getall()))
            result['ingredients'] = list(map(lambda x: re.compile(r'<.*?>').sub('', x).strip().replace('\n\n', ' '), response.css('div.recipe__list--ingredients').css('li').getall()))
            result['directions'] = list(filter(lambda x: x, list(map(lambda x: x.strip(), response.css('li.recipe__list-step *::text').getall()))))
            result['link'] = response.url
            if result['ingredients'] and result['directions']:
                yield result
        except:
            self.logger.warning("Unable to get recipe from: " + response.url)
        self.logger.info("Parsed: "+response.url)
