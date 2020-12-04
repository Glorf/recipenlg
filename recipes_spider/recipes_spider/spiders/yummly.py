# -*- coding: utf-8 -*-
import scrapy
import logging
import json
import os

from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from recipes_spider.items import RecipesItem


class YummlySpider(CrawlSpider):

    collection_name = 'scrapy_items_yummly'

    name = 'yummly'
    allowed_domains = ['yummly.com']

    # check path
    file_path = '/usr/src/app/files/hubs.txt'
    
    if(os.path.exists(file_path)):
        with open(file_path) as f:
            start_urls = list(map(lambda x: x.strip(), f.readlines()))
    else:
        # self.logger.warn("No file at:\t" + file_path)
        start_urls = ['https://www.yummly.com/recipes/cold-water-pastry']

    # allow custom settings
    custom_settings = {
        #'ITEM_PIPELINES' : {},
        'LOG_LEVEL' : 'INFO',
        'JOBDIR' : 'jobs/yummly-1'
    }

    rules = (
        Rule(LinkExtractor(allow=r'recipes/'), follow=True),
        Rule(LinkExtractor(allow=r'recipe/'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        item = {}
        self.logger.info("Code: " + str(response.status) + " Parsing: " + response.url)
        try:
            item = response.css('div.structured-data-info script::text').getall()[0]
            item = json.loads(item)
        except:
            self.logger.warning("Unable to get recipe from: " + response.url)
            pass
            
        key = 'recipeInstructions'
        result = dict()
        if key in item:
            result['title'] = item['name']
            result['ingredients'] = item['recipeIngredient']
            result['directions'] = list(map(lambda x: x['text'],item[key]))
            result['link'] = response.url
        elif item:
            result['name'] = response.css('a.source-link.micro-text::text').getall()[0]
            result['source'] = 'https://www.yummly.com' + response.css('a.source-link.micro-text').xpath('@href').getall()[0]
            result['link'] = response.url

        return result

        
    def parse_none(self, response):
        self.logger.info("NONE: " + response.url)
        pass
