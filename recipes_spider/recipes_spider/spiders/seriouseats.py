# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from recipes_spider.items import RecipesItem

class SeriouseatsSpider(CrawlSpider):

    collection_name = 'scrapy_items_seriouseats'

    name = 'seriouseats'
    allowed_domains = ['www.seriouseats.com']
    start_urls = ['http://www.seriouseats.com/']

    rules = (
        Rule(LinkExtractor(allow=r'/recipes/[0-9]*/[0-9]*/'), callback='parse_item', follow=True),
        Rule(LinkExtractor(), follow=True)
    )

    custom_settings = {
        #'ITEM_PIPELINES' : {},
        'LOG_LEVEL' : 'INFO',
        'JOBDIR' : 'jobs/seriouseats-1'
    }

    def parse_item(self, response):
        result = RecipesItem()

        try:
            result['title'] = response.css('h1::text').get()
            result['ingredients'] = response.css('li.ingredient::text').getall()
            result['directions'] = response.css('div.recipe-procedure-text p::text').getall()
            result['link'] = response.url
            if result['ingredients'] and result['directions']:
                yield result
            else:
                self.logger.warning("Wrong data: " + response.url)
        except:
            self.logger.warning("Unable to get recipe from: " + response.url)
        self.logger.info("Parsed: "+response.url)
