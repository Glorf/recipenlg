# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from recipes_spider.items import RecipesItem

#
# site spider unfriendly, banning ip
#

class CdkitchenSpider(CrawlSpider):

    collection_name = 'scrapy_items_cdkitchen'

    name = 'cdkitchen'

    allowed_domains = ['www.cdkitchen.com']

    start_urls = ['http://www.cdkitchen.com/']

    custom_settings = {
        'ITEM_PIPELINES' : {},
        'LOG_LEVEL' : 'DEBUG',
        #'JOBDIR' : 'jobs/cdkitchen-1',
        'HTTPCACHE_ENABLED': False,
        'USER_AGENT': 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:69.0) Gecko/20100101 Firefox/69.0',
        'DOWNLOAD_DELAY': 2
    }

    rules = (
        Rule(LinkExtractor(allow=r'/recipes/recs/'), callback='parse_item', follow=True),
        Rule(LinkExtractor(), follow=True)
    )

    def parse_item(self, response):
        # self.logger.info("Parsing: "+response.url)
        result = RecipesItem()

        try:
            result['title'] = response.css('h1::text').get()
            result['ingredients'] = list(filter(lambda x: x.strip(), response.css('p.ml-30 span.ft-verdana span::text').getall()))
            result['directions'] = list(map(lambda x: x.strip(), response.css('div.ft-verdana p.ml-30::text').getall()))
            result['link'] = response.url
            if result['ingredients'] and result['directions']:
                yield result
            else:
                self.logger.warning("Wrong data: " + response.url)
                self.logger.warning(result)
        except:
            self.logger.warning("Unable to get recipe from: " + response.url)
        self.logger.info("Parsed: "+response.url)
