# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from recipes_spider.items import RecipesItem

def ing_parser(x):
    try:
        amount = x.xpath('.//span[contains(@itemprop, "amount")]/text()').get()
        name = x.xpath('.//span[contains(@itemprop, "name")]/text()').get()
        return amount + ' ' + name
    except:
        return x.xpath('.//span[contains(@itemprop, "name")]/text()').get()



class TastykitchenSpider(CrawlSpider):

    collection_name = 'scrapy_items_tastykitchen'

    name = 'tastykitchen'
    allowed_domains = ['tastykitchen.com']

    custom_settings = {
        #'ITEM_PIPELINES' : {},
        'LOG_LEVEL' : 'INFO',
        'JOBDIR' : 'jobs/tastykitchen-1'
    }

    start_urls = ['https://tastykitchen.com/recipes/']

    follow = True

    rules = (
        Rule(LinkExtractor(allow=r'/recipes/category/[A-Za-z0-9]*'), follow=follow),
        Rule(LinkExtractor(allow=r'/recipes/page/[A-Za-z0-9]*'), follow=follow),
        Rule(LinkExtractor(allow=r'/recipes/[A-Za-z0-9]*/', deny=(r'/recipes/wp-login[A-Za-z0-9]*', r'/recipes/members/[A-Za-z0-9]*')), 
        callback='parse_item', follow=follow),
        Rule(LinkExtractor(), follow=follow)
    )

    
    def parse_item(self, response):
        # self.logger.info("Parsing: "+response.url)
        result = RecipesItem()

        try:
            result['title'] = response.css('h1::text').get()
            result['ingredients'] = list(map(ing_parser, response.css('ul.ingredients li').xpath('.//span[contains(@itemprop, "ingredient")]')))
            result['directions'] = response.css('div.prep-instructions span p::text').getall()
            result['link'] = response.url
            if result['ingredients'] and result['directions']:
                yield result
            else:
                self.logger.warning("Wrong data: " + response.url)
        except:
            self.logger.warning("Unable to get recipe from: " + response.url)
        self.logger.info("Parsed: "+response.url)
