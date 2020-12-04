# -*- coding: utf-8 -*-
import scrapy
import logging

from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from recipes_spider.items import RecipesItem


class UcgcrawlerSpider(CrawlSpider):
    name = 'ucgCrawler'
    allowed_domains = ['ultimatecookingguide.com']
    start_urls = ['http://ultimatecookingguide.com/recipes/']

    rules = (
        Rule(LinkExtractor(allow=r'recipes/*'), callback='parse_item', follow=True),
        # Rule(LinkExtractor(allow=r'recipes/*/*/*\.html'), callback='parse_item', follow=True),
        # Rule(LinkExtractor(allow=r'recipes/*')),
    )

    def parse_item(self, response):
        self.logger.info("Parsing: "+response.url)
        result = RecipesItem()

        try:
            result['title'] = response.css('h1::text').get()
            result['ingredients'] = response.css('ul li::text').getall()
            result['directions'] = response.css('ol li::text').getall()
            result['link'] = response.url
            if result['ingredients'] and result['directions']:
                yield result
            else:
                self.logger.warning("Unable to get recipe from: " + response.url)
        except:
            self.logger.warning("Unable to get recipe from: " + response.url)
        self.logger.info("Parsed: "+response.url)