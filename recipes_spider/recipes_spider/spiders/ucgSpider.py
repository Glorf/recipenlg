# -*- coding: utf-8 -*-
import scrapy
import logging

from recipes_spider.items import RecipesItem


class UcgspiderSpider(scrapy.Spider):
    name = 'ucgSpider'
    allowed_domains = ['www.ultimatecookingguide.com']
    start_urls = ['https://www.ultimatecookingguide.com/recipes/']

    def parse(self, response):
        logging.info("Parsing: "+response.url)
        result = RecipesItem()
        try:
            result['title'] = response.css('h1::text').get()
            result['ingredients'] = response.css('ul li::text').getall()
            result['directions'] = response.css('ol li::text').getall()
            result['link'] = response.url
            if result['ingredients'] and result['directions']:
                yield result
        except:
            logging.warning("Unable to get recipe from: " + response.url)
        logging.info("Parsed: "+response.url)

        links = response.css('a::attr(href)').getall()
        links = list(filter(lambda x: x.find('recipes/') >= 0, links))
        logging.info("Retrived "+str(len(links)))
        for l in links:
            yield response.follow(l, callback=self.parse)


        
