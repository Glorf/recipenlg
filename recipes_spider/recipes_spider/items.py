# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class RecipesItem(scrapy.Item):
    # define the fields for your item here like:
    title = scrapy.item.Field()
    ingredients = scrapy.item.Field()
    directions = scrapy.item.Field()
    link = scrapy.item.Field()
