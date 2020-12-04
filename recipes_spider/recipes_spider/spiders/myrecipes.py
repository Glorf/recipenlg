# -*- coding: utf-8 -*-
import scrapy

import os
import json

from recipes_spider.items import RecipesItem

#
# File-follow spider
#

class MyrecipesSpider(scrapy.Spider):

    collection_name = 'scrapy_items_myrecipes'

    name = 'myrecipes'
    allowed_domains = ['www.myrecipes.com']
    start_urls = ['http://www.myrecipes.com/']

    custom_settings = {
        # 'ITEM_PIPELINES' : {},
        'LOG_LEVEL' : 'INFO',
        'JOBDIR' : 'jobs/myrecipes-1'
    }

    file_path = './files/myrecipeslinks.txt'

    if(os.path.exists(file_path)):
        with open(file_path) as f:
            start_urls = list(map(lambda x: x.strip(), f.readlines()))
    else:
        # self.logger.warn("No file at:\t" + file_path)
        start_urls = ['https://www.myrecipes.com/recipe/french-onion-soup-casserole']

    cookie = {
        "_gcl_au":"1.1.1335102055.1570204792",
        "ajs_anonymous_id":"\"f05a55bf-2e99-4ff8-ba81-72c53c755e9c\"",
        "ajs_group_id":"null",
        "ajs_user_id":"null",
        "euConsent":"true",
        "euConsentId":"db727f92-9a4f-4176-af82-fcd545cfd527",
        "globalTI_SID":"db727f92-9a4f-4176-af82-fcd545cfd527",
        "monetate_profile":"{\"mdpMember\":false}",
        "muuid_cnt":"1",
        "muuid_date":"1570204791857",
        "muuid_link":"24fb3817-3db9-4286-b1a9-ca1070016135",
        "sfdmpConsentLogged":"true"
    }

    def start_requests(self):
        for u in self.start_urls:
            req = scrapy.Request(u, cookies=self.cookie, callback=self.parse, dont_filter=True)
            # self.logger.info(req.url)
            yield req

    def parse(self, response):
        result = RecipesItem()
        # self.logger.info('Parsing: '+response.url)
        if response.css('form.gdpr-form').get():
            self.logger.info("GDPR")
            yield response.follow(response.url, method='GET', callback=self.parse, cookies = self.cookie)
        try:
            result['title'] = response.css('h1.headline::text').get().strip()
            result['ingredients'] = response.css('div.ingredients').css('li::text').getall()
            result['directions'] = response.css('div.step').css('p::text').getall()
            result['link'] = response.url
            # if result['ingredients'] and result['directions']:
            yield result
        except:
            self.logger.warning("Unable to get recipe from: " + response.url)
        self.logger.info("Parsed: "+response.url)
