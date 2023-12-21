import shutil

import tldextract

import time
import scrapy
import hashlib
import os
from datetime import datetime
import json
from urllib.parse import urlparse

class PhishingSpider(scrapy.Spider):
    name = 'phishing'
    allowed_domains = ['example.com']  # 根据实际情况修改

    def start_requests(self):
        urls = [x.strip() for x in open('./datasets/phishing_url.20231221.txt').readlines()]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        url = response.url
        url_hash = hashlib.md5(url.encode()).hexdigest()
        domain = f"{url_hash}_{urlparse(url).netloc}"

        requests_dir = './datasets/sjtu_phish/20231221'
        os.makedirs(requests_dir, exist_ok=True)
        os.makedirs(os.path.join(requests_dir, domain), exist_ok=True)
        html_path = os.path.join(requests_dir, domain, "html.txt")
        info_path = os.path.join(requests_dir, domain, 'info.json')

        # 保存 HTML 内容
        with open(html_path, 'w') as f:
            f.write(response.text)

        # 保存信息
        info_dict = {'url': url, 'timestamp': datetime.now().isoformat()}
        with open(info_path, 'w') as f:
            json.dump(info_dict, f)



