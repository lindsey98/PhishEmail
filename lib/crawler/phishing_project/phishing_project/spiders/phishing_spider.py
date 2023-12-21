import scrapy
from scrapy_splash import SplashRequest
import hashlib
import os
from datetime import datetime
import json
from urllib.parse import urlparse
import base64
import io
from PIL import Image

script = """
function main(splash, args)
  splash:set_viewport_size(1920,1080)
  assert(splash:go(splash.args.url))
  assert(splash:wait(1))
  return {
    html = splash:html(),
    png = splash:png(),
    har = splash:har(),
    url = splash:url()
  }
end
"""

class PhishingSpider(scrapy.Spider):
    name = 'phishing'
    allowed_domains = ['example.com']  # 根据实际情况修改

    def start_requests(self):
        urls = [x.strip() for x in open('/home/ruofan/git_space/PhishEmail/datasets/phishing_url.20231221.txt').readlines()]
        for url in urls:
            splash_args = {
                'lua_source': script,
                'timeout': 10,
                'resource_timeout': 10
            }

            yield SplashRequest(url, self.parse, endpoint='execute', args=splash_args)

    def parse(self, response):
        url = response.url
        url_hash = hashlib.md5(url.encode()).hexdigest()
        domain = f"{url_hash}_{urlparse(url).netloc}"

        requests_dir = '/home/ruofan/git_space/PhishEmail/datasets/sjtu_phish/20231221'
        os.makedirs(requests_dir, exist_ok=True)
        os.makedirs(os.path.join(requests_dir, domain), exist_ok=True)

        shot_path = os.path.join(requests_dir, domain, "shot.png")
        html_path = os.path.join(requests_dir, domain, "html.txt")
        info_path = os.path.join(requests_dir, domain, 'info.json')
        har_path = os.path.join(requests_dir, domain, 'har.json')  # Path for the HAR file

        if os.path.exists(shot_path):
            return

        # 保存信息
        info_dict = {'url': url, 'timestamp': datetime.now().isoformat()}
        with open(info_path, 'w') as f:
            json.dump(info_dict, f)

        # 保存 screenshot 内容
        png_bytes = base64.b64decode(response.data['png'])
        screenshot_img = Image.open(io.BytesIO(png_bytes))
        with open(shot_path, 'wb+') as f:
            f.write(png_bytes)

        # 保存 HTML 内容
        with open(html_path, 'w') as f:
            f.write(response.text)

        # Save HAR data
        if 'har' in response.data:
            with open(har_path, 'w') as har_file:
                json.dump(response.data['har'], har_file)




