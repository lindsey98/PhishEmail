import shutil

import tldextract

from lib.web_utils.CustomDriver import CustomWebDriver
import os
from tldextract import tldextract
import time
import shutil
from datetime import datetime
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import sys
import hashlib

# Global variable for the executor
global_executor = None

def signal_handler(signal, frame):
    print("Signal received, shutting down...")
    if global_executor:
        global_executor.shutdown(wait=False)
    sys.exit(0)

# Setting the signal handler
signal.signal(signal.SIGINT, signal_handler)

def process_urls(url_list, max_workers=1):
    global global_executor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        global_executor = executor
        futures = {executor.submit(crawl, url, driver, './datasets/sjtu_phish/20231221'): url for url in url_list}
        for future in as_completed(futures):
            url = futures[future]
            try:
                success = future.result()
                print(f"Processed {url}: {success}")
            except Exception as exc:
                print(f"{url} generated an exception: {exc}")
        global_executor = None

def crawl(url, driver, requests_dir):
    os.makedirs(requests_dir, exist_ok=True)

    url_hash = hashlib.md5(url.encode()).hexdigest()
    domain = f"{url_hash}_{tldextract.extract(url).domain + '.' + tldextract.extract(url).suffix}"

    os.makedirs(os.path.join(requests_dir, domain), exist_ok=True)
    html_path = os.path.join(requests_dir, domain, "html.txt")
    screenshot_path = os.path.join(requests_dir, domain, "shot.png")
    info_path = os.path.join(requests_dir, domain, 'info.json')
    if os.path.exists(screenshot_path):
        return True

    success = False
    info_dict = {'url': url, 'timestamp': datetime.now().isoformat()}
    try:
        driver.delete_all_cookies()
        driver.get(url)
        time.sleep(3) # wait for the page to be fully loaded
        success = driver.save_screenshot(screenshot_path)
    except Exception as e:
        shutil.rmtree(os.path.join(requests_dir, domain))
        return False

    try:
        with open(html_path, "w") as f:
            f.write(driver.page_source)
    except:
        pass

    with open(info_path, "w") as f:
        json.dump(info_dict, f)
    return success


if __name__ == '__main__':

    driver = CustomWebDriver.boot(proxy_server="http://127.0.0.1:7890")  # Using the proxy_url variable
    time.sleep(3)
    driver.set_script_timeout(5)
    driver.set_page_load_timeout(5)

    to_process = [x.strip() for x in open('./datasets/phishing_url.20231221.txt').readlines()]
    process_urls(to_process)