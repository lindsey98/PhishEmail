import shutil
import tldextract
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
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import (WebDriverException, NoSuchWindowException,
                                        SessionNotCreatedException, InvalidSessionIdException)

# Global variable for the executor
global_executor = None

def signal_handler(signal, frame):
    print("Signal received, shutting down...")
    if global_executor:
        global_executor.shutdown(wait=False)
    sys.exit(0)

# Setting the signal handler
signal.signal(signal.SIGINT, signal_handler)

# 线程局部变量来存储 WebDriver 实例
thread_local = threading.local()

def get_driver(recreate=False):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    if recreate or not hasattr(thread_local, "driver"):
        # If recreate is True or driver does not exist, create a new driver
        thread_local.driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
    return thread_local.driver

def process_urls(url_list, max_workers=16):
    global global_executor
    today = datetime.now().strftime('%Y%m%d')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        global_executor = executor
        futures = {executor.submit(crawl, url, f'./datasets/sjtu_phish/{today}'): url for url in url_list}
        for future in as_completed(futures):
            url = futures[future]
            try:
                success = future.result()
                print(f"Processed {url}: {success}")
            except Exception as exc:
                print(f"{url} generated an exception: {exc}")
        global_executor = None

    # 线程结束后清理所有 WebDriver 实例
    for thread in threading.enumerate():
        if hasattr(thread, "driver"):
            thread.driver.quit()

def crawl(url, requests_dir):

    try:
        driver = get_driver()  # 从线程局部变量获取 WebDriver 实例
    except (WebDriverException, NoSuchWindowException, SessionNotCreatedException,
            InvalidSessionIdException) as e:
        print(f"WebDriver error for {url}: {e}")
        # Attempt to recreate the driver in case of failure
        driver = get_driver(recreate=True)

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
    except (WebDriverException, NoSuchWindowException, SessionNotCreatedException,
            InvalidSessionIdException) as e:
        print(f"WebDriver error for {url}: {e}")
        # Attempt to recreate the driver in case of failure
        driver = get_driver(recreate=True)
        return False
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

def get_todays_file_name():
    """Generate the file name for today's date."""
    base_name = './datasets/phishing_url.'
    today = datetime.now().strftime('%Y%m%d')
    return base_name + today + '.txt'

if __name__ == '__main__':

    # to_process = [x.strip() for x in open('./datasets/phishing_url.20231221.txt').readlines()]
    # process_urls(to_process)

    while True:
        try:
            file_name = get_todays_file_name()
            print(f"Processing file: {file_name}")

            with open(file_name, 'r') as file:
                to_process = [x.strip() for x in file.readlines()]
            process_urls(to_process)

        except FileNotFoundError:
            if global_executor:
                global_executor.shutdown(wait=True)
            # Clean up WebDriver instances
            for thread in threading.enumerate():
                if hasattr(thread, "driver"):
                    thread.driver.quit()
            exit()
        finally:
            if global_executor:
                global_executor.shutdown(wait=True)
            # Clean up WebDriver instances
            for thread in threading.enumerate():
                if hasattr(thread, "driver"):
                    thread.driver.quit()
            time.sleep(3600)