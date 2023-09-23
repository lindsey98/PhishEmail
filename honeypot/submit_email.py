import requests
from xdriver.XDriver import XDriver
from xdriver.xutils.Logger import Logger
from xdriver.xutils.forms.Form import Form
from mmocr.apis import MMOCRInferencer
from xdriver.xutils.forms.SubmissionButtonLocator import SubmissionButtonLocator
from xdriver.xutils.PhishIntentionWrapper import PhishIntentionWrapper
import time
import logging
import os
import signal
from tldextract import tldextract
import subprocess
import os
from datetime import datetime, date
from mongodb import get_all

def handler(signum, frame):
    print("Signal received, shutting down...")
    driver.quit()
    exit(0)

# Register the handler for SIGTERM and SIGINT
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

def fetch_phish_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.text.split('\n')  # Split by newlines to get a list
        return data
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return []


if __name__ == '__main__':
    openphish_feed_url = "https://openphish.com/feed.txt"

    sleep_time = 5; timeout_time = 60
    XDriver.set_headless()
    XDriver.enable_proxy(port=7890) # remove this if you didnt set proxy
    driver = XDriver.boot(chrome=True)
    time.sleep(sleep_time)
    driver.set_script_timeout(timeout_time / 2)
    driver.set_page_load_timeout(timeout_time)
    Logger.set_debug_on()

    # load phishintention, mmocr, button_locator_model
    phishintention_cls = PhishIntentionWrapper()
    mmocr_model = MMOCRInferencer(det=None, rec='ABINet', device='cuda')
    button_locator_model = SubmissionButtonLocator(
        button_locator_config='/home/ruofan/git_space/MyXdriver_pub/xutils/forms/button_locator_models/config.yaml',
        button_locator_weights_path='/home/ruofan/git_space/MyXdriver_pub/xutils/forms/button_locator_models/model_final.pth')

    honeypot_dir = 'datasets/honeypot'
    os.makedirs(honeypot_dir, exist_ok=True)
    while True:
        try:
            phish_list = fetch_phish_data(openphish_feed_url)
            print(f'From openphish {len(phish_list)}')
            subprocess.run(["chmod", "+x", "./honeypot/download_github_phishing_feed.sh"])
            subprocess.run(["./honeypot/download_github_phishing_feed.sh"])
            phish_list2 = [x.strip() for x in open('./datasets/phishing-links-ACTIVE-TODAY.txt').readlines()]
            phish_list.extend(phish_list2)
            print(f'From openphish {len(phish_list2)}')
            # from our dynaphish
            dynaphish_db = get_all(filter = {"phish_prediction": 1})
            phish_list3 = [x['url'] for x in dynaphish_db]
            print(f'From dynaphish {len(phish_list3)}')
            phish_list.extend(phish_list3)

            for it, orig_url in enumerate(phish_list):
                if os.path.exists('./honeypot/submitted.txt') and orig_url in open('./honeypot/submitted.txt').read():
                    print(f"{orig_url} has been logged before")
                    continue

                # initialization
                try:
                    driver.delete_all_cookies()
                    driver.get(orig_url, allow_redirections=True)
                    time.sleep(10)  # fixme: wait until page is fully loaded
                    Logger.spit('URL={}'.format(orig_url), caller_prefix=XDriver._caller_prefix, debug=True)

                    loop = 0
                    while loop <= 2:
                        form = Form(driver, phishintention_cls, mmocr_model, button_locator_model,
                                    obfuscate=False)  # initialize form
                        loop += 1

                        if len(form._inputs) > 0:
                            # form filling
                            form.fill_all_inputs()
                            driver.save_screenshot(os.path.join(honeypot_dir, tldextract.extract(
                                orig_url).subdomain + '.' + tldextract.extract(
                                orig_url).domain + '.' + tldextract.extract(
                                orig_url).suffix + '_before.png'))
                            # button maybe at the bottom, need to decide when to scroll
                            if len(form._button_visibilities) > 0 and (not form._button_visibilities[0]):
                                driver.scroll_to_bottom()
                                form.button_reinitialize()
                            form.submit(1)  # form submission
                            driver.save_screenshot(os.path.join(honeypot_dir, tldextract.extract(
                                orig_url).subdomain + '.' + tldextract.extract(
                                orig_url).domain + '.' + tldextract.extract(
                                orig_url).suffix + '_after.png'))
                            break
                        elif len(form._buttons) > 0:
                            if len(form._button_visibilities) > 0 and (not form._button_visibilities[0]):
                                driver.scroll_to_bottom()
                                form.button_reinitialize()
                            form.submit(1)  # form submission
                        else:
                            break

                except Exception as e:
                    Logger.spit('Exception when getting the URL {}'.format(e), caller_prefix=XDriver._caller_prefix, warning=True)
                    continue

                with open('./honeypot/submitted.txt', 'a+') as f:
                    f.write(orig_url+'\n')

                if (it+1)% 50 == 0:
                    driver.quit()
                    XDriver.set_headless()
                    XDriver.enable_proxy(port=7890)
                    driver = XDriver.boot(chrome=True)
                    time.sleep(sleep_time)
                    driver.set_script_timeout(timeout_time / 2)
                    driver.set_page_load_timeout(timeout_time)
                    Logger.set_debug_on()

            time.sleep(300)  # Sleep for 5 minutes
            driver.quit()
            XDriver.set_headless()
            XDriver.enable_proxy(port=7890)
            driver = XDriver.boot(chrome=True)
            time.sleep(sleep_time)
            driver.set_script_timeout(timeout_time / 2)
            driver.set_page_load_timeout(timeout_time)
            Logger.set_debug_on()

        except KeyboardInterrupt:
            driver.quit()
            exit()


