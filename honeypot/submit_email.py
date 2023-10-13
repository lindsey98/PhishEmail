from honeypot.web_utils.Logger import Logger
from honeypot.web_utils.CustomDriver import CustomWebDriver
from honeypot.web_utils.Form import Form
from honeypot.web_utils.SubmissionButtonLocator import SubmissionButtonLocator
from honeypot.web_utils.PhishIntentionWrapper import PhishIntentionWrapper
from honeypot.web_utils.utils import *
import time
import signal
from tldextract import tldextract
import subprocess
import os
from honeypot.mongodb import get_all
import logging
# Suppress warning from urllib3
logging.getLogger("urllib3").setLevel(logging.ERROR)
# Suppress debug logs from selenium
logging.getLogger("selenium").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger('faker').setLevel(logging.ERROR)

def handler(signum, frame):
    print("Signal received, shutting down...")
    driver.quit()
    exit(0)

# Register the handler for SIGTERM and SIGINT
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

if __name__ == '__main__':
    openphish_feed_url = "https://openphish.com/feed.txt"

    timeout_time = 60
    proxy_url = "127.0.0.1:7890"
    driver = CustomWebDriver.boot(proxy_server=proxy_url)  # Using the proxy_url variable
    time.sleep(3)
    driver.set_script_timeout(timeout_time / 2)
    driver.set_page_load_timeout(timeout_time)
    Logger.set_debug_on()

    # load phishintention, mmocr, button_locator_model
    phishintention_cls = PhishIntentionWrapper()
    button_locator_model = SubmissionButtonLocator(
        button_locator_config='/home/ruofan/git_space/MyXdriver_pub/xutils/forms/button_locator_models/config.yaml',
        button_locator_weights_path='/home/ruofan/git_space/MyXdriver_pub/xutils/forms/button_locator_models/model_final.pth')

    honeypot_dir = './datasets/honeypot'
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
                    driver.get(orig_url)
                    time.sleep(3)  # fixme: wait until page is fully loaded
                    Logger.spit('URL={}'.format(orig_url), caller_prefix=CustomWebDriver._caller_prefix, debug=True)

                    # CRP transition first
                    is_crp_page = phishintention_cls.perform_crp_classification(driver)
                    Logger.spit('Is it a CRP page? {}'.format(is_crp_page), caller_prefix=CustomWebDriver._caller_prefix, debug=True)
                    if not is_crp_page:
                        phishintention_cls.perform_crp_transition(driver)
                    driver.scroll_to_top()

                    loop = 0
                    while loop <= 2:
                        form = Form(driver=driver, phishintention_cls=phishintention_cls, submission_button_locator=button_locator_model)  # initialize form
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
                    Logger.spit('Exception when getting the URL {}'.format(e), caller_prefix=CustomWebDriver._caller_prefix, warning=True)
                    continue

                with open('./honeypot/submitted.txt', 'a+') as f:
                    f.write(orig_url+'\n')

                if (it+1)% 50 == 0:
                    driver.quit()
                    driver = CustomWebDriver.boot(proxy_server=proxy_url)
                    time.sleep(3)
                    driver.set_script_timeout(timeout_time / 2)
                    driver.set_page_load_timeout(timeout_time)
                    Logger.set_debug_on()

            time.sleep(3600)  # Sleep for 1 hour
            driver.quit()
            driver = CustomWebDriver.boot(proxy_server=proxy_url)
            time.sleep(3)
            driver.set_script_timeout(timeout_time / 2)
            driver.set_page_load_timeout(timeout_time)
            Logger.set_debug_on()

        except KeyboardInterrupt:
            driver.quit()
            exit()


