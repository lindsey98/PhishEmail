from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import *
from seleniumwire.webdriver import ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from .utils import lower, replace_nbsp
import json
import time
import re
import numpy as np
import logging

# Suppress warning from urllib3
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Suppress debug logs from selenium
logging.getLogger("selenium").setLevel(logging.ERROR)

class CustomWebDriver(webdriver.Chrome):
    _caller_prefix = "CustomWebDriver"
    _MAX_RETRIES = 3
    _last_url = 'https://google.com'
    _forbidden_suffixes = r"\.(mp3|wav|wma|ogg|mkv|zip|tar|xz|rar|z|deb|bin|iso|csv|tsv|dat|txt|css|log|sql|xml|sql|mdb|apk|bat|bin|exe|jar|wsf|fnt|fon|otf|ttf|ai|bmp|gif|ico|jp(e)?g|png|ps|psd|svg|tif|tiff|cer|rss|key|odp|pps|ppt|pptx|c|class|cpp|cs|h|java|sh|swift|vb|odf|xlr|xls|xlsx|bak|cab|cfg|cpl|cur|dll|dmp|drv|icns|ini|lnk|msi|sys|tmp|3g2|3gp|avi|flv|h264|m4v|mov|mp4|mp(e)?g|rm|swf|vob|wmv|doc(x)?|odt|rtf|tex|txt|wks|wps|wpd)$"

    def __init__(self, proxy_server=None, *args, **kwargs):
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--headless")
        prefs = {
            "download.default_directory": './trash',
            "download.prompt_for_download": False,  # To disable the download prompt and download automatically
            "download_restrictions": 3  # Attempt to restrict all downloads
        }
        chrome_options.add_experimental_option("prefs", prefs)
        self._proxy_server = proxy_server
        if proxy_server:
            chrome_options.add_argument(f"--proxy-server={proxy_server}")

        chrome_caps = DesiredCapabilities.CHROME
        chrome_caps['acceptSslCerts'] = True
        chrome_caps['acceptInsecureCerts'] = True

        super().__init__(
			             "./datasets/chromedriver",
                         chrome_options=chrome_options,
                         desired_capabilities=chrome_caps)

        self._RETRIES = kwargs.get("retries", {})
        self._REFS = kwargs.get("refs", {})
        self._recoverable_crashes = ["chrome not reachable", "page crash",
                                     "cannot determine loading status",
                                     "Message: unknown error"]
        with open('./lib/web_utils/scripts.js', 'r') as f:
            js_content = f.read()
        self.add_script(js_content)

    def send(self, cmd, params={}):
        resource = "/session/%s/chromium/send_command_and_get_result" % self.session_id
        url = self.command_executor._url + resource
        body = json.dumps({'cmd': cmd, 'params': params})
        response = self.command_executor._request('POST', url, body)

    def add_script(self, script):
        self.send(cmd="Page.addScriptToEvaluateOnNewDocument", params={"source": script})

    @classmethod
    def boot(cls, **kwargs):
        return cls(**kwargs)

    def reboot(self):
        self.quit()
        new_instance = CustomWebDriver.boot(proxy_server=self._proxy_server)
        time.sleep(3)
        self.__dict__.update(new_instance.__dict__)

    '''Exception handling'''
    @staticmethod
    def stringify_exception(e, strip=False):
        exception_str = ""
        try:
            exception_str = str(e)
            if strip:  # Only return first line. Useful for webdriver exceptions
                exception_str = exception_str.split("\n")[0]
        except Exception as e:
            exception_str = "Could not stringify exception"
        return exception_str

    def enter_retry(self, method, max_retries=10):
        retries, max_retries = self._RETRIES.get(method, [0, max_retries])  # Necessary so nested `_invokes` of the same method won't reset the retry counter
        self._RETRIES[method] = [retries, max_retries]
        if retries <= max_retries:
            return True
        return False

    def exit_retry(self, method):
        self._RETRIES.pop(method, None)  # Only need to pop the method name

    def _invoke_exception_handler(self, handler, *args, **kwargs):
        try:
            return handler(*args, **kwargs)
        except UnexpectedAlertPresentException:
            self._invoke_exception_handler(self._UnexpectedAlertPresentException_handler)
        except NoAlertPresentException:  # This was raised during the _UnexpectedAlertPresentException_handler call.
            self.execute_script("window.alert = null;")
        except TimeoutException:
            return False

    def _UnexpectedAlertPresentException_handler(self):
        alert = self.switch_to.alert
        alert.dismiss()
        self.execute_script("window.alert = null;")  # Try to prevent the page from popping any more alerts
        self.switch_to.default_content()

    def _StaleElementReference_handler(self, webelement):
        element_ref = id(webelement)
        if element_ref not in self._REFS:
            return False
        method, args, kwargs = self._REFS[element_ref]
        kwargs["timeout"] = 3  # add some max timeout
        new_element = method(*args, **kwargs)  # refetch element
        if not new_element:  # The element is not in the DOM, it's not just stale
            return False

        webelement.__dict__.update(new_element.__dict__)  # Transparently update old webelement's reference
        return True

    def _TimeoutException_handler(self):
        print("Handling browser crash. Will try to restore webdriver.")
        self.reboot()  # We don't want to clear the _REFS, delete the profile or the proxy config
        return True

    def _invoke(self, method, *args, **kwargs):
        original_kwargs = dict(kwargs)  # In case we re-_invoke it, we need the original kwargs
        ex = None
        ret_val = None # used to record whether the run is successful or not

        try:
            if kwargs.pop("retry", True):  # By default, retry all methods if possible, otherwise explicitly requested
                self.enter_retry(method, max_retries=kwargs.pop("max_retries", self._MAX_RETRIES))
            web_element = kwargs.pop("webelement", None)
            ret_val = method(*args, **kwargs)
            if method == super(CustomWebDriver, self).get:
                ret_val = True  # Need to explicitly set the ret value for WebDriver's get
            self.exit_retry(method)  # The operation completed, no need to keep the retry counter
            return ret_val
        except UnexpectedAlertPresentException as ex:
            self._invoke_exception_handler(self._UnexpectedAlertPresentException_handler)
            if method == super(CustomWebDriver, self).get:  # Nothing more to do for a `get`
                ret_val = True
        except (InvalidSwitchToTargetException, NoSuchFrameException, NoSuchWindowException) as ex:
            if len(self.window_handles) == 0:  # If no windows remain for some reason, raise it
                raise
            self.switch_to.default_content()  # Return to the default handle
            ret_val = False
        except (InvalidSelectorException,
                InvalidElementStateException,
                ElementNotSelectableException,
                ElementNotVisibleException,
                MoveTargetOutOfBoundsException) as ex:
            ret_val = False
        except (StaleElementReferenceException, NoSuchElementException) as ex:
            # Check _REFS for given WebElement.
            if not self._invoke_exception_handler(self._StaleElementReference_handler, web_element):
                raise
            ret_val = False
        except (TimeoutException, WebDriverException, ErrorInResponseException, RemoteDriverServerException,
                InvalidCookieDomainException, UnableToSetCookieException, ImeNotAvailableException,
                ImeActivationFailedException) as ex:
            str_ex = self.stringify_exception(ex)
            if ex is TimeoutException or any([crash in str_ex for crash in self._recoverable_crashes]):
                # Reboot browser, maintain state and retry the operation
                if not self._invoke_exception_handler(self._TimeoutException_handler):
                    raise
                if method != super(CustomWebDriver, self).get:
                    self.get(self._last_url)  # If it was `get`, it will be retried later on. For anything else, we need to manually go back to the last known URL
                ret_val = False
            else:
                raise

        retries, max_retries = self._RETRIES.get(method, [None, None])
        # If we are not in retry mode OR if the retries have exceeded the threshold, either return a default value (if set) or raise the exception to the caller
        if method not in self._RETRIES or retries >= max_retries:
            self.exit_retry(method)
            if ret_val != None:  # If a return value has been set, return it instead of raising the exception
                return ret_val
            raise  # These are considered fatal

        # About to re-invoke method. Increment retry counter
        self._RETRIES[method][0] += 1
        print("Retrying for the {}th time...".format(self._RETRIES[method][0]))
        return self._invoke(method, *args, **original_kwargs)

    '''Webpage specific'''
    def get_screenshot_encoding(self):
        return self._invoke(super(CustomWebDriver, self).get_screenshot_as_base64)

    def get_page_text(self):
        try:
            body = self.find_element_by_tag_name('html').text
            return body
        except:
            return ''

    def rendered_source(self):
        try:
            return self._invoke(self.execute_script, "return document.getElementsByTagName('html')[0].innerHTML;")
        except JavascriptException:
            return self._invoke(super(CustomWebDriver, self).page_source)

    def obfuscate_page(self):
        self._invoke(self.execute_script, """
                  var script = document.createElement('script');
                  script.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/0.5.0-beta4/html2canvas.min.js';
                  document.head.appendChild(script);
                """)
        time.sleep(1)
        return self._invoke(self.execute_script, """obfuscate_button();""")

    def current_url(self):
        return self._invoke(self._current_url)

    def _current_url(self):
        return super(CustomWebDriver, self).current_url

    def switch_to_window(self, window_handle):
        return self._invoke(self._switch_to_window, window_handle)

    def _switch_to_window(self, window_handle):
        # Default `switch_to.window` hangs in case there is an open alert, so we need a dummy op to trigger the alert handling
        self.execute_script("return 2;")
        self.switch_to.window(window_handle)
        # Inject the JavaScript into the new window
        with open('./lib/web_utils/scripts.js', 'r') as f:
            js_content = f.read()
        self.add_script(js_content)
        return True

    # Scroll to top of the page
    def scroll_to_top(self):
        try:
            self.execute_script("window.scrollTo(0, 0);")
            return True
        except Exception as e:
            return False

    # Scroll to bottom of the page
    def scroll_to_bottom(self):
        try:
            self.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            return True
        except Exception as e:
            return False

    '''Find elements'''
    def _webelement_find_element_by(self, element, by=By.ID, value=None):
        return element.find_element(by=by, value=value)

    def _webelement_find_elements_by(self, element, by=By.ID, value=None, *args, **kwargs):
        return element.find_elements(by=by, value=value, *args, **kwargs)

    def find_element(self, by=By.ID, value=None, timeout=0, visible=False, webelement=None):
        if timeout == 0 and visible is False:
            ret = self._invoke(super(CustomWebDriver, self).find_element, by=by, value=value) if webelement is None \
                else self._invoke(self._webelement_find_element_by, webelement, by=by, value=value)
        else:
            try:
                condition = EC.presence_of_element_located if not visible else EC.visibility_of_element_located
                ret = WebDriverWait(self, timeout).until(condition((by, value)))
            except TimeoutException:
                return None
        ref = id(ret)
        if ret:
            self._REFS[ref] = (self.find_element, (), {"by": by, "value": value, "timeout": timeout, "visible": visible, "webelement": webelement})
        return ret

    def find_elements(self, by=By.ID, value=None, timeout=0, visible=False, webelement=None, *args, **kwargs):
        if timeout > 0:
            self.find_element(by=by, value=value, timeout=timeout, visible=visible, webelement=webelement, *args, **kwargs)
        ret_elements = self._invoke(super(CustomWebDriver, self).find_elements, by=by, value=value, *args,
                                    **kwargs) if webelement is None \
            else self._invoke(self._webelement_find_elements_by, webelement, by=by, value=value, webelement=webelement,
                              *args, **kwargs)
        if ret_elements:
            to_remove = set()
            for el in ret_elements:
                try:
                    el_dompath = self.get_dompath(el)
                except StaleElementReferenceException:
                    to_remove.add(el)
                    continue
                # Make sure the returned elements are robust against StaleElementReferenceExceptions by simulating a `find_element_by_xpath`
                ref = id(el)
                if ref not in self._REFS:  # If not already previously fetched
                    self._REFS[ref] = (
                    self.find_element, (), {"by": By.XPATH, "value": el_dompath, "timeout": 3, "visible": False})

            for el in to_remove:
                ret_elements.remove(el)

        return ret_elements

    def find_elements_by_xpath(self, xpath_, timeout=0, visible=False, webelement=None, *args, **kwargs):
        return self.find_elements(By.XPATH, value=xpath_, timeout=timeout, visible=visible, webelement=webelement,
                                  *args, **kwargs)

    def find_element_by_location(self, point_x, point_y):
        return self._invoke(self.execute_script, 'return document.elementFromPoint(%r, %r);' % (point_x, point_y))

    def get_all_elements_from_coordinate_list(self, coordinate_list):
        ret, returned_coords = self._invoke(self.execute_script,
                                            "return get_all_elements_from_coordinate_list(arguments[0]);",
                                            coordinate_list)
        interested_elements = []
        interested_elements_dom = []

        for this_element in ret:
            ele, ele_dompath = this_element
            interested_elements.append(ele)
            interested_elements_dom.append(ele_dompath)
            self._REFS[id(ele)] = (
                self.find_element, (), {"by": By.XPATH, "value": ele_dompath, "timeout": 3, "visible": False})

        return interested_elements, interested_elements_dom, returned_coords

    def get_all_inputs(self):
        ret = self._invoke(self.execute_script, "return get_all_inputs();")
        interested_inputs = []
        interested_inputs_dom = []

        for input_ele in ret:
            input, input_dompath = input_ele
            interested_inputs.append(input)
            interested_inputs_dom.append(input_dompath)
            self._REFS[id(input)] = (
            self.find_element, (), {"by": By.XPATH, "value": input_dompath, "timeout": 3, "visible": False})

        return interested_inputs, interested_inputs_dom

    def get_all_buttons(self):
        ret = self._invoke(self.execute_script, "return get_all_buttons();")
        interested_buttons = []
        interested_buttons_dom = []

        for button_ele in ret:
            button, button_dompath = button_ele
            interested_buttons.append(button)
            interested_buttons_dom.append(button_dompath)
            self._REFS[id(button)] = (self.find_element, (), {"by": By.XPATH, "value": button_dompath, "timeout": 3, "visible": False})

        return interested_buttons, interested_buttons_dom


    def get_all_links_orig(self):
        ret = self._invoke(self.execute_script, "return get_all_links();")
        interested_links = []
        for link_ele in ret:
            link, link_dompath, link_source = link_ele
            if re.search(CustomWebDriver._forbidden_suffixes, link_source, re.IGNORECASE):
                continue
            if link not in interested_links:
                interested_links.append([link, link_source])
                self._REFS[id(link)] = (self.find_element, (), {"by": By.XPATH, "value": link_dompath, "timeout": 3, "visible": False})

        return interested_links

    def get_all_links(self):
        ret = self._invoke(self.execute_script, "return get_all_links();")
        interested_links = []
        link_doms = []
        link_sources = []
        for link_ele in ret:
            link, link_dompath, link_source = link_ele
            interested_links.append(link)
            link_doms.append(link_dompath)
            link_sources.append(link_source)
            self._REFS[id(link)] = (self.find_element, (), {"by": By.XPATH, "value": link_dompath, "timeout": 3, "visible": False})

        return interested_links, link_doms, link_sources

    def get_all_clickable_images(self):
        ret = self._invoke(self.execute_script, "return get_all_clickable_imgs();")
        images = []
        images_dom = []

        for ele in ret:
            img, dompath = ele
            images.append(img)
            images_dom.append(dompath)
            self._REFS[id(img)] = (self.find_element, (), {"by": By.XPATH, "value": dompath, "timeout": 3, "visible": False})

        return images, images_dom

    def get_all_clickable_leaf_nodes(self):
        all_leaf_node_xpath = ["//span[not(*)]", "//div[not(*)]",
                               "//p[not(*)]", "//i[not(*)]"]
        leaf_elements_pre = []
        leaf_elements = []
        leaf_elements_dom = []

        for path in all_leaf_node_xpath:
            elements = self.find_elements_by_xpath(path)
            if elements:
                leaf_elements_pre.extend(elements)

        for ele in leaf_elements_pre:
            try:
                dompath = self.get_dompath(ele)
                leaf_elements_dom.append(dompath)
                leaf_elements.append(ele)
                self._REFS[id(ele)] = (self.find_element, (), {"by": By.XPATH, "value": dompath, "timeout": 3, "visible": False})
            except:
                continue

        return leaf_elements, leaf_elements_dom

    def get_all_clickable_elements(self):
        btns, btns_dom = self.get_all_buttons()
        links, links_dom, _ = self.get_all_links()
        images, images_dom = self.get_all_clickable_images()
        leaf_elements, leaf_elements_dom = self.get_all_clickable_leaf_nodes()

        return (btns, btns_dom), (links, links_dom), \
               (images, images_dom), (leaf_elements, leaf_elements_dom)

    def _get_elements_xpath_contains(self, patterns, tag=None, role=None, is_input=False, is_free_text=False):
        property_list = ["text()", "@class", "@title", "@value", "@label", "@aria-label"]

        # For regular elements
        tag_matching_regex = "//%s" % (tag if tag else "*")
        pattern_matching_regex = " or ".join(
            ["starts-with(normalize-space(%s), '%s')" % (lower(replace_nbsp(property)), patterns.lower()) for property
             in property_list])
        rule1 = tag_matching_regex + "[" + pattern_matching_regex + "]"

        # For elements with roles
        role_matching_regex = "//*[@role='%s']" % (role if role else "*")
        pattern_matching_regex = "[%s]" % ("starts-with(normalize-space(.), '%s')" % patterns)
        rule2 = role_matching_regex + pattern_matching_regex

        # For input elements with specific types
        if is_input:
            rule1 = "//input[@type='submit' or @type='button'][" + pattern_matching_regex + "]"

        # For free text elements
        if is_free_text:
            xpath_base = "//*[starts-with(normalize-space(%s), '%s')]" % (
            lower(replace_nbsp("text()")), patterns.lower())
            rule1 = '%s[not(self::script)][not(.%s)]' % (xpath_base, xpath_base)

        return [rule1, rule2]

    def get_clickable_elements_contains(self, patterns):

        button_xpaths = self._get_elements_xpath_contains(patterns, tag='button', role='button')
        link_xpaths = self._get_elements_xpath_contains(patterns, tag='a', role='link')
        input_xpath = self._get_elements_xpath_contains(patterns, is_input=True)
        free_text_xpath = self._get_elements_xpath_contains(patterns, is_free_text=True)

        element_list = []
        xpath_list = button_xpaths + link_xpaths + input_xpath + free_text_xpath
        for path in xpath_list:
            elements = self.find_elements_by_xpath(path)
            if elements:
                element_list.extend(elements)

        return element_list

    def get_all_visible_username_password_inputs(self):
        ret_password, ret_username = self._invoke(self.execute_script,
                                                  "return get_all_visible_password_username_inputs();")
        return ret_password, ret_username

    '''Element specific'''
    def get_dompath(self, element):
        try:
            dompath = self._invoke(self.execute_script, "return get_dompath(arguments[0]).toLowerCase();", element,
                                   webelement=element)
        except Exception as e:
            raise  # Debug debug Debug
        return "//html%s" % "/".join(
            [part if ":" not in part else "*" for part in dompath.split("/")]) if dompath else dompath

    def check_visibility(self, element_loc):
        x1, y1, x2, y2 = element_loc
        return np.sum([1 for x in element_loc if x <= 0]) == 0 and \
               (x2 - x1) > 0 and \
               (y2 - y1) > 0

    # Check whether the given element is `required`
    def is_required(self, element):
        return self._invoke(self.execute_script, "return arguments[0].required", element, webelement=element)

    def get_element_property(self, element):
        try:
            return self._invoke(self.execute_script, "return get_element_properties(arguments[0]);", element,
                                webelement=element)
        except StaleElementReferenceException as e:
            return [''] * 10

    def get_property(self, element, eproperty):
        try:
            return self._invoke(self._get_property, element, eproperty, webelement=element)
        except StaleElementReferenceException as e:
            return ''

    def _get_property(self, element, eproperty):
        return element.get_property(eproperty)

    # Get an element's tag in lowercase
    def get_tag(self, element):
        try:
            return self._invoke(self.execute_script, "return arguments[0].tagName.toLowerCase()", element,
                                webelement=element)
        except StaleElementReferenceException as e:
            return ''

    # Get element location
    def get_location(self, element):
        try:
            loc = self._invoke(self.execute_script, 'return get_loc(arguments[0]);', element, webelement=element)
            return loc
        except StaleElementReferenceException as e:
            return [0, 0, 0, 0]

    # Get element text
    def get_text(self, element):
        try:
            text = self._invoke(self.execute_script, 'return arguments[0].innerText;', element, webelement=element)
            return text
        except StaleElementReferenceException as e:
            return ''

    # Get element's attribute
    def get_attribute(self, element, attribute):
        return self._invoke(self._get_attribute, element, attribute, webelement=element)

    def _get_attribute(self, element, attribute):
        try:
            attribute = element.get_attribute(attribute)
            return attribute
        except StaleElementReferenceException:
            return ''

    # Perform action on the element
    def move_to_element(self, element):
        return self._invoke(self._move_to_element, element, webelement=element)

    def _move_to_element(self, element):
        ActionChains(self).move_to_element(element).perform()
        return True

    def click(self, element):
        return self._invoke(self._click, element, webelement=element)

    def _click(self, element):
        ActionChains(self).move_to_element(element).click().perform()
        return True

    def send_keys(self, element, keys):
        return self._invoke(self._send_keys, element, keys, webelement=element)

    def _send_keys(self, element, keys):
        element.send_keys(keys)
        return True

    def set_value(self, element, value):
        self._invoke(self.execute_script, "arguments[0].value = '%s';" % value, element, webelement=element)
        return value

    # Mark the given element as `checked`
    def check_element(self, element):
        self._invoke(self.execute_script, "arguments[0].checked = true;", element, webelement=element)

    # Select the given index, when given a `Select` webelement
    def set_selected_index(self, element, idx):
        self._invoke(self.execute_script, "arguments[0].selectedIndex = '%s';" % idx, element, webelement=element)


