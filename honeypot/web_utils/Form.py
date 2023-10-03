#!/usr/bin/python
import selenium.common.exceptions
from .FormElement import FormElement
from .Logger import Logger
from .Regexes import Regexes
from .utils import special_character_replacement
import logging
logging.getLogger("selenium").setLevel(logging.ERROR)

class Form():
    _caller_prefix = "Form"
    _unknown_label = "UNKNOWN"
    verifiable_inputs = {Regexes.PASSWORD, Regexes.USERID, Regexes.USERNAME}

    def __init__(self, driver,
                 phishintention_cls,
                 submission_button_locator):

        self._driver = driver
        self.PhishIntention = phishintention_cls
        self._submission_button_locator = submission_button_locator

        # find inputs
        self._inputs, self._inputs_dom, self._input_rules, self._input_visibilities, self._input_etypes, self._inputs_locations = self._get_input_elements()
        # find clickable buttons
        self._buttons, self._buttons_dom, self._button_rules, self._button_visibilities, self._buttons_locations = self._get_button_elements()

        Logger.spit("Number of inputs = {}, Number of buttons = {}".format(len(self._inputs), len(self._buttons)),
                    debug=True, caller_prefix=Form._caller_prefix)

    """
		Reinitialize when the screenshot has changed
    """

    def button_reinitialize(self):
        self._buttons, self._buttons_dom, self._button_rules, self._button_visibilities, self._buttons_locations = self._get_button_elements()
        self._inputs_locations = list(map(lambda e: self._driver.get_location(e), self._inputs))

    """
		Get all input elements and their matched filling rules
	"""

    def _get_input_elements(self):

        elements, elements_dom = self._driver.get_all_inputs()
        elements_loc = list(map(lambda e: self._driver.get_location(e), elements))

        filter_elements = []
        filter_elements_dom = []
        rules = []
        visibilities = []
        types = []
        locations = []

        for jj in range(len(elements)):
            element = elements[jj]
            element_dom = elements_dom[jj]
            element_loc = elements_loc[jj]
            visible = self._driver.check_visibility(element_loc)
            try:
                FE = FormElement(self._driver, element)
                # Step1: use regex to match HTML attributes
                matched_rule = FE._decide_rule_inputs()
                if matched_rule is not None:
                    filter_elements.append(element)
                    filter_elements_dom.append(element_dom)
                    rules.append(matched_rule)
                    visibilities.append(visible)
                    types.append(FE._etype)
                    locations.append(element_loc)
                else:
                    matched_rule = FormElement._DEFAULT_RULE
                    filter_elements.append(element)
                    filter_elements_dom.append(element_dom)
                    rules.append(matched_rule)
                    visibilities.append(visible)
                    types.append(FE._etype)
                    locations.append(element_loc)


            except Exception as e:
                Logger.spit("Error {} when trying to decide rule".format(e), warning=True,
                            caller_prefix=Form._caller_prefix)
        return filter_elements, filter_elements_dom, rules, visibilities, types, locations

    '''
		Get all button elements
	'''

    def _get_button_elements(self):

        elements, elements_dom, elements_loc = [], [], []
        # report submission button
        cv_elements, cv_elements_dom, cv_elements_loc = self.get_button_elements_cv()

        # only if submission button is not reported from CV model, use HTML as backup
        if len(cv_elements) == 0:
            elements, elements_dom = self._driver.get_all_buttons()
            elements_loc = list(map(lambda e: self._driver.get_location(e), elements))

        for it, b in enumerate(cv_elements):
            if b not in elements:
                elements.append(b)
                elements_dom.append(cv_elements_dom[it])
                elements_loc.append(cv_elements_loc[it])

        filter_elements = []
        filter_elements_dom = []
        rules = []
        visibilities = []
        locations = []

        for jj in range(len(elements)):
            element = elements[jj]
            element_dom = elements_dom[jj]
            element_loc = elements_loc[jj]
            visible = self._driver.check_visibility(element_loc)
            try:
                if visible:
                    # this step is only to check it is NOT a "register" button
                    FE = FormElement(self._driver, element)
                    matched_rule_relaxed, matched_rule_strict = FE._decide_rule_buttons()

                    if matched_rule_strict:
                        filter_elements.insert(0, element)
                        filter_elements_dom.insert(0, element_dom)
                        rules.insert(0, matched_rule_strict)
                        visibilities.insert(0, visible)
                        locations.insert(0, element_loc)
                    elif matched_rule_relaxed:
                        filter_elements.append(element)
                        filter_elements_dom.append(element_dom)
                        rules.append(matched_rule_relaxed)
                        visibilities.append(visible)
                        locations.append(element_loc)

            except Exception as e:
                Logger.spit("Error {} when trying to decide rule".format(e), warning=True,
                            caller_prefix=Form._caller_prefix)

        return filter_elements, filter_elements_dom, rules, visibilities, locations

    """
		Use CV method to get all buttons
	"""

    def get_button_elements_cv(self):
        pred_boxes = self._submission_button_locator.return_submit_button(self._driver.get_screenshot_encoding())
        if pred_boxes is None:
            return [], [], []
        pred_boxes = pred_boxes.tolist()
        elements, elements_dom, elements_loc_raw = self._driver.get_all_elements_from_coordinate_list(pred_boxes)
        elements_loc_cross_check = list(map(lambda e: self._driver.get_location(e), elements))

        filtered_elements, filtered_elements_dom, filtered_elements_loc = [], [], []
        for jj in range(len(elements)):
            xcross1, ycross1, xcross2, ycross2 = elements_loc_cross_check[jj]
            if xcross2 - xcross1 <= 0 or ycross2 - ycross1 <= 0:
                continue
            else:
                filtered_elements.append(elements[jj])
                filtered_elements_dom.append(elements_dom[jj])
                filtered_elements_loc.append(elements_loc_raw[jj])

        return filtered_elements, filtered_elements_dom, filtered_elements_loc

    """
		Fill in inputs with specified rule
	"""

    def fill_all_inputs(self):
        filled_values = []
        for ct, element in enumerate(self._inputs):
            FE = FormElement(self._driver, element)
            etext = FE._text
            try:
                value = FE.fill(rule=self._input_rules[ct])
            except selenium.common.exceptions.StaleElementReferenceException as e:
                Logger.spit("Error {} when trying to fill in element".format(e), warning=True,
                            caller_prefix=Form._caller_prefix)
                continue
            if self._input_rules[ct] in Form.verifiable_inputs:
                filled_values.append(value)
            Logger.spit("Notice: filled value {} in input field {}".format(value, etext), debug=True,
                        caller_prefix=Form._caller_prefix)
        return filled_values

    """
		Click submit buttons
	"""
    def submit(self, prev_num_windows, cached_button_element=None):

        prev_src = self._driver.get_page_text()
        prev_html = self._driver.rendered_source()

        if len(self._buttons) == 0:
            return False
        if cached_button_element is not None:
            element = cached_button_element
        else:
            element = self._buttons[0]  # fixme: I only click the most relevant button for conservativeness

        etext = self._driver.get_text(element)
        if (not etext) or len(etext)==0:
            etext = self._driver.get_attribute(element, "value")
        Logger.spit("Try clicking button {} ...".format(etext), debug=True,
                    caller_prefix=Form._caller_prefix)
        try:
            self._driver.move_to_element(element)
            click_success = self._driver.click(element)
        except selenium.common.exceptions.StaleElementReferenceException:
            return False

        if click_success:
            Logger.spit("Clicking button {} successfully".format(etext), debug=True,
                        caller_prefix=Form._caller_prefix)
            current_src = self._driver.get_page_text()
            current_html = self._driver.rendered_source()
            current_num_windows = len(self._driver.window_handles)

            # new window is open
            if current_num_windows != prev_num_windows:
                Logger.spit("Clicking the button opens a new window", debug=True,
                            caller_prefix=Form._caller_prefix)
                return True
            # change in page source
            elif special_character_replacement(prev_src.lower()).replace(' ', '') != special_character_replacement(
                    current_src.lower()).replace(' ', '') or \
                    special_character_replacement(prev_html.lower()).replace(' ', '') != special_character_replacement(
                current_html.lower()).replace(' ', ''):
                Logger.spit("Detect changes in webpage source", debug=True,
                            caller_prefix=Form._caller_prefix)
                return True
            # NO change in page source
            else:
                Logger.spit("No change in webpage", warning=True,
                            caller_prefix=Form._caller_prefix)
                return False
        else:
            Logger.spit("No change in webpage", warning=True,
                        caller_prefix=Form._caller_prefix)
            return False
