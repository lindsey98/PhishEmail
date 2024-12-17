from tldextract import extract
from collections import Counter
from bs4 import BeautifulSoup as bs
import numpy as np
import re

'''
This file contains a list of functions to extract the HTML features from the HTML section of each email

'''

class HTMLFeatures(object):
    """
        This class defines and implements HTML features.
    """
    def __init__(self, soup):
        self.soup = soup
        self.sensitive_word_list = ['user', 'pass', 'password', 'credit', 'card', 'login', 'name', 'account', 'id',
                                    'submit', 'alert', 'warning', 'sign', 'access']
        
    def extract_text(self):
        return self.soup.text

    def sensitive_words(self):
        # return ratio of match against sensitive word list
        text = self.extract_text()
        return sum(map(lambda x: x in text.lower(), self.sensitive_word_list))/len(self.sensitive_word_list)
    
    def text_body(self):        
        return self.soup.text

    def len_text(self):
        return len(self.soup.text)

    def extract_css(self):
        # get the length of css

        #type 1: <style> tag
        css_internal = [styletag.string for styletag in self.soup.findAll('style') if styletag.string]

        #type 2: inline css
        css_inline = []
        for tag in self.soup.findAll():
            if ('style' in tag) and (tag.get('style') is not None):
                css_inline.append(tag['style'])       
        len_css = len('#'.join(css_internal)) + len('#'.join(css_inline))
        if_backward_rendering = any(map(lambda x: 'direction: rtl' in str(x), css_inline + css_internal))

        return len_css, if_backward_rendering


    def extract_js(self):
        # JS not supposed to be use in here. This function checks on that.
        # return 0 if len(self.soup.find_all('script')) == 0 else 1

        return True if len(self.soup.find_all('script')) != 0 else False

    def extract_dom(self):
        tag_list, depth_list, leaf_node_list, leaf_node_depth_list = [], [], [], []
        
        for tag in self.soup.find_all():
            tag_list.append(tag.name)
            depth_list.append(len(list(tag.parents)))
            num_des = len(list(tag.descendants))
            #need to consider both num_des == 0 and 1 cases due to BeautifulSoup implementation
            if num_des == 0:
                leaf_node_list.append(tag.name)
                leaf_node_depth_list.append(len(list(tag.parents)))
            if num_des == 1:
                if not all(substring in str(tag.contents[0]) for substring in ['<', '>', '</']):
                    leaf_node_list.append(tag.name)
                    leaf_node_depth_list.append(len(list(tag.parents)))

        popular_tags = Counter(tag_list).most_common(5)
        most_popular_tags = [pair[0] for pair in popular_tags]
        most_popular_tags_freq = [pair[1]/len(tag_list) for pair in popular_tags]
        dom_depth = np.max(depth_list) if len(depth_list) != 0 else 0
        dom_leaf_node_count = len(leaf_node_list)
        dom_leaf_node_type = len(list(set(leaf_node_list)))
        dom_leaf_node_mean = np.mean(leaf_node_depth_list)
        dom_leaf_node_std = np.std(leaf_node_depth_list)

        return most_popular_tags, most_popular_tags_freq, dom_depth, dom_leaf_node_count, dom_leaf_node_type, dom_leaf_node_mean, dom_leaf_node_std
         

    def diverse_links(self):
        email_addr = []
        # check on number of links and diversity of links inside HTML        
        a_tags = self.soup.find_all('a')
        all_links = [x.attrs['href'] for x in a_tags if 'href' in x.attrs]
        all_links.extend([x.attrs['data-saferedirecturl'] for x in a_tags if 'data-saferedirecturl' in x.attrs])
        
        for x in a_tags:
            if len(x.attrs) == 0:
                #print('No Attrs:', x.text)
                if 'http' in x.text:
                    refined_text = x.text.replace('\\\'','').replace(' ', '').replace('\n', '').replace('\"','').replace('\\n', '')
                    all_links.append(refined_text)
                    #if not refined_text.startswith('http'):
                        #print(refined_text)                        
            list_email = re.findall(r"(\w.*?@\w.*\w)", x.text, flags=re.DOTALL)            
            email_addr.extend(list_email)
            #print(list_email)
        
        all_links2 = [link.replace('\\\'','').replace(' ', '').replace('\n', '').replace('\"','').replace('\\n', '') for link in all_links]
        #url_links = [link for link in all_links2 if (link.startswith('http://') or link.startswith('https://'))]     
        list_any_url = re.findall(r'http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[\/\-?=_:;@.&+]|[!*,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', self.soup.text)
        set_url = set(list_any_url)
        #print('Mail:' , email_addr)
        domains = [extract(link).domain for link in all_links2]
        domains.extend([extract(addr).domain for addr in email_addr])
        email_addr.extend([link.replace('mailto:', '') for link in all_links2 if link.startswith('mailto:') and '@' in link])
        unique_domain_ratio = len(list(set(domains)))/len(domains) if len(domains) != 0 else 0        
        return len(all_links2), unique_domain_ratio, list(set_url), email_addr 

def extract_html_features(one_soup, dict_html_feature_use):
    sensitive_word_ratio = 0.0
    len_text = 0
    len_css = 0
    if_backward_rendering = False
    if_js = False
#     most_popular_tags = []
#     most_popular_tags_freq = []
    dom_depth = 0
    dom_leaf_node_count = 0
    dom_leaf_node_type = 0
    dom_leaf_node_mean = 0
    dom_leaf_node_std = 0.0
    num_links = 0
    unique_domain_ratio = 0.0
  
    html_class = HTMLFeatures(one_soup)       

    sensitive_word_ratio = html_class.sensitive_words()
    len_text = html_class.len_text()

    if dict_html_feature_use['css_features']:
        css_features = html_class.extract_css()
        len_css = css_features[0]
        if_backward_rendering = css_features[1]
    else:
        len_css = 0
        if_backward_rendering = False

    if dict_html_feature_use['js_features']:
        if_js = html_class.extract_js()
    else:
        if_js = False

    if dict_html_feature_use['dom_features']:
        dom_features = html_class.extract_dom()
    else:
        dom_features = ['', 0, 0, 0, 0, 0, 0]

#     most_popular_tags.append(dom_features[0])
#     most_popular_tags_freq = dom_features[1])
    dom_depth = dom_features[2]
    dom_leaf_node_count = dom_features[3]
    dom_leaf_node_type = dom_features[4]
    dom_leaf_node_mean = dom_features[5]
    dom_leaf_node_std = dom_features[6]

    if dict_html_feature_use['link_features']:
        links_features = html_class.diverse_links()
    else:
        links_features = [0, 1.0, [], []]

    num_links = links_features[0]
    unique_domain_ratio = links_features[1]

    list_html_feat = []
    list_html_feat.append(sensitive_word_ratio)
    list_html_feat.append(len_text)
    list_html_feat.append(len_css)
    list_html_feat.append(if_backward_rendering)
    list_html_feat.append(if_js)
    list_html_feat.append(dom_depth)
    list_html_feat.append(dom_leaf_node_count)
    list_html_feat.append(dom_leaf_node_type)
    list_html_feat.append(dom_leaf_node_mean)
    list_html_feat.append(dom_leaf_node_std)
    list_html_feat.append(num_links)
    list_html_feat.append(unique_domain_ratio)    
    
    return list_html_feat



def html_feature_extraction(html_data,dict_html_feature_use):

    html_body = html_data
    soup = bs(html_body, 'lxml') # Parse at once.              
    list_html_feat = extract_html_features(soup, dict_html_feature_use) 
    return list_html_feat    
    
def extract_features(html_data):
    
    # Set of features to use
    dict_feature_use = {}
    dict_feature_use['css_features'] = True
    dict_feature_use['js_features'] = True
    dict_feature_use['dom_features'] = True
    dict_feature_use['link_features'] = True      
    dict_feature_use['charset'] = True
    dict_feature_use['messageid'] = True
    dict_feature_use['mailaddr'] = True
    dict_feature_use['section_boundary'] = True
    # --
    list_html = html_feature_extraction(html_data, dict_feature_use)
    
    return list_html

def get_column_list(dict_html_feature_use):
    list_column_header = []    
       
    list_column_header.append('sensitive_word_ratio')    
    list_column_header.append('len_html_text')
    
    if dict_html_feature_use['css_features']:
        list_column_header.append('len_html_css')
        list_column_header.append('if_backward_rendering')        
    
    if dict_html_feature_use['js_features']:
        list_column_header.append('if_js')
 
    if dict_html_feature_use['dom_features']:
        list_column_header.append('dom_depth')
        list_column_header.append('dom_leaf_node_count')
        list_column_header.append('dom_leaf_node_type')
        list_column_header.append('dom_leaf_node_mean')
        list_column_header.append('dom_leaf_node_std')
        

    if dict_html_feature_use['link_features']:
        list_column_header.append('num_links')
        list_column_header.append('unique_domain_ratio')
    
    return list_column_header

