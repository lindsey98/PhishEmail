import json
from tqdm import tqdm
import os
from tldextract import tldextract
import re
import difflib
import pandas as pd
import numpy as np
ext = tldextract.TLDExtract(cache_dir='./lib/reference_db')
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'
# Function to validate the domain name


def is_valid_domain(s):
    # Regex to check valid Domain Name
    pattern = r"^(?!-)([A-Za-z0-9-]{1,63}\.)*[A-Za-z0-9-]{1,63}\.[A-Za-z]{2,24}(?<!-)$"

    # Return true if the string matches the regex
    return bool(re.match(pattern, s))


def is_english_or_allowed_symbols(s):
    """ Check if the string contains only English letters, numerals, and specific symbols """
    pattern = re.compile(r"^[a-zA-Z0-9 \-,*]+$")  # Define allowed characters
    return bool(pattern.match(s))

if __name__ == "__main__":
    '''Conventional way is to use clearbit API to find the official emails given a domain'''
    # clearbit_api = open('./datasets/clearbit_key.txt').read()
    #
    # gsearch_api, gsearch_id = [x.strip() for x in open('./datasets/google_api_key.txt').readlines()]
    # gsearch_engine = GoogleSearch(SEARCH_ENGINE_ID=gsearch_id, SEARCH_ENGINE_API=gsearch_api)
    # proxies = {
    #     "http": "http://127.0.0.1:7890",
    #     "https": "http://127.0.0.1:7890",
    # }
    #
    # with open('./lib/phishpedia/models/domain_map.pkl', "rb") as handle:
    #     domain_map = pickle.load(handle)
    #
    # # Load the existing data from JSON file
    # try:
    #     with open('./datasets/company_database.json', 'r') as json_file:
    #         company_database = json.load(json_file)
    # except FileNotFoundError:
    #     # If the file does not exist, create an empty dictionary
    #     company_database = {}
    # it = 0
    #
    # for k, v in tqdm(domain_map.items()):
    #     it += 1
    #
    #     if it <= 288:
    #         continue
    #
    #     time.sleep(1)
    #     brand_name = k
    #
    #     key = f"{brand_name}"
    #
    #     if key in company_database and len(company_database[key])>0:
    #         continue
    #
    #     if key in company_database:
    #         existing_set = set(company_database[key])
    #     else:
    #         existing_set = set()
    #
    #     urls, _ = gsearch_engine.query2url(query=brand_name,
    #                                        proxies=proxies)
    #     if len(urls) > 0:
    #         top1domain = urlparse(urls[0]).netloc
    #         command = f"curl 'https://company.clearbit.com/v2/companies/find?domain={top1domain}' -u {clearbit_api}:"
    #         process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #         stdout, stderr = process.communicate()
    #
    #         decoded_string = stdout.decode('utf-8')
    #         data_dict = json.loads(decoded_string)
    #         try:
    #             email_pattern = set([x.split('@')[1] for x in data_dict['site']['emailAddresses']])
    #         except KeyError:
    #             email_pattern = set()
    #
    #         updated_set = email_pattern.union(existing_set)
    #         company_database[key] = list(updated_set)
    #
    #         print(key, email_pattern)
    #
    #         # Saving to JSON
    #         with open('./datasets/company_database.json', 'w') as json_file:
    #             json.dump(company_database, json_file)

    '''We can also use Knowphish database to expand the existing brand name alias and domain alias'''
    # Knowphish_bkb = './datasets/KnowPhish_BKB/cat_and_pop_20240125_tld/targetlist'
    # tranco_top_1m_domains = [x.strip().split(',')[-1] for x in open('./datasets/tranco-top-1m.csv').readlines()]
    # with open('./checkpoints/company_database.json', 'r') as json_file:
    #     company_database = json.load(json_file)
    # ## manually add some?
    # company_database["Skype"] = ["skype.com"]
    # company_database["DropBox Inc."] = ["dropbox.com", "dropboxmail.com"]
    # company_database["Dropbox"] = ["dropbox.com", "dropboxmail.com"]
    # company_database['Xiaomi'] = ["xiaomi.com", "mi.com"]
    # company_database['State Technical College of Missouri'] = ["statetechmo.edu"]
    # company_database['Epos Card Co.,Ltd.'] = ['0101.co.jp']
    # company_database['MARUI GROUP CO.,LTD.'] = ['0101.co.jp']
    # company_database['Royal Bank Of Scotland'] = ['rbs.com']
    # company_database['Tepco Energy'] = ['tepco.co.jp']
    # company_database['GPC Cumbria'] = ['gpccumbria.co.uk']
    # company_database['Resona Bank'] = ['e.resonabank.co.jp']
    # company_database['ALL CITY GLOBAL LIMITED'] = ['allcitygloballtd.com']
    # company_database['FBI'] = ['ic.fbi.gov']
    # company_database['aeon financial services co., ltd'] = ['aeoncredit.com.my', 'aeonfinancialservices.com']
    # company_database['Stork Technical Services UK Limited'] = ['stork.com']
    # company_database['Ernst & Young'] = ['ey.com']
    # company_database['Bank of China'] = ['bank-of-china.com']
    # company_database['Credit Saison'] = ['creditsaison-in.com', 'creditsaison.in']
    # company_database['Swiss National Bank'] = ['snb.ch']
    # company_database['Cummins Inc'] = ['cummins.com']
    # company_database['O.P. Jindal Global University'] = ['jgu.edu']
    # company_database['Sagawa express'] = ['sagawa-exp.co.jp']
    #
    # for brand in tqdm(os.listdir(Knowphish_bkb)):
    #     for file in os.listdir(os.path.join(Knowphish_bkb, brand)):
    #         if file.endswith('.png'):
    #             os.remove(os.path.join(Knowphish_bkb, brand, file))
    #
    #         if file == 'brand_info.json':
    #             with open(os.path.join(Knowphish_bkb, brand, file), 'r') as json_file:
    #                 brand_info = json.load(json_file)
    #
    #             wiki_entity_name = [brand_info["entity_name"]] if (not brand_info["entity_name"].startswith('Q')) else []
    #             brand_names = wiki_entity_name # just the official names
    #             brand_names = list(set({v.casefold(): v for v in brand_names}.values()))
    #             brand_names = [x for x in brand_names if len(x) > 3 and is_english_or_allowed_symbols(x)]
    #             if len(brand_names) == 0:
    #                 continue
    #
    #             knowledge_needed = False
    #             for x in brand_names:
    #                 matched_brand = difflib.get_close_matches(x, company_database.keys(), n=1, cutoff=0.95)
    #                 if matched_brand:
    #                     print(x, matched_brand)
    #                     brand_domains = company_database[matched_brand[0]]
    #                     knowledge_needed = True
    #                     break
    #             if not knowledge_needed:
    #                 continue
    #
    #             brand_domains += [tldextract.extract(x).domain + '.' + tldextract.extract(x).suffix for x in brand_info["urls"]]
    #             brand_domains = list(set({v.lower(): v for v in brand_domains}.values()))
    #             brand_domains = [x for x in brand_domains if is_valid_domain(x)]
    #
    #             for brand_alias in brand_names:
    #                 company_database[brand_alias] = brand_domains
    #
    # with open('./checkpoints/company_database_knowphish.json', 'w') as json_file:
    #     json.dump(company_database, json_file)
    #
    '''Those official emails are pre-collected => Only for the LLM benchmark'''
    with open('./checkpoints/company_database_knowphish.json', 'r') as json_file:
        company_database = json.load(json_file)

    df = pd.read_csv('./datasets/senders_v6.csv')
    for it, row in df.iterrows():
        org_name = row['name']
        email = row['official_email']
        if isinstance(email, str) and '@' in email:
            company_database[org_name] = [tldextract.extract(email).domain + '.' + tldextract.extract(email).suffix]

    with open('./checkpoints/company_database_knowphish_v2.json', 'w') as json_file:
        json.dump(company_database, json_file)


