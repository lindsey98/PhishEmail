import json
from tqdm import tqdm
import os
from tldextract import tldextract
import re

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
    # hunterio_api = open('./datasets/hunterio_key.txt').read()
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
    #     # response = requests.get(f"https://api.hunter.io/v2/domain-search?company={brand_name}&api_key={hunterio_api}")
    #     # data = response.json()
    #     # email_pattern = set([x['value'].split('@')[1] for x in data['data']['emails']])
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

    # Add brand information to the database
    # add_brand('Brand A', 'Retail', 'contact@branda.com')
    # add_brand('Brand B', 'Tech', 'contact@brandb.com')
    #
    # # Retrieve contact email for a brand
    # brand_name_to_lookup = 'Brand A'
    # contact_email = get_contact_email(brand_name_to_lookup)
    #
    # if contact_email:
    #     print(f"Contact Email for {brand_name_to_lookup}: {contact_email}")
    # else:
    #     print(f"Brand {brand_name_to_lookup} not found in the database.")

    Knowphish_bkb = './datasets/KnowPhish_BKB/cat_and_pop_20240125_tld/targetlist'
    tranco_top_1m_domains = [x.strip().split(',')[-1] for x in open('./datasets/tranco-top-1m.csv').readlines()]
    with open('./datasets/company_database.json', 'r') as json_file:
        company_database = json.load(json_file)

    for brand in tqdm(os.listdir(Knowphish_bkb)):
        for file in os.listdir(os.path.join(Knowphish_bkb, brand)):
            if file.endswith('.png'):
                os.remove(os.path.join(Knowphish_bkb, brand, file))

            if file == 'brand_info.json':
                with open(os.path.join(Knowphish_bkb, brand, file), 'r') as json_file:
                    brand_info = json.load(json_file)

                wiki_entity_name = [brand_info["entity_name"]] if (not brand_info["entity_name"].startswith('Q')) else []
                # brand_names = wiki_entity_name + brand_info["alias"]
                brand_names = wiki_entity_name # just the official names
                brand_names = list(set({v.casefold(): v for v in brand_names}.values()))
                brand_names = [x for x in brand_names if len(x) > 3 and is_english_or_allowed_symbols(x)]
                if len(brand_names) == 0:
                    continue

                whois_domain_names = brand_info['whois_info'].get("domain_name", None)
                if not whois_domain_names:
                    whois_domain_names = []
                elif isinstance(whois_domain_names, str):
                    whois_domain_names = [whois_domain_names]
                else:
                    whois_domain_names = whois_domain_names
                brand_domains = [tldextract.extract(x).domain + '.' + tldextract.extract(x).suffix for x in brand_info["urls"]] + whois_domain_names
                brand_domains = list(set({v.lower(): v for v in brand_domains}.values()))
                brand_domains = [x for x in brand_domains if is_valid_domain(x)]

                ## I try to filter out those unpopular benign sites
                is_popular = any([True for x in brand_domains if x.lower() in tranco_top_1m_domains])
                if not is_popular:
                    continue

                for brand_alias in brand_names:
                    company_database[brand_alias] = brand_domains

    with open('./datasets/company_database_knowphish.json', 'w') as json_file:
        json.dump(company_database, json_file)