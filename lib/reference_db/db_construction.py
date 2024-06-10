from lib.web_utils.gsearch import GoogleSearch
from urllib.parse import urlparse
import pickle
import json
import subprocess
from tqdm import tqdm
import time


if __name__ == "__main__":
    hunterio_api = open('./datasets/hunterio_key.txt').read()
    clearbit_api = open('./datasets/clearbit_key.txt').read()

    gsearch_api, gsearch_id = [x.strip() for x in open('./datasets/google_api_key.txt').readlines()]
    gsearch_engine = GoogleSearch(SEARCH_ENGINE_ID=gsearch_id, SEARCH_ENGINE_API=gsearch_api)
    proxies = {
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890",
    }

    with open('./lib/phishpedia/models/domain_map.pkl', "rb") as handle:
        domain_map = pickle.load(handle)

    # Load the existing data from JSON file
    try:
        with open('./datasets/company_database.json', 'r') as json_file:
            company_database = json.load(json_file)
    except FileNotFoundError:
        # If the file does not exist, create an empty dictionary
        company_database = {}
    it = 0

    for k, v in tqdm(domain_map.items()):
        it += 1

        if it <= 288:
            continue

        time.sleep(1)
        brand_name = k

        key = f"{brand_name}"

        if key in company_database and len(company_database[key])>0:
            continue

        if key in company_database:
            existing_set = set(company_database[key])
        else:
            existing_set = set()

        # response = requests.get(f"https://api.hunter.io/v2/domain-search?company={brand_name}&api_key={hunterio_api}")
        # data = response.json()
        # email_pattern = set([x['value'].split('@')[1] for x in data['data']['emails']])

        urls, _ = gsearch_engine.query2url(query=brand_name,
                                           proxies=proxies)
        if len(urls) > 0:
            top1domain = urlparse(urls[0]).netloc
            command = f"curl 'https://company.clearbit.com/v2/companies/find?domain={top1domain}' -u {clearbit_api}:"
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            decoded_string = stdout.decode('utf-8')
            data_dict = json.loads(decoded_string)
            try:
                email_pattern = set([x.split('@')[1] for x in data_dict['site']['emailAddresses']])
            except KeyError:
                email_pattern = set()

            updated_set = email_pattern.union(existing_set)
            company_database[key] = list(updated_set)

            print(key, email_pattern)

            # Saving to JSON
            with open('./datasets/company_database.json', 'w') as json_file:
                json.dump(company_database, json_file)

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
