import requests
from tldextract import tldextract

class GoogleSearch():

    def __init__(self, SEARCH_ENGINE_API, SEARCH_ENGINE_ID):
        self.SEARCH_ENGINE_API = SEARCH_ENGINE_API  # API key for Gsearch
        self.SEARCH_ENGINE_ID = SEARCH_ENGINE_ID  # ID for Gsearch engine

    def query2image(self, query, proxies=None, num=3):
        '''
            Retrieve the images from Google image search
            Args:
                query: query string
                num: number of results returned
            Returns:
                returned_urls
                context_links: the source URLs for the images
        '''
        returned_urls = []
        context_links = []
        if len(query) == 0:
            return returned_urls, context_links

        URL = f"https://www.googleapis.com/customsearch/v1?key={self.SEARCH_ENGINE_API}&cx={self.SEARCH_ENGINE_ID}&q={query}&searchType=image&num={num}&filter=1"
        data = requests.get(URL, proxies=proxies).json()
        if 'error' in list(data.keys()):
            if data['error']['code'] == 429:
                raise RuntimeError("Google search exceeds quota limit")
        search_items = data.get("items")
        if search_items is None:
            return returned_urls, context_links

        # iterate over results found
        for i, search_item in enumerate(search_items, start=1):
            link = search_item.get("image")["thumbnailLink"]
            context_link = search_item.get("image")['contextLink']
            returned_urls.append(link)
            context_links.append(context_link)

        return returned_urls, context_links

    def query2url(self, query,
                  allowed_domains=[],
                  forbidden_domains=[],
                  forbidden_subdomains=[],
                  proxies=None,
                  title_trucate=False,
                  num=5):
        '''
            Retrieve the URLs from Google search
            Args:
                query: query string
                allowed_domains: interested domains
                forbidden_domains:
                num: number of results returned
            Returns:
                returned_urls:
                returned_brand_names: title of the websites
        '''
        returned_urls = []
        returned_titles = []
        if len(query) == 0 or query is None:
            return returned_urls, returned_titles

        # initiate a Google query
        URL = f"https://www.googleapis.com/customsearch/v1?key={self.SEARCH_ENGINE_API}&cx={self.SEARCH_ENGINE_ID}&q={query}&num={num}"
        data = requests.get(URL, proxies=proxies).json()
        search_items = data.get("items")
        if 'error' in list(data.keys()):
            if data['error']['code'] == 429:
                raise RuntimeError("Google search exceeds quota limit")
        if search_items is None:
            return returned_urls, returned_titles

        # iterate over results found
        for i, search_item in enumerate(search_items, start=1):
            link = search_item.get("link")
            # if 'wikipedia' in allowed_domains:
            if title_trucate:
                title = search_item.get("title").split(' - ')[0]  # remove strings after -
            else:
                title = search_item.get("title")

            # list of ignored domains is specified
            if (tldextract.extract(link).domain in forbidden_domains) or \
                    tldextract.extract(link).subdomain in forbidden_subdomains:
                continue

            # allowed domain is specified
            if len(allowed_domains):
                if tldextract.extract(link).domain in allowed_domains:
                    if tldextract.extract(query).domain in tldextract.extract(link).domain:
                        returned_titles.insert(0, title)
                        returned_urls.insert(0, link)
                    else:
                        returned_titles.append(title)
                        returned_urls.append(link)

            else:  # not specified
                if tldextract.extract(query).domain in tldextract.extract(
                        link).domain:  # prioritize the most relevant recommendation
                    returned_titles.insert(0, title)
                    returned_urls.insert(0, link)
                else:
                    returned_titles.append(title)
                    returned_urls.append(link)

        return returned_urls[:min(len(returned_urls), num)], \
               returned_titles[:min(len(returned_titles), num)]