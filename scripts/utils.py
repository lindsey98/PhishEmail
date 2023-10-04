import re
from bs4 import BeautifulSoup
import requests
from io import BytesIO, StringIO
import urllib.parse
from requests.exceptions import Timeout, ConnectionError
import base64

'''Extract all images from email body'''
def get_img_links(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Get image links from <img> tags
    img_links = [img['src'] for img in soup.find_all('img', src=True)]

    # Get image links from <picture> tags
    picture_links = [source['srcset'] for source in soup.find_all('source', srcset=True)]

    # Get image links from inline styles (background-image)
    inline_style_links = []
    for tag in soup.find_all(style=True):
        style = tag['style']
        match = re.search("url\(['\"]?(.*?)['\"]?\)", style)
        if match:
            inline_style_links.append(match.group(1))

    # Get image links from <object> tags
    object_links = [obj['data'] for obj in soup.find_all('object', data=True)]

    # Get image links from <embed> tags
    embed_links = [embed['src'] for embed in soup.find_all('embed', src=True)]

    # Get SVG images
    svg_links = [str(svg) for svg in soup.find_all('svg')]

    # Get Canvas elements
    canvas_links = [str(canvas) for canvas in soup.find_all('canvas')]

    # Get Data URIs
    data_uris = []
    for tag in soup.find_all(['img', 'source']):
        src = tag.get('src', '')
        if src.startswith('data:image'):
            data_uris.append(src)

    # Combine all links
    all_links = img_links + picture_links + inline_style_links + object_links + embed_links + svg_links + canvas_links + data_uris

    return all_links

'''Load the image links in a safe way'''
def load_images(img_links, proxies=None, max_retries=1):
    img_bytes_list = []
    for link in img_links:
        if link.startswith('data:image/'): # Handle Data URIs
            try:
                # Extract the base64 encoded part
                base64_data = link.split(',')[1]
                img_bytes = BytesIO(base64.b64decode(base64_data))
                img_bytes_list.append(img_bytes)
            except Exception as e:
                pass
        elif link.startswith('<svg'):
            # Handle SVGs
            try:
                svg_bytes = StringIO(link)  # SVG is XML text, so we use StringIO
                img_bytes_list.append(svg_bytes)
            except Exception as e:
                pass
        else:
            retries = 0
            while retries <= max_retries:
                try:
                    response = requests.get(link, proxies=proxies, timeout=5)
                    if response.status_code == 200:
                        content_type = response.headers.get('Content-Type', '')
                        if content_type.startswith('image/'):
                            img_bytes = BytesIO(response.content)
                            img_bytes_list.append(img_bytes)
                            break  # Successfully downloaded, exit the retry loop
                        else:
                            break  # Not an image, exit the retry loop
                    else:
                        break  # HTTP error, exit the retry loop
                except (Timeout, ConnectionError) as e:
                    retries += 1  # Increment the retry counter and try again
                except Exception as e:
                    break  # Unknown error, exit the retry loop

    return img_bytes_list

'''Extract all links from email body'''
def get_links(html_content):
    soup = BeautifulSoup(html_content, 'lxml')
    links = []

    # Standard <a> tags
    for tag in soup.find_all('a', href=True):
        text = tag.string if tag.string else tag.text
        url = tag['href']
        if urllib.parse.urlparse(url).scheme:
            links.append((text, url))

    # JavaScript links
    for tag in soup.find_all(onclick=True):
        onclick = tag['onclick']
        match = re.search(r"window\.location\.href='(.*?)'|window\.open\('(.*?)'\)", onclick)
        if match:
            url = match.group(1) or match.group(2)
            text = tag.string if tag.string else tag.text
            if urllib.parse.urlparse(url).scheme:
                links.append((text, url))

    # Form action links
    for tag in soup.find_all('form', action=True):
        url = tag['action']
        if urllib.parse.urlparse(url).scheme:
            text = 'Form: ' + (tag.get('name') or tag.get('id') or '')
            if urllib.parse.urlparse(url).scheme:
                links.append((text, url))

    # Area map links
    for tag in soup.find_all('area', href=True):
        url = tag['href']
        text = tag.get('alt', 'Area Map')
        if urllib.parse.urlparse(url).scheme:
            links.append((text, url))

    # SVG Links
    for tag in soup.find_all('a', {'xlink:href': True}):
        url = tag['xlink:href']
        text = tag.string if tag.string else tag.text
        if urllib.parse.urlparse(url).scheme:
            links.append((text, url))

    return links

'''Validate the domain'''
def is_valid_domain(domain: str) -> bool:
    '''
        Check if the provided string is a valid domain
        :param domain:
        :return:
    '''
    # Regular expression to check if the string is a valid domain without spaces
    pattern = re.compile(
        r'^(?!-)'  # Cannot start with a hyphen
        r'(?!.*--)'  # Cannot have two consecutive hyphens
        r'(?!.*\.\.)'  # Cannot have two consecutive periods
        r'(?!.*\s)'  # Cannot contain any spaces
        r'[a-zA-Z0-9-]{1,63}'  # Valid characters are alphanumeric and hyphen
        r'(?:\.[a-zA-Z]{2,})+$'  # Ends with a valid top-level domain
    )
    it_is_a_domain = bool(pattern.fullmatch(domain))
    return it_is_a_domain


def remove_specific_special_chars(text):
    text = text.replace('\xa0', ' ')
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text)

    # Find all URLs in the text
    urls = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text)

    # Replace URLs with placeholders
    for i, url in enumerate(urls):
        placeholder = f"URLPLACEHOLDER{i}"
        text = text.replace(url, placeholder)

    # Remove specific special characters
    text = re.sub('[$%^*+#=\[\]\\\\{}~]', '', text)

    # Replace placeholders with original URLs
    for i, url in enumerate(urls):
        placeholder = f"URLPLACEHOLDER{i}"
        text = text.replace(placeholder, url)

    return text


class UnsupportedContentTypeError(Exception):
    pass

