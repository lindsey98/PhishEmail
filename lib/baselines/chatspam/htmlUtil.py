from lxml.html.clean import Cleaner
from urllib.parse import urlparse, urlunparse

filtered_attributes = ['src', 'href', 'alt', 'title', 'name', 'id', 'class']
unwrapped_tags = ['font', 'strong', 'b']


def shorten_url_path(url, max_length=10):
    parsed_url = urlparse(url)
    parsed_url = parsed_url._replace(path=parsed_url.path[:max_length],
                                     params=parsed_url.params[:max_length], query=parsed_url.query[:max_length],
                                     fragment=parsed_url.fragment[:max_length])
    return urlunparse(parsed_url)


def clean_script_style_comments(html):
    for script in html.xpath('.//script'):
        script.drop_tree()

    for style in html.xpath('.//style'):
        style.drop_tree()

    for comment in html.xpath('.//comment()'):
        comment.drop_tree()


def clean_attributes(html):
    for element in html.iter():
        attributes = list(element.attrib.keys())
        for attr in attributes:
            if attr not in filtered_attributes:
                del element.attrib[attr]


def clean_textless_tag(html):
    for element in html.xpath('//*[not(normalize-space())]'):
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)


def unwrap_tag(html):
    for element in html.xpath('.//font'):
        element.drop_tag()

    for element in html.xpath('.//strong'):
        element.drop_tag()

    for element in html.xpath('.//b'):
        element.drop_tag()


def shorten_url(html):
    for img in html.xpath('//img[@src]'):
        img.attrib['src'] = shorten_url_path(img.attrib['src'])

    for a in html.xpath('//a[@href]'):
        a.attrib['href'] = shorten_url_path(a.attrib['href'])


def remove_middle(html):
    elements = html.xpath('//*')
    center_index = len(elements) // 2
    element_to_remove = elements[center_index]
    parent = element_to_remove.getparent()
    if parent is not None:
        parent.remove(element_to_remove)
