import os
import time
from selenium.webdriver.common.by import By
import pandas as pd
import json
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os
import urllib.parse  # For URL joining

def parse_json(text, delimiter='{"text":'):
    """
    Finds and extracts JSON objects from a string based on a given delimiter.
    """
    json_objects = []
    start = 0
    while True:
        start = text.find(delimiter, start)
        if start == -1:
            break
        end = text.find(delimiter, start + 1)
        json_str = text[start:end].strip()
        if end == -1:
            json_str = text[start:]
        try:
            json_obj = json.loads(json_str)
            json_objects.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
        start += len(delimiter)
    return json_objects

if __name__ == '__main__':
    root_dir = "./datasets/spam_archieve"
    os.makedirs(root_dir, exist_ok=True)
    '''Download feed'''
    # chrome_options = Options()
    # chrome_options.add_argument("--headless")
    # chrome_options.add_argument("--window-size=1920,1080")
    # prefs = {
    #     "profile.default_content_settings.popups": 0,
    #     "download.default_directory": root_dir,  # Change this to a valid directory if you want to save downloads
    #     "directory_upgrade": True,
    #     "download.prompt_for_download": False,
    #     "download.directory_upgrade": True,
    #      "safebrowsing.enabled": True
    # }
    # chrome_options.add_experimental_option("prefs", prefs)
    #
    # # If recreate is True or driver does not exist, create a new driver
    # driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
    # time.sleep(3)
    #
    # # Open the URL
    # driver.get('https://untroubled.org/spam/')
    # time.sleep(3)
    #
    # # Find all <a> elements under the specified <p><b>Please select date range:</b>
    # # Assuming this is a unique identifier on the page
    # a_tags = driver.find_elements_by_tag_name('a')
    # pattern = re.compile(r'\b\d{4}(-\d{2})?\.7z\b')
    # links = [urllib.parse.urljoin('https://untroubled.org/spam/', a.get_attribute('href')) for a in a_tags if pattern.match(a.text)]
    #
    # print(len(links))
    #
    # columns = ['Id', 'Parsed email']
    # df = pd.DataFrame(columns=columns)
    #
    # for link in links:
    #     driver.get(link)
    #
    # driver.quit()

    # FROM, TO, subject, body, date


    # file_names = [f"./datasets/miller-data-{i}.csv" for i in range(1, 8)]
    #
    # # Read and merge the files
    # df = pd.concat([pd.read_csv(file, header=0) for file in file_names])
    #
    # # Read the CSV file
    # #
    # # # Extract the part after "Body:" in 'Parsed email' and store it in a new column
    # df['Body_Content'] = df['Parsed email'].apply(lambda x: x.split('Body:', 1)[1] if 'Body:' in x else x)
    # #
    # # # Remove duplicate rows based on 'Body_Content' column
    # df = df.drop_duplicates(subset='Body_Content')
    # #
    # # Replace all occurrences of \" with "
    # df['Parsed email'] = df['Parsed email'].str.replace(r'\\\"', '"', regex=True)
    #
    # # Select only the 'Parsed email' column
    # texts = df['Parsed email'].tolist()
    #
    # # Convert to the required JSON format
    # formatted_data = [{'text': text} for text in texts]
    #
    # # Save to a new JSON file
    # with open('./datasets/miller-data.json', 'w', encoding='utf-8') as f:
    #     for entry in formatted_data:
    #         json.dump(entry, f)
    #         f.write('\n')