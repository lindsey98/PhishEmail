


# sudo apt install spamassassin python-pytest
# python -m pip install spamassassin_client==0.0.3
# sudo systemctl start spamassassin

import os
from spamassassin_client import SpamAssassin
from lib.data.Dataset import EmailDataset
import pandas as pd
import time

def main():
    path = os.path.dirname(__file__)
    for test in FILES:
        filename = os.path.join(path, test['name'])
        with open(filename, "rb") as f:
            print("\nProcessing file: {}".format(filename))
            assassin = SpamAssassin(f.read())
            print(assassin)
            if assassin.is_spam():
                print("The received message is considered spam with a score of {0}".format(assassin.get_score()))
            print('\nreport_fulltext:', assassin.get_fulltext())
            print('score:', assassin.get_score())
            print('report_json:', assassin.get_report_json())

def test(email_dir):
    """test with labeled files, it can be: 1 mal and 1 benign, 1 benign only or 1 mal only
    """
    test_dataset = EmailDataset(email_dir)
    email_files = test_dataset.file_list
    data = []

    for file_path in email_files:
        with open(file_path, 'rb') as f:
            email_content = f.read()

        max_entries = 3
        while True and max_entries > 0:
            try:
                start_time = time.time()
                assassin = SpamAssassin(email_content)
                runtime = time.time() - start_time
                data.append({'spam': assassin.is_spam(),
                             'score': assassin.get_score(),
                             'runtime': runtime,
                             'metadata': assassin.report_json,
                             'file': file_path})
                break
            except (UnicodeDecodeError, ValueError):
                data.append({'spam': 0,
                             'score': 0,
                             'runtime': 0,
                             'metadata': {},
                             'file': file_path})
                break
            except ConnectionRefusedError:
                time.sleep(2)
                max_entries -= 1
                if max_entries == 0:
                    raise ConnectionRefusedError

    # Create DataFrame
    df = pd.DataFrame(data)
    pred_class, pred_score, pred_runtime, pred_metadata = df['spam'], df['score'],  df['runtime'], df['metadata']
    pred_class = pred_class.tolist()
    pred_score = pred_score.tolist()
    pred_runtime = pred_runtime.tolist()
    pred_metadata = pred_metadata.to_list()

    return pred_class, pred_score, pred_runtime, pred_metadata

if __name__ == "__main__":
    FILES = [dict(type='spam', name='sample-spam.txt'),
             dict(type='ham', name='sample-nonspam.txt')]
    main()