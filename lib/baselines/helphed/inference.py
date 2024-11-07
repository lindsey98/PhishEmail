import warnings
warnings.filterwarnings('ignore')
import argparse
from lib.baselines.helphed.getFeatures import parse_email_parts
from lib.data.Dataset import EmailDataset
import configparser
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import time
import csv

'''
Main function
'''
cfg = configparser.ConfigParser()
cfg.read('./lib/baselines/helphed/conf.cfg')

MODEL_DIR_PATH = os.path.abspath(cfg.get('env', 'model_dir_path'))
## Method 1: stacking model
with open(f'{MODEL_DIR_PATH}/stacking_model.pkl', 'rb') as file:
    STACKING_MODEL = pickle.load(file)

### Method 2: ensemble model
with open(f'{MODEL_DIR_PATH}/dt_model.pkl', 'rb') as file:
    dt_model = pickle.load(file)
with open(f'{MODEL_DIR_PATH}/knn_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)
INDIVIDUAL_ESTIMATORS = [dt_model, knn_model]
with open(f'{MODEL_DIR_PATH}/label_encoder.pkl', 'rb') as file:
    LABEL_ENCODER = pickle.load(file)

def predict_from_multiple_estimator(estimators, label_encoder, X_list, weights = None):
    pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)])
    pred2 = np.average(pred1, axis=0, weights=weights)
    pred = np.argmax(pred2, axis=1)
    return label_encoder.inverse_transform(pred)

def test(email_dir):
    """test with labeled files, it can be: 1 mal and 1 benign, 1 benign only or 1 mal only
    """
    test_dataset = EmailDataset(email_dir)
    total_time = 0
    result_list = []
    for it in range(len(test_dataset)):
        email_file_path = test_dataset.file_list[it]
        start_time = time.time()
        result = parse_email_parts(email_file_path, 0)
        total_time += time.time() - start_time
        result_list.append(result)

    df = pd.DataFrame(result_list)

    # flatten Word2Vec features
    w2v = df['Word2vec']
    vec = np.array(w2v.to_list())
    df_new = pd.DataFrame(vec)

    ######################## Content-based features training remove text-based features and unwanted content-based features
    df = df.drop(['scripts', 'forms', 'nports', 'link_images', 'Word2vec', 'label'], axis=1)  # Converting the encoding column to categorical - it assigns an int on each encoding-name
    df['encoding'] = df['encoding'].astype('category')
    # Integer Encoding the 'encoding' column
    enc_encode = LabelEncoder()
    # Integer encoding the 'encoding' column
    df['encoding'] = enc_encode.fit_transform(df.encoding)

    #########################################
    # Concat word2vec with content-based features TRAINING
    X_test = pd.concat([df, df_new], axis=1)
    X_test.columns = X_test.columns.astype(str)
    start_time = time.time()
    y_pred_stacked = STACKING_MODEL.predict(X_test).tolist()
    total_time1 = time.time() - start_time + total_time

    X_test1, X_test2 = X_test.iloc[:, 0:18], X_test.iloc[:, 18:]
    X_test_list = [X_test1, X_test2]
    start_time = time.time()
    y_pred_voting = predict_from_multiple_estimator(INDIVIDUAL_ESTIMATORS, LABEL_ENCODER, X_test_list).tolist()
    total_time2 = time.time() - start_time + total_time

    return y_pred_stacked, y_pred_voting, total_time1, total_time2


if __name__ == '__main__':
    desc_folder = './datasets/GPT_V6/v6'
    dataset = EmailDataset(desc_folder)
    csv_file_path = './datasets/GPT_results_helphed_corrected.csv'

    # # Check if we're writing to a new file, and write the header if so
    # if not os.path.exists(csv_file_path):
    #     with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(['email_file_path',
    #                          'sender_name', 'sender_address',
    #                          'to_names', 'to_addresses',
    #                          'subject',
    #                          'helphed_stacking_pred',
    #                          'helphed_stacking_runtime',
    #                          'helphed_voting_pred',
    #                          'helphed_voting_runtime'])
    #
    # for it in tqdm(range(len(dataset))):
    #     if dataset.file_list[it] in [x.split(',')[0] for x in open(csv_file_path).readlines()]:
    #         continue
    #
    #     email_file_path, (sender_name, sender_address), \
    #     (to_names, to_addresses), reply_to_address, \
    #     subject, email_body_text, header = dataset[it]
    #
    #     # if email_file_path != './datasets/GPT_Dataset/Giancarlo Pellegrino_Web_Zero_Gemini.eml':
    #     #     continue
    #
    #     helphed_stacking_pred, helphed_voting_pred, helphed_stacking_runtime, helphed_voting_runtime = test(email_file_path)
    #     helphed_stacking_pred = helphed_stacking_pred[0]
    #     helphed_voting_pred = helphed_voting_pred[0]
    #     print(
    #         f"HelpHed stacking prediction = {helphed_stacking_pred} with runtime = {helphed_stacking_runtime} \t"
    #         f"HelpHed voting prediction = {helphed_voting_pred} with runtime = {helphed_voting_runtime}")
    #
    #     # Append the new row to the CSV file
    #     with open(csv_file_path, mode='a', newline='', encoding='utf-8', errors='ignore') as file:
    #         writer = csv.writer(file)
    #         writer.writerow([email_file_path,
    #                          sender_name, sender_address,
    #                          to_names, to_addresses,
    #                          subject,
    #                          helphed_stacking_pred,
    #                          helphed_stacking_runtime,
    #                          helphed_voting_pred,
    #                          helphed_voting_runtime,
    #                          ])

    df_helphed = pd.read_csv(csv_file_path)
    csv_file_path = f'./datasets/GPT_results_augmented.csv'
    df = pd.read_csv(csv_file_path)
    df = df.drop([
                  'helphed_stacking_pred',
                 'helphed_stacking_runtime',
                 'helphed_voting_pred',
                 'helphed_voting_runtime'], axis=1)

    df_new = df.merge(df_helphed[['email_file_path',
                                  'sender_name', 'sender_address',
                                   'to_names', 'to_addresses',
                                  'subject',
                                  'helphed_stacking_pred',
                             'helphed_stacking_runtime',
                             'helphed_voting_pred',
                             'helphed_voting_runtime']], on='email_file_path', how='left')

    df_new.to_csv(f'./datasets/GPT_results_augmented.csv', index=False)


