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

def feature_importance_viz():
    from xgboost import plot_importance
    import matplotlib.pyplot as plt
    structural_features = ['nurls', 'encoding', 'nparts', 'hasHTML', 'attachments', 'badwords', 'ipurls', 'diffhref', 'ndots', 'nrecs', 'checkdomains', 'subject_badwords', 'distinct_words', 'char_count', 'word_count', 'richness', 'RE_presence', 'named_urls']
    # Sort feature importances and select top-k
    top_k = 3
    dt_feature_importances = STACKING_MODEL.clfs_[0]['decisiontreeclassifier'].feature_importances_
    sorted_indices = np.argsort(dt_feature_importances)[::-1]  # Sort descending
    top_features = [structural_features[i] for i in sorted_indices[:top_k]]
    top_importances = dt_feature_importances[sorted_indices[:top_k]]

    plt.figure(figsize=(16, 4.5))
    plt.barh(top_features, top_importances, color="skyblue")  # Reverse to have top feature on top
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=20, fontweight='bold')
    plt.gca().invert_yaxis()  # Ensure the highest importance feature is at the top
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('./debug.png')

def predict_from_multiple_estimator(estimators, label_encoder, X_list, weights = None):
    pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)])
    pred2 = np.average(pred1, axis=0, weights=weights)
    pred = np.argmax(pred2, axis=1)
    return label_encoder.inverse_transform(pred)

def predict_from_multiple_estimator_proba(estimators, label_encoder, X_list, weights = None):
    pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)])
    pred2 = np.average(pred1, axis=0, weights=weights)
    pred = pred2[:, 1]
    return pred

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

def test_proba(email_dir):
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
    y_pred_stacked = STACKING_MODEL.predict_proba(X_test).tolist()[0][1]
    total_time1 = time.time() - start_time + total_time

    X_test1, X_test2 = X_test.iloc[:, 0:18], X_test.iloc[:, 18:]
    X_test_list = [X_test1, X_test2]
    start_time = time.time()
    y_pred_voting = predict_from_multiple_estimator_proba(INDIVIDUAL_ESTIMATORS, LABEL_ENCODER, X_test_list).tolist()
    total_time2 = time.time() - start_time + total_time

    return y_pred_stacked, y_pred_voting, total_time1, total_time2


