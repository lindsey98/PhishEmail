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

'''
Main function
'''
cfg = configparser.ConfigParser()
cfg.read('./lib/baselines/helphed/conf.cfg')

MODEL_DIR_PATH = os.path.abspath(cfg.get('env', 'model_dir_path'))

def predict_from_multiple_estimator(estimators, label_encoder, X_list, weights = None):
    pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)])
    pred2 = np.average(pred1, axis=0, weights=weights)
    pred = np.argmax(pred2, axis=1)
    return label_encoder.inverse_transform(pred)

def test(email_dir, model_name=''):
    """test with labeled files, it can be: 1 mal and 1 benign, 1 benign only or 1 mal only
    """
    test_dataset = EmailDataset(email_dir)

    result_list = []
    for it in tqdm(range(len(test_dataset))):
        email_file_path = test_dataset.file_list[it]
        result = parse_email_parts(email_file_path, 0)
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

    ## Method 1: stacking model
    model_path = f'{MODEL_DIR_PATH}/stacking_model.pkl'
    with open(model_path, 'rb') as file:
        sclf = pickle.load(file)

    y_pred_stacked = sclf.predict(X_test).tolist()

    ### Method 2: ensemble model
    model_path = f'{MODEL_DIR_PATH}/dt_model.pkl'  # name[0] should contain the identifier of the classifier
    with open(model_path, 'rb') as file:
        dt_model = pickle.load(file)
    model_path = f'{MODEL_DIR_PATH}/knn_model.pkl'  # name[0] should contain the identifier of the classifier
    with open(model_path, 'rb') as file:
        knn_model = pickle.load(file)
    fitted_estimators = [dt_model, knn_model]

    model_path = f'{MODEL_DIR_PATH}/label_encoder.pkl'  # name[0] should contain the identifier of the classifier
    with open(model_path, 'rb') as file:
        label_encoder = pickle.load(file)

    X_test1, X_test2 = X_test.iloc[:, 0:18], X_test.iloc[:, 18:]
    X_test_list = [X_test1, X_test2]
    y_pred_voting = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_test_list).tolist()

    return y_pred_stacked, y_pred_voting


if __name__ == '__main__':
    desc_folder = './datasets/sjtu_phish/'
    test(desc_folder)
