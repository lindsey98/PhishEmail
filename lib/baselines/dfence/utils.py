import numpy as np
import pandas as pd
import os
import glob
import pickle
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import configparser
from datetime import datetime
import threading, time
import psutil

'''
This file consists of different misc functions
'''
cfg = configparser.ConfigParser()
cfg.read('./lib/baselines/dfence/conf.cfg')
URLDEEP_TRUNCATION_LENGTH = cfg.get('env', 'urldeep_truncation_length')
URL_ENCODING_MODE = cfg.get('env', 'urldeep_encoding_mode')
DEBUG_LEVEL = int(cfg.get('env', 'debug_level'))


def read_pkl(str_file_path):
    with open(str_file_path, 'rb') as f_read:
        df_dest = pickle.load(f_read)
        
    return df_dest

def extract_encoding(url_data, tokenizer, truncate_len,
                     label_mode, encoding_mode='default'):
    """Encode characters in URL in numbers so that neural network can understand it
      There are two possible ways to encode characters by specifying emb_mode parameter.
      By default we use 'embedding' mode as we are using Embedding for top layer of the neural network
    """

    if label_mode:
        url_data = url_data[['url', 'label']]
    else:
        url_data = url_data[['url']]

    sequence_of_int = tokenizer.texts_to_sequences(url_data['url'])
    data_size = len(sequence_of_int)
    num_tokens = len(tokenizer.word_index) + 1  # +1 for reserve padding 0

    if encoding_mode == 'embedding':  # embedding layer does not require binary encoding..so we stop at integer output
        X = np.zeros(shape=(data_size, truncate_len))
        Y = np.zeros(shape=(data_size, 1))
        for i in range(len(sequence_of_int)):
            if len(sequence_of_int[i]) <= truncate_len:
                url_len = len(sequence_of_int[i])
            else:
                url_len = truncate_len
            X[i, :url_len] = sequence_of_int[i][:url_len]
            if label_mode:
                Y[i] = url_data['label'][i]

    else:  # default and standard way of converting to one hot encoding
        X = np.zeros(shape=(data_size, truncate_len, num_tokens))  # +1 for reserve padding 0
        Y = np.zeros(shape=(data_size, 1))
        for i in range(len(sequence_of_int)):
            if len(sequence_of_int[i]) <= truncate_len:
                url_len = len(sequence_of_int[i])
            else:
                url_len = truncate_len
            X[i, :url_len, :] = to_categorical(sequence_of_int[i], num_classes=num_tokens)[:url_len]
            if label_mode:
                Y[i] = url_data['label'][i]

    if label_mode:
        return X, Y
    else:
        return X


def prepMetaFeatures(structural_pred, text_pred, url_pred):
    """
        This function take prediction results of three sub-modules as inputs
        merge the prediction into one single dataframe
        return the dataframe for meta classifier prediction
    """
    structural_pred.rename(columns={'Predicted Score': 'struct_pred'}, inplace=True)
    text_pred.rename(columns={'Predicted Score': 'text_pred'}, inplace=True)
    url_pred.rename(columns={'Predicted Score': 'url_pred'}, inplace=True)
    
    if 'label' in structural_pred.columns:
        meta_test_data = structural_pred[['struct_pred', 'label','ID']].merge(text_pred[['text_pred', 'ID']], on='ID', how='left')
    else:
        meta_test_data = structural_pred[['struct_pred','ID']].merge(text_pred[['text_pred', 'ID']], on='ID', how='left')

    meta_test_data = meta_test_data.merge(url_pred[['url_pred','ID']], on='ID', how='left')
    return meta_test_data
