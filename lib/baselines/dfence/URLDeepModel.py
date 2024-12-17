import time

import numpy as np
import os
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, auc
import pandas as pd
import pickle
# from beautifultable import BeautifulTable
from lib.baselines.dfence.utils import extract_encoding
import configparser
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, Embedding, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer

'''
URL model for training and prediction
'''


cfg = configparser.ConfigParser()
cfg.read('./lib/baselines/dfence/conf.cfg')

URLDEEP_TRUNCATION_LENGTH = int(cfg.get('env', 'urldeep_truncation_length'))
URL_ENCODING_MODE = cfg.get('env', 'urldeep_encoding_mode')
URLDEEP_LSTM_UNITS = int(cfg.get('env', 'urldeep_lstm_units'))
URLDEEP_CONV_FILTER = int(cfg.get('env', 'urldeep_conv1d_filter'))
URLDEEP_DENSE_UNITS = int(cfg.get('env', 'urldeep_dense_units'))
URLDEEP_EPOCH = int(cfg.get('env', 'urldeep_epoch'))
URLDEEP_MAXPOOLING_SIZE = int(cfg.get('env', 'urldeep_maxpooling_size'))
URLDEEP_BATCH_SIZE = int(cfg.get('env', 'urldeep_batch_size'))


class URLDeep(object):
    """This class implements URLDeep model for processing URLs using embeddings and CNNLSTM architecture.
    """
    def __init__(self, model_name, model_type, model_dir_path, report_dir_path):
        """ Initialize the object: model architecture is also defined here
         """
        self.model_name = model_name
        self.report_dir_path = report_dir_path
        self.model_dir_path = os.path.join(model_dir_path, 'url')
        self.model_type = model_type
        self.model_file_path = os.path.join(self.model_dir_path, self.model_type + '.h5')
        self.model_report_dir_path = os.path.join(self.report_dir_path, self.model_name, 'url', self.model_type)
        
        if not os.path.isdir(self.model_report_dir_path):
            os.makedirs(self.model_report_dir_path)
        
        if not os.path.isdir(self.model_dir_path):
            os.makedirs(self.model_dir_path)
        
        # this is the default model we use

        self.num_input_tokens = None
        self.truncate_len = 100
        self.tokenizer = None
        self.model = None
        
        # Configuration. MUST BE Come from External CFG.
        self.cfg_LSTM = URLDEEP_LSTM_UNITS
        self.cfg_Conv1D = URLDEEP_CONV_FILTER
        self.cfg_Dense = URLDEEP_DENSE_UNITS 
        self.cfg_Epoch = URLDEEP_EPOCH
        self.cfg_MaxPooling = URLDEEP_MAXPOOLING_SIZE
        self.cfg_batch_size = URLDEEP_BATCH_SIZE

    #classifier model
    def make_model(self):
        """Make a model
        """
        print("Making the model...")
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.num_input_tokens, input_length=self.truncate_len,
                                 output_dim=int(self.num_input_tokens / 1.1)))  # shrinking factor

        self.model.add(Conv1D(filters = self.cfg_Conv1D, kernel_size=5, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size = self.cfg_MaxPooling))
        self.model.add(LSTM(self.cfg_LSTM, return_sequences=False, dropout=0.1))
        self.model.add(Dense(self.cfg_Dense))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    def load_trained_model(self):
        """ Load the trained model
        """
        self.model = load_model(self.model_file_path)
    
    def load_tokenizer(self):
        """ Load tokenizer information
        """
        with open(os.path.join(self.model_dir_path, 'url_tokenizer.pkl'), 'rb') as handle:
            extracted = pickle.load(handle)
        self.tokenizer = extracted['tokenizer']
        self.num_input_tokens = extracted['num_tokens']
        self.truncate_len = extracted['truncate_len']

    def fit(self, train_data):
        """ Fit the model
        """
        print("Fitting the model...")
        if not os.path.isdir(self.model_report_dir_path):
            os.makedirs(self.model_report_dir_path)
            
        #X_id = self.train_data['ID']
        random_state = 42
        val_size = 0.15 # 15%
       
        list_ID = []
        list_url = []
        list_label = []
        for tup in train_data.itertuples():
            list_ID.extend([tup.ID]*len(tup.url_list))
            list_url.extend(tup.url_list)
            list_label.extend([tup.label]*len(tup.url_list))

        df_url_data = pd.DataFrame({'ID':list_ID, 'url':list_url, 'label':list_label})
       
        X_encoded, Y_encoded = extract_encoding(df_url_data, URLDEEP_TRUNCATION_LENGTH, True, URL_ENCODING_MODE)        
        Xtrain, Xval, Ytrain, Yval = train_test_split(X_encoded, Y_encoded, test_size = val_size, random_state = random_state)
                
        checkpoint = ModelCheckpoint(self.model_file_path, monitor='val_loss', verbose=0,
                                     save_best_only=True, mode='min')

        early = EarlyStopping(monitor="val_loss",
                              mode="min",
                              patience=5)
        
        reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                                           patience=2,
                                           verbose=1, mode='min', epsilon=0.0001, cooldown=3,
                                           min_lr=1e-9)  # epsilon here is a good starting value

        history = self.model.fit(Xtrain, Ytrain, batch_size = self.cfg_batch_size, epochs = self.cfg_Epoch, verbose=1,
                                 validation_data=(Xval, Yval),
                                 callbacks=[checkpoint, early, reduceLROnPlat])
     
        
        print("Training the model on URL features done!")
        #return history
     
    def __get_class(self, prediction):
        """helper function: get class label based on prediction proba value
        """
        if prediction.shape[-1] > 1:
            prediction_class = prediction.argmax(axis=-1)
        else:
            prediction_class = (prediction > 0.5).astype('int32')

        return prediction_class

    def predict(self, test_data, mode=True):
        """ Use the model to make prediction on given input data
        """
        test_data_col = test_data.columns

        list_ID = []
        list_url = []
        list_score = []
        list_class = []

        # Flatten the data
        for it, row in test_data.iterrows():
            file_name = row['ID']
            url_list = row['url_list']

            if not url_list:
                # Handle cases with no URLs
                list_ID.append(file_name)
                list_url.append([])
                list_score.append(0)  # Default score
                list_class.append(0)  # Default class

            for url in url_list:
                list_ID.append(file_name)
                list_url.append(url)

        # Create DataFrame
        df_url_data = pd.DataFrame({
            'ID': list_ID,
            'url': list_url,
        })

        # Batch encode
        start_time = time.time()
        X_encoded = extract_encoding(df_url_data, self.tokenizer, self.truncate_len, False, URL_ENCODING_MODE)
        Ypred_url = self.model.predict(X_encoded, verbose=0)
        total_time = time.time() - start_time
        Ypred_class_url = self.__get_class(Ypred_url)

        # Combine predictions
        df_url_data['Predicted Score'] = Ypred_url.flatten()
        df_url_data['Predicted Class'] = Ypred_class_url.flatten()

        # Find index of max score per ID
        idx = df_url_data.groupby('ID')['Predicted Score'].idxmax()

        # Select rows with max scores
        res_agg = df_url_data.loc[idx, ['ID', 'Predicted Score', 'Predicted Class']].reset_index(drop=True)
        res_agg['Predicted Class'] = res_agg['Predicted Class'].astype(int)

        return res_agg, total_time
