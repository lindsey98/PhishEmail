from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os
import pandas as pd
import pickle
import time
'''
Structural model (HTML+headers) for training and prediction

'''

class StructuralModel(object):
    """This class implements our model that builds on structural features (html + header) using xgboost classifier
    """
    def __init__(self, model_name, model_type, model_dir_path, report_dir_path):
        """ Initialize the object: model architecture is also defined here
         """
        self.model_name = model_name
        self.report_dir_path = report_dir_path
        self.model_dir_path = os.path.join(model_dir_path, 'struct')
        self.model_type = model_type
        self.model_file_path = os.path.join(self.model_dir_path, self.model_type + '.h5')
        self.scaler_file_path = os.path.join(self.model_dir_path, 'scaler_' + self.model_type+'.pkl')
        self.model_report_dir_path = os.path.join(self.report_dir_path, self.model_name, 'struct', self.model_type)
        
        if not os.path.isdir(self.model_report_dir_path):
            os.makedirs(self.model_report_dir_path)
        
        if not os.path.isdir(self.model_dir_path):
            os.makedirs(self.model_dir_path)
            
        # this is the default model we use
        self.model = None
        self.scaler = None

    #classifier model
    def make_model(self):
        """Make a model
        """
        if self.model_type == 'xgb':
            self.model = xgb.XGBClassifier()
            
    def load_trained_model(self):
        """ Load the trained model and scaler
        """
        self.model = pickle.load(open(self.model_file_path, 'rb'))
        self.scaler = pickle.load(open(self.scaler_file_path, 'rb'))

    def fit(self, train_data ):
        """ Fit the model      
        """
        features_col = [col for col in train_data if col.startswith('features')]

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(train_data[features_col])
        self.model.fit(X_train_scaled, train_data['label'])
            
        #save model and scaler for future use
        with open(self.model_file_path, 'wb') as f_write:
            pickle.dump(self.model, f_write, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.scaler_file_path, 'wb') as f_write:
            pickle.dump(self.scaler, f_write, protocol=pickle.HIGHEST_PROTOCOL)
       

    def __get_class(self, prediction):
        """
        helper function: get class label based on prediction proba value
        """
        if prediction.shape[-1] > 1:
            prediction_class = prediction.argmax(axis=-1)
        else:
            prediction_class = (prediction > 0.5).astype('int32')

        return prediction_class


    def predict(self, test_data):
        """ Use the model to make prediction on given input data
           Assuming test_data is in format ['ID', features..., 'label']
        """
                
        test_data_col = test_data.columns
        features_col = [col for col in test_data if col.startswith('features')]
        
        Xtest = test_data[features_col]
        start_time = time.time()
        Xtest = self.scaler.transform(Xtest)

        Ypred = self.model.predict_proba(Xtest)
        total_time = time.time() - start_time
        Ypred_class = self.__get_class(Ypred)
        Ypred = Ypred[:, 1]

        res = pd.DataFrame({'ID': test_data['ID'], 'Predicted Score': Ypred, 'Predicted Class': Ypred_class})
        res = res.astype({"Predicted Class": int})

        return res, total_time