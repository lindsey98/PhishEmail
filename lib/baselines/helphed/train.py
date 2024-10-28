import os
from pathlib import Path
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, f1_score
from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def fit_multiple_estimators(classifiers, X_list, y, sample_weights = None):

    # Convert the labels `y` using LabelEncoder, because the predict method is using index-based pointers
    # which will be converted back to original data later.
    le_ = LabelEncoder()
    le_.fit(y)
    transformed_y = le_.transform(y)

    # Fit all estimators with their respective feature arrays
    estimators_ = [clf.fit(X, y) if sample_weights is None else clf.fit(X, y, sample_weights) for clf, X in zip([clf for _, clf in classifiers], X_list)]

    return estimators_, le_


# info gain feature selection
def ig_select_features(X_train, y_train):
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    return X_train_fs, fs

if __name__ == '__main__':
    base = Path('./datasets')
    df_benign_enron = pq.read_table(base / 'benign_train_enron.parquet').to_pandas()
    df_phish_nazario = pq.read_table(base / 'phish_train_nazario.parquet').to_pandas()
    df_benign_enron['dataset'] = 'train'
    df_phish_nazario['dataset'] = 'train'
    df = pd.concat([df_benign_enron, df_phish_nazario], ignore_index=True)

    ##### Voting Model #####
    #Word2Vec features
    count = 0
    w2v = df['Word2vec']
    vec=np.array(w2v.to_list())
    df_new = pd.DataFrame(vec)

    #add a new column with the label in the new df
    df_new['label'] = df['label']

    #training set only w2v features
    df_new['dataset'] = df['dataset']
    X_train = df_new[df_new['dataset'] == 'train'].drop(['label', 'dataset'], axis=1)
    y_train = df_new[df_new['dataset'] == 'train']['label']

    ############# Word2Vec Feature selection ###################
    features= []
    for i in X_train:
        features.append(i)

    FS = []
    df2 = []
    X_train_fs, fs = ig_select_features(X_train, y_train)
    for i in range(len(fs.scores_)):
        if fs.scores_[i] > 0.01:
            count += 1
            FS.append(features[i])
    for i in FS:
        df2.append(df_new[i])

    df2 = np.array(df2)
    df2 = pd.DataFrame(df2)
    df2 = df2.transpose()

    ######################## Content-based features training remove text-based features and unwanted content-based features
    df = df.drop(['scripts', 'forms', 'nports', 'link_images', 'Word2vec'], axis =1) #Converting the encoding column to categorical - it assigns an int on each encoding-name
    df['encoding'] = df['encoding'].astype('category')
    enc_encode = LabelEncoder()
    df['encoding'] = enc_encode.fit_transform(df.encoding)
    #########################################

    #Concat word2vec with content-based features TRAINING
    df2 = pd.concat([df, df2], axis=1)

    X_train = df2[df2['dataset'] == 'train'].drop(['label', 'dataset'], axis=1)
    y_train = df2[df2['dataset'] == 'train']['label']
    X_train.columns = X_train.columns.astype(str)

    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_knn.fit(X_train.iloc[:, 18:], y_train)

    clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=20, min_samples_leaf=3, min_samples_split=10, max_features='sqrt')
    clf_dt.fit(X_train, y_train)

    #Divide the dataset between content and word based
    X_train1, X_train2 = X_train.iloc[:, 0:18], X_train.iloc[:,18:]
    X_train_list = [X_train1, X_train2]
    # Make sure the number of estimators here are equal to number of different feature datas
    classifiers = [('dt',  clf_dt), ('knn', clf_knn)]

    fitted_estimators, label_encoder = fit_multiple_estimators(classifiers, X_train_list, y_train)
    for idx, (name, estimator) in enumerate(zip(classifiers, fitted_estimators)):
        model_path = f'./checkpoints/helphed_models/{name[0]}_model.pkl'  # name[0] should contain the identifier of the classifier
        with open(model_path, 'wb') as file:
            pickle.dump(estimator, file)
        print(f"Saved {name[0]} model to {model_path}")

    # Save the label encoder
    label_encoder_path = './checkpoints/helphed_models/label_encoder.pkl'
    with open(label_encoder_path, 'wb') as file:
        pickle.dump(label_encoder, file)
    print(f"Saved label encoder to {label_encoder_path}")

    # y_pred_voting = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_test_list)

    ##### Stacking Model #####
    pipe1 = make_pipeline(ColumnSelector(cols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18)),
                          DecisionTreeClassifier(criterion='entropy', max_depth=100, min_samples_leaf=3, min_samples_split=2, max_features='sqrt'))

    pipe2 = make_pipeline(ColumnSelector(cols=list(range(19, X_train.shape[1]))),
                          KNeighborsClassifier(n_neighbors=3))

    sclf = StackingClassifier(classifiers=[pipe1, pipe2],
                              meta_classifier=MLPClassifier(activation='tanh', hidden_layer_sizes=(20, 40, 20), solver='adam', alpha=0.5, learning_rate='adaptive', max_iter=1000))

    sclf = sclf.fit(X_train, y_train)

    # Save the model
    model_path = './checkpoints/helphed_models/stacking_model.pkl'
    with open(model_path, 'wb') as file:
        pickle.dump(sclf, file)
    print(f"Saved stacking model to {model_path}")
