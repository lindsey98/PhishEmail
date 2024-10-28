import warnings
warnings.filterwarnings('ignore')
import argparse
from lib.baselines.dfence.getFeatures import processEmails
from lib.baselines.dfence.utils import *
from lib.baselines.dfence.StructuralModel import StructuralModel
from lib.baselines.dfence.TextModel import TextModel
from lib.baselines.dfence.URLDeepModel import URLDeep
from lib.baselines.dfence.MetaModel import MetaModel
import pandas as pd
from lib.data.Dataset import EmailDataset
'''
Main function
'''

cfg = configparser.ConfigParser()
cfg.read('./lib/baselines/dfence/conf.cfg')

MODEL_DIR_PATH = os.path.abspath(cfg.get('env', 'model_dir_path'))
FEATURE_DIR_PATH = os.path.abspath(cfg.get('env', 'feature_dir_path'))
REPORT_DIR_PATH = os.path.abspath(cfg.get('env', 'report_dir_path'))
FEATURE_EXTRACTION_BATCH_SIZE = cfg.get('env', 'feature_extraction_batch_size')
TRAIN_TEST_SPLIT = cfg.get('env', 'train_test_split')
STRUCTURAL_MODEL_TYPE = cfg.get('env', 'structural_model_type')
TEXT_MODEL_TYPE = cfg.get('env', 'text_model_type')
URL_MODEL_TYPE = cfg.get('env', 'url_model_type')
META_MODEL_TYPE = cfg.get('env', 'meta_model_type')


def test(email_dir, model_name=''):
    """test with labeled files, it can be: 1 mal and 1 benign, 1 benign only or 1 mal only
    """
    console_log('Start testing...')
    test_dataset = EmailDataset(email_dir)
    model_directory = os.path.join(MODEL_DIR_PATH, model_name)

    console_log("Start extracting features from emails...")
    df_header_full, df_html_full, df_text_full, df_URL_full= \
        processEmails(test_dataset, int(FEATURE_EXTRACTION_BATCH_SIZE))

    df_structural_full = df_header_full.merge(df_html_full, on='ID', how='left')

    ## Load pretrained models
    structural_model = StructuralModel(model_name, STRUCTURAL_MODEL_TYPE, model_directory, REPORT_DIR_PATH)
    structural_model.load_trained_model()

    text_model = TextModel(model_name, TEXT_MODEL_TYPE, model_directory, REPORT_DIR_PATH)
    text_model.load_trained_model()

    url_model = URLDeep(model_name, URL_MODEL_TYPE, model_directory, REPORT_DIR_PATH)
    url_model.load_trained_model()
    url_model.load_tokenizer()

    structural_pred = structural_model.predict(df_structural_full)
    text_pred = text_model.predict(df_text_full)
    url_pred = url_model.predict(df_URL_full)

    meta_test_data = prepMetaFeatures(structural_pred, text_pred, url_pred)
    meta_model = MetaModel(model_name, META_MODEL_TYPE, model_directory, REPORT_DIR_PATH)
    meta_model.load_trained_model()
    res = meta_model.predict(meta_test_data)
    pred_confidence, pred_class = res['Predicted Score'].tolist(), res['Predicted Class'].tolist()

    return pred_confidence, pred_class

if __name__ == '__main__':
    desc_folder = './datasets/sjtu_phish/email_195.eml'
    test(desc_folder)
