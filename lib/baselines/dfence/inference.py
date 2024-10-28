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
## Load baseline models
model_directory = os.path.join(MODEL_DIR_PATH)
STRUCTURAL_MODEL = StructuralModel("", STRUCTURAL_MODEL_TYPE, model_directory, REPORT_DIR_PATH)
STRUCTURAL_MODEL.load_trained_model()

TEXT_MODEL = TextModel("", TEXT_MODEL_TYPE, model_directory, REPORT_DIR_PATH)
TEXT_MODEL.load_trained_model()

URL_MODEL = URLDeep("", URL_MODEL_TYPE, model_directory, REPORT_DIR_PATH)
URL_MODEL.load_trained_model()
URL_MODEL.load_tokenizer()

META_MODEL = MetaModel("", META_MODEL_TYPE, model_directory, REPORT_DIR_PATH)
META_MODEL.load_trained_model()

def test(email_dir):
    """test with labeled files, it can be: 1 mal and 1 benign, 1 benign only or 1 mal only
    """
    test_dataset = EmailDataset(email_dir)
    total_time = 0
    start_time = time.time()
    df_header_full, df_html_full, df_text_full, df_URL_full= \
        processEmails(test_dataset, int(FEATURE_EXTRACTION_BATCH_SIZE))
    total_time += time.time() - start_time

    df_structural_full = df_header_full.merge(df_html_full, on='ID', how='left')

    start_time = time.time()
    structural_pred = STRUCTURAL_MODEL.predict(df_structural_full)
    text_pred = TEXT_MODEL.predict(df_text_full)
    url_pred = URL_MODEL.predict(df_URL_full)
    meta_test_data = prepMetaFeatures(structural_pred, text_pred, url_pred)
    res = META_MODEL.predict(meta_test_data)
    total_time += time.time() - start_time

    pred_confidence, pred_class = res['Predicted Score'].tolist(), res['Predicted Class']

    return pred_confidence, pred_class, total_time

if __name__ == '__main__':
    desc_folder = './datasets/sjtu_phish/email_195.eml'
    test(desc_folder)
