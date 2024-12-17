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

def feature_importance_viz():
    from xgboost import plot_importance
    import matplotlib.pyplot as plt
    structural_features = ['plain_sec_cnt', 'html_sec_cnt',
                           'image_sec_cnt', 'app_sec_cnt',
                           'plain_html_ratio', 'std_hdr_cnt',
                           'user_agent_exist', 'x_mailer_exist',
                           'mime_ver', 'mime_ver_case_var',
                           'is_from_bracketed', 'is_to_bracketed',
                           'section_boundary_len',
                           'section_boundary_startwith_equal',
                           'section_boundary_has_equal',
                           'section_boundary_has_underscore',
                           'section_boundary_is_hex', 'section_boundary_is_num',
                           'section_boundary_has_dot',
                           'section_boundary_has_spec', 'cont_type_char_set',
                           'num_unique_charset', 'msg_id_boundary',
                           'msg_id_len', 'msg_id_is_hex',
                           'msg_id_is_num', 'msg_id_has_dot',
                           'msg_id_has_spec', 'msg_id_domain_is_hex',
                           'msg_id_domain_is_num', 'msg_id_domain_has_dot',
                           'msg_id_domain_has_spec', 'num_from',
                           'num_sender', 'num_to', 'num_reply_to',
                           'num_in_reply_to', 'num_received',
                           'num_to_rcpt', 'num_cc_rcpt', 'num_bcc_rcpt',
                           'ratio_same_domain_from_to',
                           'ratio_same_domain_from_replyto',
                           'ratio_same_domain_from_return',
                           'ratio_same_domain_from_cc',
                           'ratio_same_domain_from_bcc',
                           'ratio_unique_domain_from', 'ratio_unique_domain_to',
                           'ratio_unique_domain_sender',
                           'ratio_unique_domain_replyto',
                           'ratio_unique_domain_return',
                           'sensitive_word_ratio', 'len_html_text',
                           'len_html_css', 'if_backward_rendering',
                           'if_js', 'dom_depth', 'dom_leaf_node_count',
                           'dom_leaf_node_type', 'dom_leaf_node_mean',
                           'dom_leaf_node_std', 'num_links',
                           'unique_domain_ratio']
    # Sort feature importances and select top-k
    top_k = 3
    sorted_indices = np.argsort(STRUCTURAL_MODEL.model.feature_importances_)[::-1]  # Sort descending
    top_features = [structural_features[i] for i in sorted_indices[:top_k]]
    top_importances = STRUCTURAL_MODEL.model.feature_importances_[sorted_indices[:top_k]]

    plt.figure(figsize=(16, 4.5))
    plt.barh(top_features, top_importances, color="skyblue")  # Reverse to have top feature on top
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=20, fontweight='bold')
    plt.gca().invert_yaxis()  # Ensure the highest importance feature is at the top
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('./debug.png')

def test(email_dir):
    """test with labeled files, it can be: 1 mal and 1 benign, 1 benign only or 1 mal only
    """
    test_dataset = EmailDataset(email_dir)
    total_time = 0
    df_header_full, df_html_full, df_text_full, df_URL_full, extraction_time = \
        processEmails(test_dataset, int(FEATURE_EXTRACTION_BATCH_SIZE))
    total_time += extraction_time

    df_structural_full = df_header_full.merge(df_html_full, on='ID', how='left')

    structural_pred, structural_time = STRUCTURAL_MODEL.predict(df_structural_full)
    text_pred, text_time = TEXT_MODEL.predict(df_text_full)
    url_pred, url_time = URL_MODEL.predict(df_URL_full)
    total_time += structural_time + text_time + url_time

    meta_test_data = prepMetaFeatures(structural_pred, text_pred, url_pred)
    res, meta_time = META_MODEL.predict(meta_test_data)
    total_time += meta_time

    pred_confidence, pred_class = res['Predicted Score'].tolist(), res['Predicted Class']

    return pred_confidence, pred_class, total_time

