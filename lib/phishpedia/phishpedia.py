
import time
import sys
from datetime import datetime
import argparse
import os
import torch
from lib.phishpedia.configs import load_config
from tldextract import tldextract
import cv2
from lib.phishpedia.logo_recog import pred_rcnn, vis
from lib.phishpedia.logo_matching import check_domain_brand_inconsistency
import pickle
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class PhishpediaWrapper:
    _caller_prefix = "PhishpediaWrapper"
    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self):
        self._load_config()
        self._to_device()

    def _load_config(self):
        self.ELE_MODEL, self.SIAMESE_THRE, self.SIAMESE_MODEL, \
            self.LOGO_FEATS, self.LOGO_FILES, \
            self.DOMAIN_MAP_PATH = load_config()
        print(f'Length of reference list = {len(self.LOGO_FEATS)}')

    def _to_device(self):
        self.SIAMESE_MODEL.to(self._DEVICE)

    '''Phishpedia'''
    def test_orig_phishpedia(self, url, screenshot_path):
        # 0 for benign, 1 for phish, default is benign
        phish_category = 0
        pred_target = None
        matched_domain = None
        siamese_conf = None
        logo_recog_time = 0
        logo_match_time = 0
        print("Entering phishpedia")

        ####################### Step1: Logo detector ##############################################
        start_time = time.time()
        pred_boxes, _, _, _ = pred_rcnn(im=screenshot_path, predictor=self.ELE_MODEL)
        logo_recog_time = time.time() - start_time

        if pred_boxes is not None:
            pred_boxes = pred_boxes.detach().cpu().numpy()
        plotvis = vis(screenshot_path, pred_boxes)

        # If no element is reported
        if pred_boxes is None or len(pred_boxes) == 0:
            print('No element is detected, report as benign')
            return phish_category, pred_target, matched_domain,  plotvis, siamese_conf, pred_boxes, logo_recog_time, logo_match_time
        print('Entering siamese')

        ######################## Step2: Siamese (Logo matcher) ########################################
        start_time = time.time()
        pred_target, matched_domain, matched_coord, siamese_conf = check_domain_brand_inconsistency(logo_boxes=pred_boxes,
                                                                                  domain_map_path=self.DOMAIN_MAP_PATH,
                                                                                  model=self.SIAMESE_MODEL,
                                                                                  logo_feat_list=self.LOGO_FEATS,
                                                                                  file_name_list=self.LOGO_FILES,
                                                                                  url=url,
                                                                                  shot_path=screenshot_path,
                                                                                  ts=self.SIAMESE_THRE)
        logo_match_time = time.time() - start_time

        if pred_target is None:
            print('Did not match to any brand, report as benign')
            return phish_category, pred_target, matched_domain, plotvis, siamese_conf, pred_boxes, logo_recog_time, logo_match_time

        else:
            phish_category = 1
            # Visualize, add annotations
            cv2.putText(plotvis, "Target: {} with confidence {:.4f}".format(pred_target, siamese_conf),
                        (int(matched_coord[0] + 20), int(matched_coord[1] + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return phish_category, pred_target, matched_domain, plotvis, siamese_conf, pred_boxes, logo_recog_time, logo_match_time


if __name__ == '__main__':



    '''update domain map'''
    # with open('./lib/phishpedia/models/domain_map.pkl', "rb") as handle:
    #     domain_map = pickle.load(handle)
    #
    # domain_map['Tencent'] = ['tencent', 'tencentcloud', 'qq']
    # domain_map['baidu'] = ['baidu']
    # domain_map['Shanghai Jiao Tong University'] = ['sjtu']
    # domain_map['weibo'] = ['weibo']
    #
    # with open('./lib/phishpedia/models/domain_map.pkl', "wb") as handle:
    #     pickle.dump(domain_map, handle)

    '''run'''
    while True:
        today = datetime.now().strftime('%Y%m%d')
        # today = "20231221"
        request_dir = f'./datasets/sjtu_phish/{today}'
        results_dir = f'./datasets/sjtu_phish_reported/{today}'

        phishpedia_cls = PhishpediaWrapper()
        result_txt = f'./datasets/{today}_results.txt'

        os.makedirs(results_dir, exist_ok=True)

        for folder in tqdm(os.listdir(request_dir)):
            # if folder != '004eee200a6c4c302d45f17028abc68e_cloudflare-ipfs.com':
            #     continue
            html_path = os.path.join(request_dir, folder, "html.txt")
            screenshot_path = os.path.join(request_dir, folder, "shot.png")
            info_path = os.path.join(request_dir, folder, 'info.json')

            if not os.path.exists(screenshot_path):
                continue

            url = eval(open(info_path).read())['url']

            if os.path.exists(result_txt) and url in open(result_txt).read():
                continue

            phish_category, pred_target, matched_domain, \
                        plotvis, siamese_conf, pred_boxes, \
                        logo_recog_time, logo_match_time = phishpedia_cls.test_orig_phishpedia(url, screenshot_path)

            try:
                with open(result_txt, "a+", encoding='ISO-8859-1') as f:
                    f.write(folder + "\t")
                    f.write(url + "\t")
                    f.write(str(phish_category) + "\t")
                    f.write(str(pred_target) + "\t")  # write top1 prediction only
                    f.write(str(matched_domain) + "\t")
                    f.write(str(siamese_conf) + "\t")
                    f.write(str(round(logo_recog_time, 4)) + "\t")
                    f.write(str(round(logo_match_time, 4)) + "\n")
            except UnicodeError:
                with open(result_txt, "a+", encoding='utf-8') as f:
                    f.write(folder + "\t")
                    f.write(url + "\t")
                    f.write(str(phish_category) + "\t")
                    f.write(str(pred_target) + "\t")  # write top1 prediction only
                    f.write(str(matched_domain) + "\t")
                    f.write(str(siamese_conf) + "\t")
                    f.write(str(round(logo_recog_time, 4)) + "\t")
                    f.write(str(round(logo_match_time, 4)) + "\n")
            if phish_category:
                os.makedirs(os.path.join(results_dir, folder), exist_ok=True)
                cv2.imwrite(os.path.join(results_dir, folder, "predict.png"), plotvis)


    # import matplotlib.pyplot as plt
    # plt.imshow(cropped)
    # plt.savefig('./debug.png')