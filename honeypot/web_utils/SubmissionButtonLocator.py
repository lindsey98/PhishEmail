
from phishintention.src.AWL_detector import element_config
from .PhishIntentionWrapper import run_object_detection, encoding2array
import torch

class SubmissionButtonLocator():

    def __init__(self, button_locator_weights_path, button_locator_config):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _, self.BUTTON_SUBMISSION_MODEL = element_config(rcnn_weights_path=button_locator_weights_path,
                                                         rcnn_cfg_path=button_locator_config,
                                                         device=device)

    def return_submit_button(self, screenshot_encoding):
        screenshot_img_arr = encoding2array(screenshot_encoding)  # RGB2BGR

        pred_classes, pred_boxes, pred_scores = run_object_detection(img_arr=screenshot_img_arr,
                                                                     model=self.BUTTON_SUBMISSION_MODEL)
        if pred_boxes is None or len(pred_boxes) == 0:
            return None
        pred_boxes = pred_boxes.detach().cpu().numpy()
        return pred_boxes