from paddleocr import PaddleOCR
from paddleocr.ppocr.utils.logging import get_logger
import logging
from typing import Union
import numpy as np
from PIL import Image
logger = get_logger()
logger.setLevel(logging.ERROR)

# pip install paddlepaddle
# pip install paddleocr

class OCR(PaddleOCR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def ocr(self, img: Union[str, np.ndarray, Image.Image], det=True, rec=True, cls=False) -> str:
        '''
        Improve the ocr function to clean the detected text
        :param img:
        :param det:
        :param rec:
        :param cls:
        :return:
        '''
        most_fit_results = super().ocr(img, det=det, rec=rec, cls=cls)
        if len(most_fit_results):
            most_fit_results = most_fit_results[0]
            ocr_text = [line[1][0] for line in most_fit_results]
            detected_text = ' '.join(ocr_text)
        else:
            detected_text = ""
        return detected_text