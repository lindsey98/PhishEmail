from paddleocr import PaddleOCR
from paddleocr.ppocr.utils.logging import get_logger
import logging
logger = get_logger()
logger.setLevel(logging.ERROR)

# pip install paddlepaddle
# pip install paddleocr

class OCR(PaddleOCR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def ocr(self, img, det=True, rec=True, cls=False):
        most_fit_results = super().ocr(img, det=det, rec=rec, cls=cls)
        if len(most_fit_results):
            most_fit_results = most_fit_results[0]
            ocr_text = [line[1][0] for line in most_fit_results]
            detected_text = ' '.join(ocr_text)
        else:
            detected_text = ""
        return detected_text