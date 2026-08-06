import logging
from typing import Union

import numpy as np
from paddleocr import PaddleOCR
from paddleocr.ppocr.utils.logging import get_logger
from PIL import Image

logger = get_logger()
logger.setLevel(logging.ERROR)


class OCR(PaddleOCR):
    def __init__(self, **kwargs):
        # Angle classification is unused (ocr() is always called with cls=False),
        # so don't load it. Pass use_gpu=True (and other PaddleOCR kwargs) through
        # for a further speedup when a GPU build of paddle is installed.
        kwargs.setdefault("use_angle_cls", False)
        super().__init__(**kwargs, use_space_char=True)

    def ocr(self, img: Union[str, np.ndarray, Image.Image], det=True, rec=True, cls=False) -> str:
        """
        Improve the ocr function to clean the detected text
        :param img: image path or image in ndarray or in PIL.Image
        :param det: character detection on?
        :param rec: character recognition on?
        :param cls:
        :return: detected_text
        """

        most_fit_results = super().ocr(img, det=det, rec=rec, cls=cls)
        if len(most_fit_results):
            most_fit_results = most_fit_results[0]
            ocr_text = [line[1][0] for line in most_fit_results]
            detected_text = " ".join(ocr_text)
        else:
            detected_text = ""
        return detected_text
