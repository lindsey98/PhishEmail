import neuspell
from neuspell import SclstmChecker
import re
from ...utilities.data_utils import preserve_case
# checkpoints is downloaded from https://drive.google.com/drive/folders/1sn5zKRN0ZQnx9obu5I-LI7u05Jkezrue?usp=drive_link


class SCLSTMFixer:

    def __init__(self, model_id="/home/ruofan/git_space/PhishEmail/checkpoints/scrnn-probwordnoise"):
        self.model = SclstmChecker()
        self.model.from_pretrained(model_id)
        self.max_length = 512

    def __call__(self, paragraph: str):
        corrected_paragraph = self.model.correct(paragraph)
        return corrected_paragraph

