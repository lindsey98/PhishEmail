import asyncio
from PIL import Image
import io
import numpy as np
import os
import sys
import email
import email.header
import quopri
import hashlib
import base64
import re
import pdfkit
from tqdm import tqdm
import pdf2image
from pdf2image import convert_from_path
from lib.data import EmailDataset
from lib.utilities import Logger
from email.message import Message
from typing import Union, Tuple
from lib.data.OCR import OCR
from lib.utilities.data_utils import normalization
import subprocess

# pip install pdf2image
# sudo apt-get install poppler-utils  # For Linux (required by pdf2image)
# pip install git+https://github.com/akhilnarang/python-pdfkit.git@add-timeout-option
# sudo apt-get install wkhtmltopdf

class RenderDataset(EmailDataset):
    _CallerPrefix = "Dataset Loader (render the eml)"

    def __init__(self, root_path, ocr_model: OCR, dumpDir: str="temp", translate_on=False, save_imgs=False):
        super().__init__(root_path, translate_on)
        self.ocr_model = ocr_model
        self.dumpDir = dumpDir
        os.makedirs(dumpDir, exist_ok=True)
        self.save_imgs = save_imgs

    @staticmethod
    def pdfs_to_combined_image(pdf_list, dpi=200):
        images = []

        # Convert each PDF to images
        for pdf in pdf_list:
            try:
                pages = convert_from_path(pdf, dpi=dpi)
                images.extend(pages)  # Append all pages as images
            except pdf2image.exceptions.PDFPageCountError as e:
                continue

        # Combine the images
        if len(images):
            combo = RenderDataset.concat_images(images)
            return combo
        return False

    @staticmethod
    def concat_images(images):
        bgColor = (255, 255, 255)
        widths, heights = zip(*(i.size for i in images))
        new_width = max(widths)
        new_height = sum(heights)
        new_im = Image.new('RGB', (new_width, new_height), color=bgColor)
        offset = 0
        for im in images:
            x = 0
            new_im.paste(im, (x, offset))
            offset += im.size[1]
        return new_im

    @staticmethod
    def kill_wkhtmltopdf_processes(): # fixme: wkhtmltopdf never properly close
        try:
            subprocess.run(['pkill', 'wkhtmltopdf'], check=True)
            print("All wkhtmltopdf processes have been terminated.")
        except subprocess.CalledProcessError:
            print("No wkhtmltopdf processes found or failed to terminate.")

    # Usage
    def render_eml(self, data:bytes, dumpName="new") -> (Union[bool, str], str):
        '''

        :param data:
        :param dumpDir:
        :param dumpName:
        :return:
        '''
        textTypes = ['text/plain', 'text/html']
        imageTypes = ['image/gif', 'image/jpeg', 'image/png', "application/pdf"]
        imgkitOptions = {'load-error-handling': 'ignore',
                        'load-media-error-handling': 'ignore',
                         'enable-local-file-access': "",
                         "quiet": None}
        imagesList = []

        msg = email.message_from_bytes(data)
        html_content = ""

        for part in msg.walk():
            mimeType = part.get_content_type()
            if part.is_multipart():
                Logger.spit('[INFO] Multipart found, continue', debug=True)
                continue

            Logger.spit('[INFO] Found MIME part: %s' % mimeType, debug=True)
            if mimeType in textTypes:
                charset = part.get_content_charset('utf-8')
                raw_email_content = part.get_payload(decode=True)  # decoding is handled here
                transfer_encoding = part.get('Content-Transfer-Encoding', '').lower()
                if raw_email_content:
                    if transfer_encoding == 'quoted-printable':
                        payload = self.decode_quoted_printable(raw_email_content, charset)
                    else:
                        try:
                            payload = raw_email_content.decode(charset, 'replace')
                        except LookupError:  # in case charset is not recognized
                            payload = raw_email_content.decode('utf-8', 'replace')
                else:
                    payload = raw_email_content  # handle cases where payload is not encoded

                # Cleanup dirty characters
                dirtyChars = ['\n', '\\n', '\t', '\\t', '\r', '\\r']
                if isinstance(payload, bytes):
                    payload = str(payload)
                    for char in dirtyChars:
                        payload = payload.replace(char, '')

                html_content += payload

                if mimeType == 'text/html':
                    # Generate MD5 hash of the payload
                    m = hashlib.md5()
                    m.update(payload.encode('utf-8'))
                    imagePath = m.hexdigest() + '.pdf'
                    try:
                        pdfkit.from_string(payload, os.path.join(self.dumpDir, imagePath), options=imgkitOptions, timeout=5)
                        Logger.spit('[INFO] Decoded %s' % imagePath, debug=True)
                        imagesList.append(os.path.join(self.dumpDir, imagePath))
                    except Exception as e:
                        Logger.spit('[WARNING] Decoding this MIME part returned error', debug=True)

            elif mimeType in imageTypes:
                payload = part.get_payload(decode=False)
                imgdata = base64.b64decode(payload)
                m = hashlib.md5()
                m.update(payload.encode('utf-8'))
                imagePath = m.hexdigest() + '.pdf'
                try:
                    with open(os.path.join(self.dumpDir, imagePath), 'wb') as f:
                        f.write(imgdata)
                    Logger.spit('[INFO] Decoded %s' % imagePath, debug=True)
                    imagesList.append(os.path.join(self.dumpDir, imagePath))
                except:
                    Logger.spit('[WARNING] Decoding this MIME part returned error', debug=True)


        resultImage = os.path.join(self.dumpDir, f'{dumpName}.png')
        if len(imagesList) > 0:
            combo = self.pdfs_to_combined_image(imagesList)
            for i in imagesList:  # Clean up temporary images
                if os.path.exists(i):
                    os.remove(i)
            if combo is False:
                return False, html_content
            combo.save(resultImage)
            return resultImage, html_content
        else:
            return False, html_content


    def __getitem__(self, idx):
        email_file_path = self.file_list[idx]
        email_content = self.load_email_content(email_file_path)

        headers = str(email_content._headers)

        sender_name, sender_address = self.extract_sender(email_content)
        to_names, to_addresses = self.extract_recipients(email_content)
        reply_to_address = self.extract_reply_to_address(email_content)
        if reply_to_address is None:
            reply_to_address = sender_address

        subject = self.extract_subject(email_content)
        subject = self.auto_translate(subject)
        sender_name = normalization(sender_name)
        subject = normalization(subject)

        text_content, html_content = self.extract_text_content(email_content)
        if len(text_content) < 100 or len(text_content) > 512*2:  ## too short or too long => extract OCR text from HTML
            with open(email_file_path, "rb") as f:
                msg_bytes = f.read()
            resultImage, html_content = self.render_eml(msg_bytes, dumpName=os.path.basename(email_file_path))

            if resultImage is not False:
                text_content_from_ocr = self.ocr_model.ocr(resultImage)
                if not self.save_imgs:
                    os.remove(resultImage)
                text_content = text_content_from_ocr + text_content


        text_content = self.remove_prev_messages(text_content)
        text_content = self.clean_text_content(text_content)
        text_content = self.auto_translate(text_content)
        text_content = normalization(text_content)
        text_content = self.unfragment_text(text_content)

        return email_file_path, \
               (sender_name, sender_address), \
                (to_names, to_addresses), \
                reply_to_address, \
                subject, \
                text_content, \
                headers



if __name__ == '__main__':
    rootDir = "./datasets/phishpot"
    dumpDir = "./datasets/phishpot_imgs/"
    ocr_model = OCR()
    dataset = RenderDataset(rootDir, ocr_model=ocr_model, dumpDir=dumpDir)

    for i in range(len(dataset)):
        email_file_path, (sender_name, sender_address), \
        (to_names, to_addresses), reply_to_address, \
        subject, email_body_text, header =   dataset[i]

        print()
