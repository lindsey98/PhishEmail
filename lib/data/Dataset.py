import functools
import os
import quopri
import re
import time
import unicodedata
from email import message_from_string
from email.header import decode_header
from email.message import Message
from email.utils import getaddresses
from typing import Dict, List, Optional, Set, Tuple, Union

import bs4
import langdetect
from bs4 import BeautifulSoup, Comment, Doctype, NavigableString, ProcessingInstruction
from deep_translator import GoogleTranslator
from langdetect import detect_langs
from tldextract import tldextract
from torch.utils.data import Dataset

from ..utilities import Logger
from ..utilities.data_utils import normalization


class EmailDataset(Dataset):
    _CallerPrefix = "Dataset Loader"

    def __init__(self, root_path, translate_on: bool = False):

        self.file_list: List[str] = []
        supported_extensions: List[str] = [".eml", ".txt"]

        if isinstance(root_path, list):
            for filename in root_path:
                if any(filename.endswith(ext) for ext in supported_extensions):
                    self.file_list.append(filename)
        elif os.path.isdir(root_path):  # Check if root_path is a directory
            for root, dirs, files in os.walk(root_path):
                for filename in files:
                    if any(filename.endswith(ext) for ext in supported_extensions):
                        full_path = os.path.join(root, filename)  # Join root with filename
                        self.file_list.append(full_path)
        elif os.path.isfile(root_path) and any(root_path.endswith(ext) for ext in supported_extensions):
            self.file_list.append(root_path)  # Add the file directly if it ends with .eml or .txt
        else:
            raise NotImplementedError

        proxy_set: Optional[str] = os.getenv("http_proxy", None)
        if proxy_set is not None:
            self.translator = GoogleTranslator(
                source="auto", target="en", proxies={"https": proxy_set, "http": proxy_set}
            )
        else:
            self.translator = GoogleTranslator(source="auto", target="en")

        self.translate_on: bool = translate_on

    def __len__(self):
        return len(self.file_list)

    @staticmethod
    def domain_parsing(address: Union[None, float, str, Set[str], List[str]]) -> Set[str]:
        """
        Convert address to domain.suffix format
        :param address:
        :return: set of domain.tld
        """

        url_pattern = re.compile(r"(https?://[^)]+)")
        parsed_domains = set()

        if isinstance(address, float) or address is None:
            return parsed_domains
        elif isinstance(address, str):
            match = url_pattern.search(address)
            if match:  # is a URL
                domain = tldextract.extract(address).domain + "." + tldextract.extract(address).suffix
                parsed_domains.add(domain)
            else:  # is not a URL but an email address
                invalid_address = "@" not in address
                if not invalid_address:
                    domain = address.split("@")[-1]
                    domain = tldextract.extract(domain).domain + "." + tldextract.extract(domain).suffix
                    parsed_domains.add(domain)
            return parsed_domains
        else:
            for add in address:
                ind_domain = EmailDataset.domain_parsing(add)
                parsed_domains = parsed_domains.union(ind_domain)

        return parsed_domains

    @staticmethod
    def load_email_content(email_file_path: str) -> Message:
        with open(email_file_path, "r", encoding="utf-8", errors="ignore") as file:
            email_content = file.read()
        email_content = message_from_string(email_content)
        return email_content

    @staticmethod
    def decode_header(header_value) -> str:
        header_parts = decode_header(header_value)
        try:
            header_parts_decoded = "".join(
                part if isinstance(part, str) else part.decode(charset or "utf-8", "replace")
                for part, charset in header_parts
            )
        except LookupError:
            header_parts_decoded = "".join(str(part) for part, charset in header_parts)

        return header_parts_decoded

    @staticmethod
    def extract_invisible_chars(text: str) -> Dict[str, str]:
        return {
            char: unicodedata.name(char, "UNKNOWN")
            for char in text
            if not char.isprintable() or unicodedata.category(char).startswith("C")
        }

    @staticmethod
    def shrink_urls(text_content):
        """
        Shrink long URL into its protocal://domain format
        :param text_content:
        :return: text_content with all URLs being shrunk
        """

        url_pattern = re.compile(
            r"(?P<protocol>https?:)\/\/"  # Capture protocol (http: or https:)
            r"(?P<domain>(?:[\w-]+\.)+[\w-]+\.[a-zA-Z]{2,})"  # Capture domain with optional subdomains
            r"(?:\/\S*)?",  # Optional path, query parameters, etc.
            re.IGNORECASE,
        )

        # Define the replacement function
        def shrink_url(match):
            protocol = match.group("protocol")
            domain = match.group("domain")
            return f"{protocol}//{domain}"

        text_content = url_pattern.sub(shrink_url, text_content)
        return text_content

    @staticmethod
    def clean_text_content(text_content: str) -> str:
        # Remove text within angle brackets (likely comments or invisible)
        text_content = re.sub(r"<[^>]*>", "", text_content)

        # Replace multiple newlines with a single newline, and multiple spaces with a single space
        text_content = re.sub(r"\n{2,}", "\n", text_content)
        text_content = re.sub(r"\s+", " ", text_content)

        # Replace multiple consecutive periods with a single period
        text_content = re.sub(r"\.{2,}", ".", text_content)

        # Replace non-breaking spaces and soft hyphens
        text_content = re.sub(r"\xa0|&nbsp;", " ", text_content)
        text_content = re.sub(r"\xad|&shy;", "", text_content)

        # Remove zero-width characters
        zero_width_pattern = r"[\u200B-\u200D\u2060\uFEFF\u034F\u17B4\u17B5]"
        text_content = re.sub(zero_width_pattern, "", text_content)

        # Normalize the text
        text_content = unicodedata.normalize("NFKD", text_content)

        # Remove remaining invisible characters
        invisible_chars = EmailDataset.extract_invisible_chars(text_content)
        text_content = "".join(char for char in text_content if char not in invisible_chars)

        # Shrink URLs
        text_content = EmailDataset.shrink_urls(text_content)
        return text_content

    @staticmethod
    def decode_quoted_printable(encoded_str: bytes, charset: str = "utf-8") -> str:
        """
        Decode a quoted-printable encoded string to its original form using the specified charset.
        :param encoded_str:
        :param charset:
        :return:
        """
        try:
            # Decode the quoted-printable encoded string
            decoded_bytes = quopri.decodestring(encoded_str)
            # Decode the bytes to string using the specified charset
            decoded_str = decoded_bytes.decode(charset, "replace")
        except LookupError:
            # Fallback to utf-8 if charset is not recognized
            decoded_str = decoded_bytes.decode("utf-8", "replace")
        except Exception:
            Logger.spit(
                "Failed to decode the email with exception", debug=True, caller_prefix=EmailDataset._CallerPrefix
            )
            decoded_str = encoded_str.decode(charset, "replace")  # Fallback to raw decoding
        return decoded_str

    @staticmethod
    def remove_prev_messages(text_content: str) -> str:
        """
        Remove previous conversations
        :param text_content:
        :return: text_content with Re:, Fwd: part being removed
        """

        reply_pattern = re.compile(r"On.*wrote:")
        forward_pattern = re.compile(r"^-{2,}\s*Forwarded message\s*-+$", re.MULTILINE)

        # First, attempt to split by the forwarding pattern
        forward_parts = forward_pattern.split(text_content, 1)
        if len(forward_parts) > 1:
            # If there is text before the forwarding header, return it
            pre_forward_content = forward_parts[0].strip()
            if pre_forward_content:
                text_content = pre_forward_content

        # If no forwarding header or no content before it, check for reply pattern
        reply_parts = reply_pattern.split(text_content, 1)
        if len(reply_parts) > 1:
            # Return text before the reply pattern
            text_content = reply_parts[0].strip()

        return text_content

    @functools.lru_cache(maxsize=1000)
    def auto_translate(self, text):
        """
        Translate non-english email
        :param text:
        :return:
        """

        # Skip language detection + translation entirely when translation is off.
        # (Previously langdetect ran on every subject/body regardless, and
        # non-English text hit the network even with the flag disabled.)
        if not self.translate_on:
            return text

        is_in_english = True
        max_retries = 3  # Number of times to retry the translation
        retry_delay = 2  # Seconds to wait before retrying
        chunk_size = 4900  # Character limit for each translation request

        try:
            detected_langs = detect_langs(text)
            for lang in detected_langs:
                if lang.lang != "en":
                    is_in_english = False
                    break
        except langdetect.lang_detect_exception.LangDetectException:
            is_in_english = False

        if not is_in_english:
            chunk = text[:chunk_size]

            for attempt in range(max_retries):
                try:
                    translated_chunk = self.translator.translate(chunk, source="auto", target="english")
                    if translated_chunk is not None:
                        return translated_chunk  # Combine all translated chunks
                    return chunk
                except Exception:
                    Logger.spit(
                        "Email translation fails with exception", debug=True, caller_prefix=EmailDataset._CallerPrefix
                    )
                    time.sleep(retry_delay)

        return text

    def extract_subject(self, email_content: Message) -> str:
        """
        Get subject
        :param email_content:
        :return:
        """

        subject = email_content.get("subject", "")
        if len(subject):
            try:
                subject = self.decode_header(subject)
            except UnicodeDecodeError:  # unicode error
                subject = email_content.get("subject", "")

        return subject

    def extract_sender(self, email_content: Message) -> Tuple[Optional[str], Optional[str]]:
        """
        Get sender name and address
        :param email_content:
        :return: sender_name, sender_address
        """

        from_header = email_content.get("From", "")

        # Name in parentheses, Email without angle brackets
        comment_match = re.search(r"(.+?)\s*\(([^)]+)\)$", from_header)
        if comment_match:
            sender_address = comment_match.group(1).strip()
            sender_name = comment_match.group(2).strip()
        else:
            email_pattern = r"<([^<>]+)>$"
            # Extract the last email address
            match = re.search(email_pattern, from_header)
            if match:
                sender_address = match.group(1).strip()  # Extract the email
                sender_name = from_header[: match.start()].strip()  # Everything before the email
            else:
                sender_name = sender_address = from_header.strip()

            if sender_name:
                sender_name = self.decode_header(sender_name)

        return sender_name, sender_address

    @staticmethod
    def extract_recipients(email_content: Message) -> Tuple[List[str], List[str]]:
        """
        Get all recipients names and addresses
        :param email_content:
        :return:
        """

        to_addresses = email_content.get("To", "")
        cc_addresses = email_content.get("Cc", "")  # Handling Cc if needed
        bcc_addresses = email_content.get("Bcc", "")  # Handling Bcc if needed, though Bcc should not be visible

        # Parsing multiple addresses
        all_recipients = getaddresses([to_addresses, cc_addresses, bcc_addresses])

        # Extract just the email addresses, and filter out any empty entries
        to_names = [name for name, addr in all_recipients if addr]
        to_addresses = [addr for name, addr in all_recipients if addr]

        # If 'To' addresses are missing, use 'Delivered-To' as a fallback
        if len(to_addresses) == 0:
            if email_content.get("Delivered-To"):
                to_addresses = [email_content.get("Delivered-To")]

        # Check for forged recipient in 'Received' headers
        received_headers = email_content.get_all("Received", [])
        for header in received_headers:
            if "for" in header.lower():
                parts = header.split("for")
                if len(parts) > 1:
                    forged_address = parts[1].split(";")[0].strip()
                    if "<" in forged_address and ">" in forged_address:
                        forged_address = forged_address.strip("<>")
                    if "@" in forged_address:
                        to_names.append("")
                        to_addresses.append(forged_address)

        return to_names, to_addresses

    @staticmethod
    def extract_reply_to_address(email_content: Message) -> Optional[str]:
        reply_to = email_content.get("Reply-To", "").strip()
        if reply_to:
            reply_to_addresses = getaddresses([reply_to])  # Handles potential list of addresses
            return reply_to_addresses[0][1]  # Return the first parsed email address
        return None

    @staticmethod
    def unfragment_text(text):
        """
        Merge consecutive single characters into a word
        :param text:
        :return:
        """

        pattern = r"\b(?:\w\s+){2,}\w\b"

        def replace_match(match):
            # Extract the matched fragment
            fragment = match.group()
            # Check if the fragment ends before an uppercase letter
            # Find the position after the fragment
            end_pos = match.end()
            # If the next character is uppercase, stop unfragmenting here
            if end_pos < len(text) and text[end_pos].isupper():
                return fragment  # Do not replace; leave as is
            else:
                # Remove spaces between single characters
                return "".join(fragment.split())

        # Use a loop to iteratively replace fragmented parts
        previous_text = None
        while previous_text != text:
            previous_text = text
            text = re.sub(pattern, replace_match, text)

        # Finally, replace multiple spaces with a single space to clean up the text
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        return cleaned_text

    @staticmethod
    def extract_rendered_text_from_html(html_content) -> str:
        """
        Extract the text from HTML
        :param html_content:
        :return:
        """

        unwanted_extensions = {
            ".css",
            ".js",
            ".ico",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
        }
        unwanted_attachment = {"mailto:", "tel:", "#", "javascript", "mso["}

        # Initialize BeautifulSoup with the 'html.parser'
        try:
            soup = BeautifulSoup(html_content, "html.parser")
        except bs4.builder.ParserRejectedMarkup:
            soup = BeautifulSoup(html_content, "lxml")
        except Exception:
            return ""

        # Remove all comments and processing instructions
        for element in soup.find_all(text=lambda text: isinstance(text, (Comment, ProcessingInstruction, Doctype))):
            element.extract()

        # Remove unwanted tags such as style, script, head, meta, and conditional comments
        for tag in soup(["style", "script", "head", "meta"]):
            tag.decompose()

        # Function to determine if a tag's text should be ignored
        def is_ignorable(element):
            return element.name in ["style", "script", "head", "meta"]

        # Initialize a list to hold text parts
        text_parts = []

        # Iterate over all descendants in the soup
        for element in soup.descendants:
            if isinstance(element, NavigableString):
                # Skip if the parent tag is ignorable
                if is_ignorable(element.parent):
                    continue
                # Strip the text to remove leading/trailing whitespace
                stripped_text = element.strip()
                if stripped_text:
                    text_parts.append(stripped_text)
            elif element.name == "a":
                # Process <a> tags with href attributes
                href = element.get("href", "")
                text = element.get_text(strip=True)

                # fixme: how to handle the embedded urls?
                if (
                    href
                    and (not any(href.lower().endswith(ext) for ext in unwanted_extensions))
                    and (not any(noisy in href.lower() for noisy in unwanted_attachment))
                ):
                    href = EmailDataset.shrink_urls(href)
                    text_parts.append(f"{text} ({href})")
                else:
                    text_parts.append(f"{text}")

        # Join all text parts with a space to maintain readability
        combined_text = " ".join(text_parts)

        # Use regex to replace multiple spaces with a single space
        cleaned_text = re.sub(r"\s+", " ", combined_text)

        # Optionally, further clean up specific artifacts from the original HTML
        # For example, remove residual conditional comment indicators if any
        cleaned_text = re.sub(r"\[if .*?\]>", "", cleaned_text)
        cleaned_text = re.sub(r"<!\[endif\]", "", cleaned_text)

        # Remove any remaining square-bracketed text that may not have been removed
        cleaned_text = re.sub(r"\[.*?\]", "", cleaned_text)
        return cleaned_text.strip()

    def extract_text_content(self, part: Message) -> Tuple[str, str]:
        """
        Recursively extracts text content from an email part, including handling nested multiparts.
        :param part:
        :return:
        """

        text_content = ""
        html_content = ""

        # Check if the part is a multipart
        if part.is_multipart():
            for subpart in part.get_payload():
                content = self.extract_text_content(subpart)
                text_content += content[0]
                html_content += content[1]
        else:
            if (
                self.translate_on
            ):  ## fixme: When the email is in Chinese, need to turn this flag on, no decoding, direct translate
                raw_email_content = part.get_payload(decode=False)
                decoded_content = self.auto_translate(raw_email_content)
                content_type = "text"
            else:
                content_type = part.get_content_type()
                charset = part.get_content_charset("utf-8")
                raw_email_content = part.get_payload(decode=True)  # decoding is handled here
                transfer_encoding = part.get("Content-Transfer-Encoding", "").lower()
                # Handle decoded content directly if available
                if raw_email_content:
                    # Decode content based on encoding specified
                    if transfer_encoding == "quoted-printable":
                        decoded_content = self.decode_quoted_printable(raw_email_content, charset)
                    else:
                        try:
                            decoded_content = raw_email_content.decode(charset, "replace")
                        except LookupError:  # in case charset is not recognized
                            decoded_content = raw_email_content.decode("utf-8", "replace")
                else:
                    decoded_content = raw_email_content  # handle cases where payload is not encoded

            ### Extract text from HTML
            if "html" in content_type:
                html_content = decoded_content
                text_content = self.extract_rendered_text_from_html(decoded_content)

            elif "text" in content_type:  ### plain text
                if decoded_content:
                    text_content = decoded_content.strip()

        return str(text_content), str(html_content)

    def __getitem__(self, idx):
        email_file_path = self.file_list[idx]
        email_content = self.load_email_content(email_file_path)

        headers = str(email_content._headers)

        sender_name, sender_address = self.extract_sender(email_content)
        to_names, to_addresses = self.extract_recipients(email_content)
        reply_to_address = self.extract_reply_to_address(email_content)
        if reply_to_address is None:
            reply_to_address = sender_address
        if self.translate_on:
            sender_name = self.auto_translate(sender_name)

        subject = self.extract_subject(email_content)
        subject = self.auto_translate(subject)
        sender_name = normalization(sender_name)
        subject = normalization(subject)

        text_content, html_content = self.extract_text_content(email_content)
        text_content = self.remove_prev_messages(text_content)
        text_content = self.clean_text_content(text_content)
        text_content = self.auto_translate(text_content)
        text_content = normalization(text_content)
        text_content = self.unfragment_text(text_content)

        return (
            email_file_path,
            (sender_name, sender_address),
            (to_names, to_addresses),
            reply_to_address,
            subject,
            text_content,
            headers,
        )
