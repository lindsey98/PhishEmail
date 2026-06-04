import email
import email.header
import html
import os
import re
from typing import List, Tuple, Union

from ..data import OCR, EmailDataset
from ..utilities import Logger
from ..utilities.data_utils import normalization

# Playwright is imported lazily inside html_to_png_playwright() so that this
# module can be imported even if the playwright browsers are not yet installed;
# rendering then degrades gracefully to the visible-text fallback.
# Setup: `pip install playwright` then `playwright install chromium`


class RenderDataset(EmailDataset):
    _CallerPrefix = "Dataset Loader (render the eml)"

    def __init__(self, root_path, ocr_model: OCR, dumpDir: str = "temp", translate_on=False, save_imgs=False):
        super().__init__(root_path, translate_on)
        self.ocr_model = ocr_model
        self.dumpDir = dumpDir
        os.makedirs(dumpDir, exist_ok=True)
        self.save_imgs = save_imgs

    # ---------------------- Playwright HTML → PNG ---------------------- #
    @staticmethod
    def html_to_png_playwright(full_html: str, output_path: str) -> bool:
        MAX_HEIGHT = 20000  # px; safely under OpenCV's 32767 limit
        VIEWPORT_W = 1024

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch()
                context = browser.new_context(
                    viewport={"width": VIEWPORT_W, "height": 800},
                    device_scale_factor=1,  # don't 2x the pixels
                )
                page = context.new_page()
                page.set_content(full_html, wait_until="domcontentloaded")

                # Measure the actual content height
                content_h = page.evaluate(
                    "() => Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)"
                )
                clip_h = min(content_h, MAX_HEIGHT)

                page.screenshot(
                    path=output_path,
                    clip={"x": 0, "y": 0, "width": VIEWPORT_W, "height": clip_h},
                )
                browser.close()
            return True
        except Exception as e:
            Logger.spit(f"[ERROR] Playwright render failed: {e}", debug=True)
            return False

    # ---------------------- Sanitization for text-salting ---------------------- #
    @staticmethod
    def sanitize_html(html_body: str) -> str:
        """
        Sanitize HTML to mitigate text-salting / invisible text:
        - Remove hidden spans (width:0, max-width:0, overflow:hidden, opacity:0, etc.)
        - Remove script/style blocks
        - Strip zero-width / bidi chars
        - Strip obviously bad style attributes from other tags
        """
        if not html_body:
            return ""

        # 1) Remove zero-width and bidi control characters
        zero_width_chars = [
            "\u200b",
            "\u200c",
            "\u200d",
            "\u200e",
            "\u200f",
            "\u202a",
            "\u202b",
            "\u202c",
            "\u202d",
            "\u202e",
            "\u2060",
            "\u2066",
            "\u2067",
            "\u2068",
            "\u2069",
            "\ufeff",
        ]
        for ch in zero_width_chars:
            html_body = html_body.replace(ch, "")

        # 2) Remove script and style blocks completely
        html_body = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", html_body)

        # 3) Remove hidden spans entirely (tag + contents)
        suspicious_keys = [
            "display:none",
            "visibility:hidden",
            "opacity:0",
            "color:transparent",
            "font-size:0",
            "font-size:1px",
            "width:0",
            "width:0px",
            "max-width:0",
            "max-width:0px",
            "overflow:hidden",
            "text-indent:-9999px",
            "text-indent: -9999px",
        ]

        span_re = re.compile(r'<span\b[^>]*style=(["\'])(?P<style>.*?)\1[^>]*>.*?</span>', re.IGNORECASE | re.DOTALL)

        def remove_hidden_span(m: re.Match) -> str:
            style = m.group("style").lower()
            if any(k in style for k in suspicious_keys):
                # Drop the span + its content
                return ""
            return m.group(0)

        html_body = span_re.sub(remove_hidden_span, html_body)

        # 4) For *other* tags, strip only the bad style attributes
        def strip_bad_style_attr(m: re.Match) -> str:
            quote = m.group(1)
            style = m.group(2)
            lower_style = style.lower()
            if any(k in lower_style for k in suspicious_keys):
                # Remove style attribute entirely
                return ""
            return f" style={quote}{style}{quote}"

        html_body = re.sub(r'style=(["\'])(.*?)\1', strip_bad_style_attr, html_body, flags=re.IGNORECASE | re.DOTALL)

        return html_body

    # ---------------------- EML → single PNG via Playwright ---------------------- #
    def render_eml(self, data: bytes, dumpName="new") -> Tuple[Union[bool, str], str]:
        """
        Render the eml as a single PNG image via Playwright.

        Steps:
          - Build a single sanitized HTML body from all text/* parts.
          - Wrap with a “safe” stylesheet that normalizes fonts/colors.
          - Use Playwright (Chromium) to render full-page screenshot.
          - Return (image_path or False, visible_text_fallback).
        """

        textTypes = ["text/plain", "text/html"]
        imageTypes = ["image/gif", "image/jpeg", "image/png"]
        # You *can* use PDFs/attachments separately if you want, but for pure
        # anti-salting you usually only care about main body rendering.
        # pdfTypes = ['application/pdf']

        msg = email.message_from_bytes(data)

        # Collect text body as HTML fragments
        body_html_fragments: List[str] = []

        for part in msg.walk():
            mimeType = part.get_content_type()
            if part.is_multipart():
                Logger.spit("[INFO] Multipart found, continue", debug=True)
                continue

            Logger.spit("[INFO] Found MIME part: %s" % mimeType, debug=True)

            # -------- text --------
            if mimeType in textTypes:
                charset = part.get_content_charset("utf-8")
                raw_email_content = part.get_payload(decode=True)
                transfer_encoding = part.get("Content-Transfer-Encoding", "").lower()

                if raw_email_content:
                    if transfer_encoding == "quoted-printable":
                        payload = self.decode_quoted_printable(raw_email_content, charset)
                    else:
                        try:
                            payload = raw_email_content.decode(charset, "replace")
                        except LookupError:
                            payload = raw_email_content.decode("utf-8", "replace")
                else:
                    payload = raw_email_content

                # Basic cleanup of control chars that break layout
                dirtyChars = ["\r"]
                if isinstance(payload, bytes):
                    payload = str(payload)
                for char in dirtyChars:
                    payload = payload.replace(char, "")

                # Convert text/plain to simple HTML, keep text/html as-is
                if mimeType == "text/plain":
                    safe_text = html.escape(payload or "")
                    html_fragment = f"<pre>{safe_text}</pre>"
                else:  # text/html
                    html_fragment = payload or ""

                body_html_fragments.append(html_fragment)

            elif mimeType in imageTypes:
                Logger.spit("[INFO] image/* part found; currently ignored for main body rendering", debug=True)
                continue

        # Build final HTML for the body
        html_visible_text = ""
        resultImagePath = os.path.join(self.dumpDir, f"{dumpName}.png")

        if body_html_fragments:
            # Join fragments with a simple separator
            combined_body_html = "\n<hr />\n".join(body_html_fragments)
            sanitized_body_html = self.sanitize_html(combined_body_html)

            # Wrap with a controlled stylesheet to neutralize obfuscating formatting
            style_block = """
                <style>
                  * {
                    font-family: sans-serif !important;
                    font-size: 12pt !important;
                  }
                  body {
                    margin: 1cm;
                    background-color: #ffffff;
                  }
                  a {
                    text-decoration: underline !important;
                  }
                </style>
            """
            full_html = (
                "<!DOCTYPE html>"
                "<html><head><meta charset='utf-8'/>"
                f"{style_block}"
                "</head><body>"
                f"{sanitized_body_html}"
                "</body></html>"
            )

            # Visible text fallback (for when OCR fails completely)
            html_visible_text = re.sub(r"<[^>]+>", " ", full_html)
            html_visible_text = " ".join(html_visible_text.split())

            # Render HTML → PNG via Playwright
            ok = self.html_to_png_playwright(full_html, resultImagePath)
            if ok:
                return resultImagePath, html_visible_text
            else:
                # Rendering failed; no image, but still return visible text
                return False, html_visible_text

        # No renderable content -> no image, but still return (possibly empty) visible text
        return False, html_visible_text

    # ---------------------- Dataset logic ---------------------- #
    def __getitem__(self, idx):
        email_file_path = self.file_list[idx]
        email_content = self.load_email_content(email_file_path)

        headers = str(email_content._headers)

        sender_name, sender_address = self.extract_sender(email_content)
        to_names, to_addresses = self.extract_recipients(email_content)
        # Drop the sender's own address from the recipient list. sender_address is a
        # single string, so wrap it in a set literal — set(sender_address) would
        # iterate the string into individual characters.
        to_addresses = list(set(to_addresses) - {sender_address})
        reply_to_address = self.extract_reply_to_address(email_content)
        if reply_to_address is None:
            reply_to_address = sender_address

        subject = self.extract_subject(email_content)
        subject = self.auto_translate(subject)
        sender_name = normalization(sender_name)
        subject = normalization(subject)

        text_content, html_content_orig = self.extract_text_content(email_content)

        with open(email_file_path, "rb") as f:
            msg_bytes = f.read()
        resultImage, html_content = self.render_eml(msg_bytes, dumpName=os.path.basename(email_file_path))

        if resultImage is not False:
            text_content_from_ocr = self.ocr_model.ocr(resultImage)
            if not self.save_imgs and os.path.exists(resultImage):
                os.remove(resultImage)
            text_content = text_content_from_ocr
            # Fallback only if OCR really failed (very short), using sanitized visible text
            if len((text_content or "").strip()) < 10:
                text_content = html_content or text_content_from_ocr

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


if __name__ == "__main__":
    rootDir = "./temp.eml"
    dumpDir = "./datasets/phishpot_imgs/"
    Logger.set_debug_on()
    ocr_model = OCR()
    dataset = RenderDataset(rootDir, ocr_model=ocr_model, dumpDir=dumpDir)

    for i in range(len(dataset)):
        (
            email_file_path,
            (sender_name, sender_address),
            (to_names, to_addresses),
            reply_to_address,
            subject,
            email_body_text,
            header,
        ) = dataset[i]

        print()
