"""Email input/output: convert the various container formats users provide
(.mbox, .pst, .msg) into plain ``.eml`` files, and resolve an arbitrary input
path into something the dataset loaders understand.

This module deliberately depends only on the standard library at import time.
Heavier / optional third-party packages (``tqdm``, ``pypff`` for .pst,
``extract_msg`` for .msg, ``bs4`` for .pst HTML bodies) are imported lazily inside
the functions that need them, so importing this module is always cheap and never
fails because an optional backend is missing.
"""

from __future__ import annotations

import os
from email import message_from_string
from email.utils import parseaddr
from typing import List, Union

# Plain RFC-822 email files the dataset loaders can read directly.
PLAIN_EXTENSIONS = (".eml", ".txt")
# Container / proprietary formats we know how to convert into .eml files.
CONVERTIBLE_EXTENSIONS = (".mbox", ".pst", ".msg")
SUPPORTED_EXTENSIONS = PLAIN_EXTENSIONS + CONVERTIBLE_EXTENSIONS


def _tqdm(iterable, **kwargs):
    """Use tqdm if available, otherwise fall back to a no-op wrapper."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def _collect_plain_emails(folder: str) -> List[str]:
    """Recursively collect .eml/.txt files under ``folder``."""
    collected: List[str] = []
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(PLAIN_EXTENSIONS):
                collected.append(os.path.join(root, name))
    return sorted(collected)


def mbox_to_eml(mbox_path: str, output_dir: str) -> List[str]:
    """Export every message in an mbox file to an individual .eml file."""
    import mailbox

    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []
    mbox = mailbox.mbox(mbox_path)

    for i, message in _tqdm(enumerate(mbox), desc="Exporting mbox to eml files"):
        try:
            msg_str = message.as_string()
        except UnicodeEncodeError:
            continue

        eml_filename = os.path.join(output_dir, f"email_{i + 1}.eml")
        try:
            with open(eml_filename, "w", encoding="utf-8") as eml_file:
                eml_file.write(msg_str)
            written.append(eml_filename)
        except UnicodeEncodeError:
            if os.path.exists(eml_filename):
                os.remove(eml_filename)
    return written


def msg_to_eml(msg_path: str, output_dir: str) -> List[str]:
    """Convert a single Outlook ``.msg`` file to an ``.eml`` file.

    Requires the optional ``extract_msg`` package.
    """
    try:
        import extract_msg
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "Reading .msg files requires the 'extract-msg' package. "
            "Install it with: pip install extract-msg"
        ) from exc

    os.makedirs(output_dir, exist_ok=True)
    msg = extract_msg.openMsg(msg_path)
    try:
        # extract_msg can emit a standards-compliant EML representation.
        eml_bytes = msg.asEmlBytes()
    finally:
        msg.close()

    base = os.path.splitext(os.path.basename(msg_path))[0]
    eml_filename = os.path.join(output_dir, f"{base}.eml")
    with open(eml_filename, "wb") as eml_file:
        eml_file.write(eml_bytes)
    return [eml_filename]


def pst_to_eml(pff_file_path: str, output_dir: str) -> List[str]:
    """Export every message in an Outlook ``.pst`` archive to ``.eml`` files.

    Requires the optional ``pypff`` package (``pip install libpff-python``).
    """
    try:
        import pypff
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "Reading .pst files requires the 'pypff' package. "
            "Install it with: pip install libpff-python"
        ) from exc

    written: List[str] = []

    def process_email(email, out_dir):
        eml_filename = os.path.join(out_dir, f"{email.identifier}.eml")

        with open(eml_filename, "w", encoding="utf-8") as eml_file:
            if email.transport_headers:
                eml_file.write(email.transport_headers)
                eml_file.write("\n")
            else:
                headers = ""
                sender_name = email.get_sender_name() if hasattr(email, "get_sender_name") else "Unknown"
                sender_attributes = parseaddr(message_from_string(email.transport_headers or "").get("From", ""))
                sender_email = sender_attributes[1] if len(sender_attributes) else ""
                headers += f"From: {sender_name} <{sender_email}>\n"
                headers += f"Date: {email.delivery_time}\n" if hasattr(email, "delivery_time") else "Date: \n"
                headers += f"Subject: {email.subject}\n" if hasattr(email, "subject") else "Subject: \n"
                headers += "Content-Type: text/plain; charset=utf-8\n"
                headers += "Content-Transfer-Encoding: 8bit\n"
                headers += "\n"
                eml_file.write(headers)

            if email.html_body:
                from bs4 import BeautifulSoup, Comment, Doctype, NavigableString

                soup = BeautifulSoup(email.html_body, "html.parser")
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose()
                for element in soup.contents:
                    if isinstance(element, (Comment, Doctype)):
                        element.extract()
                text_parts = [
                    element.strip()
                    for element in soup.descendants
                    if isinstance(element, NavigableString) and element.strip()
                ]
                email_body = ". ".join(text_parts)
            elif email.plain_text_body:
                try:
                    email_body = email.plain_text_body.decode("utf-8", errors="replace")
                except UnicodeDecodeError:
                    email_body = email.plain_text_body.decode("iso-8859-1", errors="replace")
                except AttributeError:
                    email_body = email.plain_text_body
            else:
                email_body = ""

            eml_file.write(email_body)
        written.append(eml_filename)

    def extract_emails_and_folders(folder, out_dir):
        for item in _tqdm(folder.sub_items, desc="Exporting pst to eml files"):
            if isinstance(item, pypff.folder):
                subfolder_output_dir = os.path.join(out_dir, item.name)
                os.makedirs(subfolder_output_dir, exist_ok=True)
                extract_emails_and_folders(item, subfolder_output_dir)
            elif isinstance(item, pypff.message):
                process_email(item, out_dir)

    pff_file = pypff.file()
    pff_file.open(pff_file_path)
    os.makedirs(output_dir, exist_ok=True)
    extract_emails_and_folders(pff_file.get_root_folder(), output_dir)
    pff_file.close()
    return written


# Map a convertible extension to the function that expands it into .eml files.
_CONVERTERS = {
    ".mbox": mbox_to_eml,
    ".pst": pst_to_eml,
    ".msg": msg_to_eml,
}


def _convert_file(path: str, output_dir: str) -> List[str]:
    ext = os.path.splitext(path)[1].lower()
    converter = _CONVERTERS[ext]
    converter(path, output_dir)
    # Re-scan so we also pick up nested output (e.g. .pst folder hierarchies).
    return _collect_plain_emails(output_dir)


def resolve_email_input(path: str) -> Union[str, List[str]]:
    """Normalize an arbitrary email-input path into something the dataset
    loaders accept (a directory path, a single file path, or a list of .eml
    file paths).

    Supported inputs:
      - A single ``.eml`` / ``.txt`` file  -> returned unchanged.
      - A single ``.mbox`` / ``.pst`` / ``.msg`` file -> converted to .eml files.
      - A directory -> returned as-is, unless it also contains convertible
        archives (.mbox/.pst/.msg), in which case those are expanded and the
        full list of .eml/.txt files is returned.

    Raises a clear error for unsupported files or missing paths.
    """
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in PLAIN_EXTENSIONS:
            return path
        if ext in CONVERTIBLE_EXTENSIONS:
            out_dir = os.path.splitext(path)[0] + "_eml"
            produced = _convert_file(path, out_dir)
            if not produced:
                raise ValueError(f"No emails could be extracted from {path!r}.")
            return produced
        raise ValueError(
            f"Unsupported email file {path!r}. Supported extensions: "
            f"{', '.join(SUPPORTED_EXTENSIONS)}."
        )

    if os.path.isdir(path):
        convertibles = [
            os.path.join(root, name)
            for root, _dirs, files in os.walk(path)
            for name in files
            if name.lower().endswith(CONVERTIBLE_EXTENSIONS)
        ]
        if not convertibles:
            # Plain directory of .eml/.txt — preserve the original behaviour.
            return path

        out_dir = path.rstrip("/").rstrip("\\") + "_converted"
        os.makedirs(out_dir, exist_ok=True)
        emails = _collect_plain_emails(path)
        for archive in convertibles:
            sub = os.path.join(out_dir, os.path.splitext(os.path.basename(archive))[0])
            emails.extend(_convert_file(archive, sub))
        if not emails:
            raise ValueError(f"No emails could be found or extracted under {path!r}.")
        return sorted(set(emails))

    raise FileNotFoundError(f"Email input path does not exist: {path!r}")
