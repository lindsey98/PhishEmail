"""Tests for email-format resolution and conversion.

``lib.utilities.email_io`` is intentionally stdlib-only, so these run without the
heavy ML stack. The .msg test skips itself if ``extract_msg`` is not installed.
"""
import mailbox
import os

import pytest

from lib.utilities import email_io


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def test_single_eml_passthrough(tmp_path):
    eml = tmp_path / "msg.eml"
    _write(str(eml), "From: a@example.com\nSubject: hi\n\nbody")
    assert email_io.resolve_email_input(str(eml)) == str(eml)


def test_single_txt_passthrough(tmp_path):
    txt = tmp_path / "msg.txt"
    _write(str(txt), "From: a@example.com\nSubject: hi\n\nbody")
    assert email_io.resolve_email_input(str(txt)) == str(txt)


def test_plain_directory_returned_as_is(tmp_path):
    _write(str(tmp_path / "1.eml"), "Subject: a\n\nx")
    _write(str(tmp_path / "2.txt"), "Subject: b\n\ny")
    assert email_io.resolve_email_input(str(tmp_path)) == str(tmp_path)


def test_mbox_is_expanded(tmp_path):
    mbox_path = str(tmp_path / "inbox.mbox")
    box = mailbox.mbox(mbox_path)
    for subj in ("first", "second"):
        msg = mailbox.mboxMessage(f"From: s@example.com\nSubject: {subj}\n\nhello {subj}")
        box.add(msg)
    box.flush()

    result = email_io.resolve_email_input(mbox_path)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(p.lower().endswith(".eml") for p in result)
    assert all(os.path.isfile(p) for p in result)


def test_directory_with_mbox_is_merged(tmp_path):
    _write(str(tmp_path / "loose.eml"), "Subject: loose\n\nx")
    mbox_path = str(tmp_path / "archive.mbox")
    box = mailbox.mbox(mbox_path)
    box.add(mailbox.mboxMessage("From: s@example.com\nSubject: inside\n\nhi"))
    box.flush()

    result = email_io.resolve_email_input(str(tmp_path))
    assert isinstance(result, list)
    # the loose .eml plus the one extracted from the mbox
    assert len(result) == 2


def test_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "data.csv"
    _write(str(bad), "not,an,email")
    with pytest.raises(ValueError):
        email_io.resolve_email_input(str(bad))


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        email_io.resolve_email_input("/no/such/path/here.eml")


def test_msg_conversion(tmp_path):
    pytest.importorskip("extract_msg")
    # We only assert the converter path is wired; building a real .msg here is
    # out of scope. The import check above keeps this meaningful where the
    # optional backend exists.
    assert ".msg" in email_io.CONVERTIBLE_EXTENSIONS
    assert email_io._CONVERTERS[".msg"] is email_io.msg_to_eml
