"""Smoke tests for the HTML labelling helpers.

Only depends on BeautifulSoup (bs4); skipped automatically if it is not
installed, so the suite still runs in a minimal environment.
"""
import pytest

pytest.importorskip("bs4")

from lib.labeling import label_eml


def test_append_css_styling_inserts_style_block():
    html = "<html><head></head><body>hi</body></html>"
    out = label_eml.append_css_styling(html, "body{color:red}")
    assert "<style>" in out
    assert "body{color:red}" in out


def test_append_css_styling_without_head_prepends_style():
    out = label_eml.append_css_styling("<body>hi</body>", "x{}")
    assert out.strip().startswith("<style>")


def test_remove_images_strips_img_tags():
    out = label_eml.remove_images('<p><img src="a.png"/>hello</p>')
    assert "<img" not in out
    assert "hello" in out


def test_label_html_wraps_identity():
    out = label_eml.label_html("<p>PayPal account</p>", identities=["PayPal"], actions=[])
    assert "entity identity" in out
