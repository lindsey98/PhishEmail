"""Characterization tests for the deterministic (OpenAI-free) helpers of the
knowledge-base expansion agent.

These import ``lib.reference_db.agent_helpers`` directly; thanks to the lazy
package ``__init__`` they do not pull in torch/faiss/transformers. The import
chain does need ``tldextract`` (used by ``agent_constants``), so the whole module
skips itself if that is unavailable.
"""

import pytest

pytest.importorskip("tldextract")

from lib.reference_db import agent_helpers as h

# ---- email recognition / cleaning ----


def test_is_email():
    assert h._is_email("a@b.com")
    assert not h._is_email("not-an-email")
    assert not h._is_email("a@b")


def test_clean_emails_dedups_case_insensitively_and_drops_non_emails():
    assert h._clean_emails(["A@B.COM", "a@b.com", "x"]) == ["A@B.COM"]
    assert h._clean_emails([]) == []


def test_extract_emails_from_text():
    found = h._extract_emails_from_text("contact: info@paypal.com or sales@paypal.com.")
    assert found == ["info@paypal.com", "sales@paypal.com"]


def test_parse_mailto_href():
    assert h._parse_mailto_href("mailto:Foo@Bar.com?subject=hi") == "Foo@Bar.com"
    assert h._parse_mailto_href("https://example.com") is None


def test_decode_cloudflare_cfemail_roundtrip():
    plain = "a@b.com"
    key = 0x7F
    enc = format(key, "02x") + "".join(format(ord(c) ^ key, "02x") for c in plain)
    assert h._decode_cloudflare_cfemail_hex(enc) == plain


# ---- host / domain utilities ----


def test_hostname_from_url_lowercases_and_strips_www():
    assert h._hostname_from_url("https://www.PayPal.com/x?y=1") == "paypal.com"


def test_strip_www():
    assert h._strip_www("www.paypal.com") == "paypal.com"


def test_site_root_url():
    assert h._site_root_url("https://a.paypal.com/login/page") == "https://a.paypal.com/"


def test_normalize_official_urls_to_site_roots_dedups_and_drops_junk():
    out = h._normalize_official_urls_to_site_roots(["https://x.paypal.com/a", "https://x.paypal.com/b", "notaurl"])
    assert out == ["https://x.paypal.com/"]


def test_naive_registrable_domain():
    assert h._naive_registrable_domain("a.b.paypal.co.uk") == "paypal.co.uk"
    assert h._naive_registrable_domain("mail.paypal.com") == "paypal.com"


def test_allowed_hosts_matching():
    allowed = h._allowed_hosts_set(["paypal.com"])
    assert h._host_matches_allowed("www.paypal.com", allowed)
    assert h._host_matches_allowed("paypal.com", allowed)
    assert not h._host_matches_allowed("evil.com", allowed)


def test_html_to_visible_text():
    assert h._html_to_visible_text("<p>Hi <b>there</b></p>") == "Hi there"
