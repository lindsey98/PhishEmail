"""Deterministic, network-free helpers for the KB expansion agent: email/HTML/domain
parsing and identity-core matching. No OpenAI dependency."""

from __future__ import annotations

import html
import json
import random
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import unquote, urljoin, urlparse

try:
    import difflib
except ImportError:
    difflib = None  # type: ignore

from .agent_constants import (
    _ATTACHMENT_EXT_RE,
    _CFEMAIL_DATA_RE,
    _DOMAIN_ONLY_FALSE_TLDS,
    _EMAIL_FIND_RE,
    _EMAIL_RE,
    _MAILTO_GLOBAL_RE,
    _NAIVE_MULTI_LABEL_SUFFIXES,
    _STANDALONE_AT_DOMAIN_RE,
    _TOKEN_SPLIT,
    _URL_EMAIL_HINTS,
    BASE_BACKOFF,
    USE_HEURISTIC_DOMAIN_FALLBACK,
    _ext,
)


def _jitter_delay(attempt: int) -> float:
    base = BASE_BACKOFF * (2**attempt)
    return random.uniform(base * 0.5, base * 1.5)


def _parse_json_strict(text: str) -> Any:
    if not text or not isinstance(text, str):
        raise ValueError("empty")
    s = text.strip()
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.I).strip()
    s = re.sub(r"\s*```\s*$", "", s).strip()
    s = re.sub(r"^```[a-zA-Z]*\n?", "", s, flags=re.MULTILINE).strip()
    s = re.sub(r"\n?```\s*$", "", s, flags=re.MULTILINE).strip()
    return json.loads(s)


def _get_attr(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _is_email(s: str) -> bool:
    return isinstance(s, str) and bool(_EMAIL_RE.match(s.strip()))


def _clean_emails(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        if not isinstance(x, str):
            continue
        y = x.strip()
        if _is_email(y) and y.lower() not in seen:
            seen.add(y.lower())
            out.append(y)
    return out


def _allowed_hosts_set(allowed_domains: List[str]) -> Set[str]:
    s: Set[str] = set()
    for x in allowed_domains or []:
        x = (x or "").strip().lower().rstrip(".")
        if x:
            s.add(x)
    return s


def _host_matches_allowed(host: Optional[str], allowed_set: Set[str]) -> bool:
    if not host:
        return False
    h = host.lower().strip().rstrip(".")
    if not h:
        return False
    for a in allowed_set:
        if h == a or h.endswith("." + a):
            return True
    return False


def _strip_trailing_junk(addr: str) -> str:
    s = addr.strip()
    while s and s[-1] in ".,;:)]}>\"'":
        s = s[:-1]
    return s


def _standalone_at_domain_to_email(domain: str) -> Optional[str]:
    d = (domain or "").strip().lower().rstrip(".")
    if not d or "." not in d:
        return None
    parts = d.split(".")
    tld = parts[-1].lower()
    if tld in _DOMAIN_ONLY_FALSE_TLDS or not tld.isascii():
        return None
    if not all(parts):
        return None
    cand = f"postmaster@{d}"
    return cand if _is_email(cand) else None


def _extract_emails_from_text(text: str) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for m in _EMAIL_FIND_RE.finditer(text):
        s = _strip_trailing_junk(m.group(0))
        if not _is_email(s):
            continue
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    for m in _STANDALONE_AT_DOMAIN_RE.finditer(text):
        cand = _standalone_at_domain_to_email(m.group(1))
        if not cand:
            continue
        k = cand.lower()
        if k not in seen:
            seen.add(k)
            out.append(cand)
    return out


def _decode_cloudflare_cfemail_hex(hexs: str) -> Optional[str]:
    h = (hexs or "").strip().lower()
    if len(h) < 4 or (len(h) & 1) != 0:
        return None
    try:
        raw = bytes.fromhex(h)
    except ValueError:
        return None
    if len(raw) < 2:
        return None
    key, rest = raw[0], raw[1:]
    out = bytes(b ^ key for b in rest)
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return out.decode("latin-1", errors="replace")


def _parse_mailto_href(href: str) -> Optional[str]:
    h = (href or "").strip()
    if not h.lower().startswith("mailto:"):
        return None
    rest = h[7:].split("?", 1)[0].split("#", 1)[0].strip()
    if not rest:
        return None
    try:
        rest = unquote(rest)
    except Exception:
        pass
    return rest or None


def _html_to_visible_text(page_html: str) -> str:
    if not page_html:
        return ""
    s = page_html
    s = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", s)
    s = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", s)
    s = re.sub(r"(?is)<noscript[^>]*>.*?</noscript>", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[\s\xa0\u200b\u00ad]+", " ", s).strip()
    return s


def _extract_emails_from_html_page(page_html: str) -> List[str]:
    if not page_html:
        return []
    seen: Set[str] = set()
    ordered: List[str] = []

    def add_one(addr: str) -> None:
        s = _strip_trailing_junk(addr)
        if not _is_email(s):
            return
        k = s.lower()
        if k in seen:
            return
        seen.add(k)
        ordered.append(s)

    for m in _CFEMAIL_DATA_RE.finditer(page_html):
        dec = _decode_cloudflare_cfemail_hex(m.group(1))
        if dec:
            add_one(dec)

    for m in _MAILTO_GLOBAL_RE.finditer(page_html):
        u = m.group(1)
        u = (u or "").strip()
        if not u:
            continue
        try:
            u = unquote(u)
        except Exception:
            pass
        add_one(u)

    unesc = html.unescape(page_html)
    for e in _extract_emails_from_text(unesc):
        add_one(e)
    for e in _extract_emails_from_text(_html_to_visible_text(unesc)):
        add_one(e)

    return ordered


def _url_visit_key(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    u = u.split("#", 1)[0]
    return u


class _AHrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "a":
            return
        for k, v in attrs:
            if k and k.lower() == "href" and v:
                self.hrefs.append(v)
                return


def _link_priority_for_email_hunt(url: str) -> float:
    s = 0.0
    try:
        p = urlparse(url)
        path_q = f"{(p.path or '/').lower()}?{(p.query or '').lower()}"
    except Exception:
        path_q = (url or "").lower()
    for hint in _URL_EMAIL_HINTS:
        if hint in path_q:
            s += 2.0
    if re.search(r"/(contact|support|help|legal|impress|privacy|kontakt|mail)(/|$|\.html?)", path_q, re.I):
        s += 1.0
    return s


def _parse_page_links(html: str, base_url: str) -> Tuple[List[str], List[str]]:
    if not html:
        return [], []
    par = _AHrefParser()
    try:
        par.feed(html)
    except Exception:
        pass
    http_links: List[str] = []
    mailtos: List[str] = []
    for h in par.hrefs:
        h0 = h.strip()
        if not h0 or h0.lower().startswith(("javascript:", "#")):
            continue
        if h0.lower().startswith("mailto:"):
            p = _parse_mailto_href(h0)
            if p:
                mailtos.append(p)
            continue
        absu = urljoin(base_url, h0)
        if absu.startswith("http://") or absu.startswith("https://"):
            http_links.append(absu)
    return http_links, mailtos


def _hostname_from_url(url: str) -> Optional[str]:
    if not isinstance(url, str):
        return None
    u = url.strip()
    if not u:
        return None
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    try:
        p = urlparse(u)
        h = p.netloc.split("@")[-1].split(":")[0].lower()
        if h.startswith("www."):
            h = h[4:]
        return h or None
    except Exception:
        return None


def _site_root_url(url: str) -> str:
    u = (url or "").strip()
    if not u.startswith(("http://", "https://")):
        return u
    try:
        p = urlparse(u)
        if not p.netloc:
            return u
        h = p.netloc.split("@")[-1].split(":")[0].lower()
        if h.startswith("www."):
            h = h[4:]
        if not h:
            return u
        return f"{p.scheme}://{h}/"
    except Exception:
        return u


def _normalize_official_urls_to_site_roots(urls: Any) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for u in urls or []:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u.startswith(("http://", "https://")):
            continue
        r = _site_root_url(u)
        k = _url_visit_key(r)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _filter_candidate_urls(urls: List[str], allowed_set: Set[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u.startswith(("http://", "https://")):
            continue
        if not _host_matches_allowed(_hostname_from_url(u), allowed_set):
            continue
        try:
            path = urlparse(u).path.lower()
        except Exception:
            path = ""
        if _ATTACHMENT_EXT_RE.search(path.rstrip("/")):
            continue
        k = _url_visit_key(u)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(u)
    return out


def _naive_registrable_domain_for_plan(hostname: str) -> str:
    parts = hostname.lower().strip().split(".")
    if len(parts) < 2:
        return hostname.lower()
    if (
        len(parts) >= 3
        and len(parts[-1]) == 2
        and parts[-2]
        in (
            "com",
            "co",
            "gov",
            "ac",
            "org",
            "net",
            "edu",
        )
    ):
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _allowed_domains_from_plan(plan: Dict[str, Any]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []

    def add(s: str) -> None:
        s = s.strip().lower().rstrip(".")
        if not s or s in seen:
            return
        seen.add(s)
        out.append(s)

    for u in plan.get("official_urls") or []:
        if not isinstance(u, str):
            continue
        h = _hostname_from_url(u)
        if h:
            add(h)
            add(_naive_registrable_domain_for_plan(h))

    for hint in plan.get("target_site_hints") or []:
        if not isinstance(hint, str):
            continue
        h = hint.strip().lower().rstrip(".")
        if not h or " " in h:
            continue
        if "/" in h:
            hh = _hostname_from_url(h if h.startswith("http") else "https://" + h)
            if hh:
                add(hh)
                add(_naive_registrable_domain_for_plan(hh))
        else:
            add(h)
            if "." in h:
                add(_naive_registrable_domain_for_plan(h))

    return out


# ---------- filter.py domain helpers ----------
def _normalize_domain(d: str) -> str:
    return d.strip().lower().rstrip(".")


def _email_registrable_host(addr: str) -> Optional[str]:
    if not isinstance(addr, str) or "@" not in addr:
        return None
    host = addr.rsplit("@", 1)[-1].strip().lower().strip(">")
    host = host.strip()
    return host if host else None


def _strip_www(host: str) -> str:
    h = _normalize_domain(host)
    if h.startswith("www."):
        return h[4:]
    return h


def _domain_labels(hostname: str) -> List[str]:
    return [x for x in _normalize_domain(hostname).split(".") if x]


def _naive_registrable_domain(hostname: str) -> str:
    hostname = _strip_www(hostname)
    if not hostname:
        return hostname
    parts = hostname.split(".")
    if len(parts) < 2:
        return hostname
    if len(parts) >= 3 and parts[-2] == "co" and parts[-1] == "jp":
        return ".".join(parts[-3:])
    suf2 = ".".join(parts[-2:])
    if suf2 in _NAIVE_MULTI_LABEL_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[-3:])
    if len(parts) >= 3 and parts[-2] == "co" and len(parts[-1]) == 2 and parts[-1].isalpha():
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _suffix_labels_match(host: str, core: str) -> bool:
    hl = _domain_labels(host)
    cl = _domain_labels(core)
    if not hl or not cl or len(hl) < len(cl):
        return False
    return hl[-len(cl) :] == cl


def _apex_weak_match(host: str, core: str) -> bool:
    return _naive_registrable_domain(host) == _naive_registrable_domain(core)


def _host_weak_matches_core(host: str, core: str) -> bool:
    host = _strip_www(host)
    core = _strip_www(core)
    if not host or not core:
        return False
    if host == core:
        return True
    if _suffix_labels_match(host, core):
        return True
    if _apex_weak_match(host, core):
        return True
    return False


def _noise_augment_blocklisted(domain: str) -> bool:
    d = domain.lower()
    if d.endswith(".gov") or d.endswith(".mil"):
        return True
    if d.endswith(".gov.uk") or d.endswith(".gov.au"):
        return True
    if ".gov." in d:
        return True
    if d.endswith(".go.jp") or d.endswith(".gov.cn"):
        return True
    return False


def _validate_llm_keep_domains(keeps: List[str], emails: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for raw in keeps:
        if not isinstance(raw, str):
            continue
        d = _normalize_domain(raw.strip())
        if not d or _noise_augment_blocklisted(d):
            continue
        matched = False
        for e in emails:
            h = _email_registrable_host(e)
            if not h:
                continue
            hn = _normalize_domain(h)
            ap = _naive_registrable_domain(hn)
            if d == ap or d == hn or _suffix_labels_match(hn, d):
                matched = True
                break
        if matched and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _brand_prefixes_from_llm_domains(llm_domains: Set[str]) -> Set[str]:
    prefs: Set[str] = set()
    for dom in llm_domains:
        lab = dom.split(".")[0].lower()
        if len(lab) < 4:
            continue
        prefs.add(lab[:4])
        if len(lab) >= 6:
            prefs.add(lab[:6])
    return prefs


def _hyphen_token_overlap_with_llm(left_label: str, llm_domains: Set[str]) -> bool:
    left_chunks = {t.lower() for t in _TOKEN_SPLIT.split(left_label) if len(t) >= 4}
    if not left_chunks:
        left_chunks = {left_label.lower()} if len(left_label) >= 4 else set()
    if not left_chunks:
        return False
    for dom in llm_domains:
        stem = dom.split(".")[0].lower()
        stem_chunks = {t for t in _TOKEN_SPLIT.split(stem) if len(t) >= 4}
        if len(stem) >= 4:
            stem_chunks.add(stem)
        if left_chunks & stem_chunks:
            return True
    return False


def _fuzzy_brand_sibling(left_label: str, llm_domains: Set[str]) -> bool:
    if difflib is None or len(left_label) < 4 or len(left_label) > 14:
        return False
    for dom in llm_domains:
        stem = dom.split(".")[0].lower()
        if len(stem) < 4 or len(stem) > 18:
            continue
        if abs(len(left_label) - len(stem)) > 6:
            continue
        if difflib.SequenceMatcher(None, left_label, stem).ratio() >= 0.65:
            return True
    return False


def _apex_left_label_aligned_with_brand(apex: str, llm_domains: Set[str]) -> bool:
    if not apex:
        return False
    left = apex.split(".")[0].lower()
    if len(left) < 3:
        return False
    prefs = _brand_prefixes_from_llm_domains(llm_domains)
    for p in prefs:
        if len(p) >= 4 and left.startswith(p):
            return True
    if _hyphen_token_overlap_with_llm(left, llm_domains):
        return True
    if _fuzzy_brand_sibling(left, llm_domains):
        return True
    return False


def _build_effective_identity_cores(
    llm_cores: List[str],
    llm_keep_domains_validated: Optional[List[str]],
    allowed_domains: Optional[List[str]],
    emails: List[str],
) -> List[str]:
    eff: Set[str] = set()
    for c in llm_cores:
        if isinstance(c, str) and c.strip():
            eff.add(_normalize_domain(c))

    for d in llm_keep_domains_validated or []:
        if isinstance(d, str) and d.strip():
            dn = _normalize_domain(d)
            if dn and not _noise_augment_blocklisted(dn):
                eff.add(dn)

    llm_frozen = set(eff)
    allowed_norm = [_normalize_domain(a) for a in (allowed_domains or []) if isinstance(a, str) and a.strip()]
    allowed_set = set(allowed_norm)

    hosts_seen: Set[str] = set()
    apex_seen: Set[str] = set()
    for e in emails:
        h = _email_registrable_host(e)
        if not h:
            continue
        hn = _normalize_domain(h)
        hosts_seen.add(hn)
        apex_seen.add(_naive_registrable_domain(hn))

    for cand in allowed_set:
        if cand in llm_frozen:
            continue
        if cand not in hosts_seen and cand not in apex_seen:
            continue
        if _noise_augment_blocklisted(cand):
            continue
        eff.add(cand)

    if USE_HEURISTIC_DOMAIN_FALLBACK:
        for e in emails:
            h = _email_registrable_host(e)
            if not h:
                continue
            hn = _normalize_domain(h)
            apex = _naive_registrable_domain(hn)
            if _noise_augment_blocklisted(apex):
                continue
            if apex in eff or hn in eff:
                continue
            if _apex_left_label_aligned_with_brand(apex, llm_frozen):
                eff.add(apex)

    return sorted(eff)


def _host_under_identity_core(host: str, core_domains: List[str]) -> bool:
    host = _normalize_domain(host)
    if not host:
        return False
    for raw in core_domains:
        if not isinstance(raw, str):
            continue
        c = raw.strip()
        if not c:
            continue
        if _host_weak_matches_core(host, c):
            return True
    return False


def _reject_standalone_grants_program_apex(apex: Optional[str], org: str) -> bool:
    if not apex:
        return False
    if not apex.lower().endswith("grants.org"):
        return False
    ol = (org or "").lower()
    if "grant" in ol:
        return False
    return True


def _apply_identity_core_filter(
    emails: List[str],
    personal_clean: List[str],
    identity_core_domains: List[str],
    allowed_domains: Optional[List[str]],
    llm_keep_domains_validated: Optional[List[str]],
    org: str = "",
) -> Tuple[List[str], List[str], List[str]]:
    personal_lower = {e.lower() for e in personal_clean}
    cores_eff = _build_effective_identity_cores(
        identity_core_domains,
        llm_keep_domains_validated,
        allowed_domains,
        emails,
    )
    official: List[str] = []
    third_party: List[str] = []
    if not cores_eff:
        raise ValueError("identity_core_domains empty")
    for e in emails:
        el = e.lower()
        if el in personal_lower:
            continue
        host = _email_registrable_host(e)
        apex = _naive_registrable_domain(host) if host else None
        if host and _host_under_identity_core(host, cores_eff):
            if _reject_standalone_grants_program_apex(apex, org):
                third_party.append(e)
            else:
                official.append(e)
        else:
            third_party.append(e)
    return official, third_party, cores_eff


# ---------- tldextract (extract_official_domains) ----------
def _registered_domain_etld1(hostname: str) -> str:
    hn = _strip_www(hostname)
    if not hn:
        return ""
    ext = _ext(hn)
    rd = getattr(ext, "top_domain_under_public_suffix", None)
    if rd is None or rd == "":
        rd = getattr(ext, "registered_domain", "") or ""
    return str(rd).lower()


def _extract_etld1_from_url(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url.strip())
        host = parsed.hostname or ""
        host = _normalize_domain(host)
        if not host:
            return None
        rd = _registered_domain_etld1(host) or None
        return rd
    except Exception:
        return None


def _extract_etld1_from_email(email: str) -> Optional[str]:
    if not isinstance(email, str) or "@" not in email:
        return None
    host = email.rsplit("@", 1)[-1].strip().lower().strip(">")
    if not host:
        return None
    rd = _registered_domain_etld1(host) or None
    return rd
