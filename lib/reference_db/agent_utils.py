"""
Supporting utilities for the knowledge-base agent (plan/search, fetch BFS,
DNS, LLM prompts, filtering). Imported by knowledge_base_agent.

Integrates concepts from agent_improve_recall / filter /
extract_official_domains_from_urls (no sibling imports).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import html
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import deque
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import unquote, urljoin, urlparse

try:
    import difflib
except ImportError:
    difflib = None  # type: ignore

try:
    import dns.exception
    import dns.resolver

    _DNSPYTHON_AVAILABLE = True
except ImportError:
    _DNSPYTHON_AVAILABLE = False

import tldextract
from openai import APIStatusError, APITimeoutError, OpenAI, OpenAIError

# ---------- Configuration ----------
API_MODEL = "o4-mini"
PER_PHASE_TIMEOUT = 90.0
MAX_RETRIES = 3
BASE_BACKOFF = 0.6
REASONING_EFFORT_PLAN_SEARCH = "low"
REASONING_EFFORT_SEARCH = "low"
REASONING_EFFORT_FILTER = "low"
REASONING_EFFORT_CLASSIFY = "low"

USE_PLAYWRIGHT = True
PLAYWRIGHT_CONCURRENCY = 3
PLAYWRIGHT_PAGE_TIMEOUT_MS = 20_000

FETCH_CONCURRENCY = 4
FETCH_TIMEOUT_SEC = 20
FETCH_MAX_BODY_BYTES = 2_000_000
_FETCH_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

USE_RECALL_LINK_TREE = True
RECALL_LINK_TREE_MAX_DEPTH = 3
RECALL_LINK_TOPK = 6
RECALL_LINK_TOPK_L0 = 15
RECALL_LINK_MAX_FETCH = 80

SITEMAP_TOPK = 30
SITEMAP_MAX_CHILD_SITEMAPS = 5

_DEFAULT_PUBLIC_DNS_SERVERS = [
    "1.1.1.1",
    "1.0.0.1",
    "8.8.8.8",
    "8.8.4.4",
    "9.9.9.9",
    "149.112.112.112",
    "208.67.222.222",
    "208.67.220.220",
    "223.5.5.5",
    "223.6.6.6",
    "119.29.29.29",
]
DNS_VERIFY_TIMEOUT = 3.0
DNS_VERIFY_LIFETIME = 8.0
DNS_VERIFY_CONCURRENCY = 6

USE_HEURISTIC_DOMAIN_FALLBACK = True

_CFEMAIL_DATA_RE = re.compile(r"data-cfemail\s*=\s*['\"]?([0-9a-fA-F]{4,})['\"]?", re.I)
_MAILTO_GLOBAL_RE = re.compile(r"mailto:([^\"'\s<>#?]+)(?:[?#\"\s'<>]|$)", re.I)
_ATTACHMENT_EXT_RE = re.compile(
    r"\.(pdf|docx?|xlsx?|pptx?|zip|csv|ods|odt|7z|tar\.gz|tar|gz|rar"
    r"|png|jpe?g|gif|svg|webp|mp4|mp3|wav|exe|dmg|apk|iso)$",
    re.I,
)
_URL_EMAIL_HINTS = (
    "contact",
    "support",
    "help",
    "about",
    "legal",
    "privacy",
    "impressum",
    "kontakt",
    "customer",
    "service",
    "connect",
    "reach",
    "team",
    "career",
    "press",
    "investor",
    "sitemap",
    "site-map",
    "abuse",
    "phishing",
    "foot",
    "get-in",
    "inquiry",
    "inquiries",
    "ayuda",
    "aide",
    "contacto",
    "contato",
    "kund",
    "kunden",
    "fraud",
)

_EMAIL_RE = re.compile(r"(?i)^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$")
_EMAIL_FIND_RE = re.compile(r"[A-Z0-9._%+-]{1,64}@[A-Z0-9.-]{1,253}\.[A-Z]{2,63}", re.I)
_STANDALONE_AT_DOMAIN_RE = re.compile(
    r"(?<![A-Za-z0-9._%+])@"
    r"([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+)\b",
    re.I,
)
_DOMAIN_ONLY_FALSE_TLDS = frozenset(
    "png jpg jpeg gif webp svg ico pdf zip css js json xml html htm ts mp3 mp4 wav exe dll".split()
)

_SKIP_PATTERNS = re.compile(r"^it |\.com|\. com|random|abc", re.IGNORECASE)

_NAIVE_MULTI_LABEL_SUFFIXES = frozenset(
    {
        "co.uk",
        "com.au",
        "com.br",
        "com.hk",
        "co.nz",
        "com.sg",
        "com.mx",
        "com.ar",
        "com.tr",
        "com.tw",
        "com.cn",
    }
)

_TOKEN_SPLIT = re.compile(r"[-_]+")

_ext = tldextract.TLDExtract(cache_dir=False)

# ---------- Phase 1 / 2 prompts ----------
SYSTEM_JSON_PHASE1 = (
    "You are a careful analyst for site discovery. Reply with JSON only (no markdown fences, no commentary). "
    "Use UTF-8. Obey the schema in the user message. "
    "You have the web_search tool: you MUST call it at least once before the final answer to find **current, verifiable** "
    "first-party website roots. Do not rely on static knowledge alone — corporate domains, TLDs, and regional entry points change. "
    "Synthesize only legitimate first-party homepages/roots implied by real search results; do not list third-party aggregators, "
    "Wikipedia-only guesses, or unrelated name collisions as official_urls unless clearly the brand’s own property."
)

PHASE1_USER = """Entity (identity/brand) to cover: {org!r}

Goal: list **only first-party site entry points** (website roots) for anti-phishing contact discovery — **do not** target subpages in this step.

**web_search (required)**: Before the final JSON, use **web_search** one or more times. Example directions (adapt as needed):
- "<entity> official website" / "site:<suspected-domain>" when you can name a likely domain
- "<entity> corporate" / country or regional site names the brand actually uses
Use live results to **update** your list of first-party roots; if training-time knowledge might be stale, defer to what search and linked official pages support.

RULES:
1) Find **every distinct official base site** (different registrable domains / country sites) that legitimately belong to or clearly market as **{org!r}** (legal name, trade name, or unambiguous public brand containing this token). Regional subsidiaries, separate apex domains, and TLD variants count as **separate** sites when each is a real first-party presence.
2) **official_urls** MUST be **root site URLs only**: `https://<host>/` (or root home). **Do not** put /contact, /support, /help, paths, or deep links here — one URL per site entry (the homepage / root of that host). Include as many distinct relevant roots as you find; **no fixed count**.
3) **queries**: search strings that Phase2 can use to discover **subpages** (support, help, contact, security, abuse, phishing report, trust, legal, imprint, fraud) **within** these domains (you may name path patterns in queries, not in official_urls).
4) **target_site_hints**: apex hostnames (e.g. example.com, example.de) for all first-party properties you listed.
5) **notes**: which sites are the same company vs. collision; regional coverage; if web_search differed from prior expectations, say briefly.

Return JSON with this exact shape:
{{
  "entity": string,
  "official_urls": string[],
  "queries": string[],
  "target_site_hints": string[],
  "channels_to_prioritize": string[],
  "notes": string
}}
"""

SYSTEM_JSON_PHASE2_URLS = (
    "You are a careful analyst for official contact-channel discovery. "
    "Reply with JSON only (no markdown fences, no commentary). Use UTF-8. "
    "Obey the schema in the user message. "
    "Your ONLY task is to identify the most relevant SUBPAGE URLS — "
    "do NOT extract or return email addresses. "
    "Use web_search to find real pages; only return URLs you have actually seen in search results."
)

PHASE2_URL_USER = """Entity: {org!r}

Retrieval plan (JSON):
{plan_json}

Web search is RESTRICTED to these allowed registrable domains / hosts (tool filter): {allowed_domains_json}

Goal: Find up to 10 subpage URLs within the allowed domains that are MOST LIKELY to list official contact / support / security / abuse / phishing-report emails for {org!r}.

Rules:
1) Use web_search (with the allowed-domain filter). Run multiple targeted queries, e.g.:
   site:<domain> contact   |   site:<domain> abuse   |   site:<domain> phishing report   |   site:<domain> security   |   site:<domain> support email
2) Only return URLs whose host is within the allowed domains — no external or third-party links.
3) Prefer child/subpages over site roots (roots are already crawled separately).
4) Do NOT return attachment URLs (.pdf, .doc, .xlsx, etc.).
5) Do NOT hallucinate or guess URLs. Only include URLs that actually appeared in search results.
6) Return at most 10 URLs, ranked by likelihood of containing official email addresses.

Return ONLY JSON:
{{
  "top_urls": string[],
  "retrieval_evidence": [{{"query": string, "urls": string[], "notes": string}}]
}}"""

SYSTEM_FILTER = """\
You classify emails for ONE phishing **identity** (brand / organization name under evaluation).

**Critical:** You will receive **allowed_domains**. That list is a **noisy superset** from crawling/search — it often includes government portals (*.gov), foreign regulators, news, banks, SaaS vendors, partners, or unrelated sites that merely mentioned the brand. You MUST NOT treat every allowed_domains entry as a mailbox domain for this identity.

Your jobs:
1. Infer **identity_core_domains**: the **minimal** set of **registrable domains** that are **actually operated by / branded as** this identity’s primary commercial organization (corporate website, official alternate TLDs, clearly branded regional or JV domains).
   - EXCLUDE by default: national/state government (*.gov, *.go.jp regional gov unless the identity IS that government), securities regulators, unrelated banks/fintech vendors, random agencies, partner marketing domains, unless the identity literally IS that entity.
   - INCLUDE: main corporate domains and clearly same-brand properties (e.g. multi-TLD brand homes, documented JV domains like *brand-partner.co.jp when JV mail legitimately lives there).
   - Be strict: when several domains appear in allowed_domains, output **only** those that truly belong to this named identity’s org—not every domain that co-occurred in search results.
   - Illustration (pattern only): for a private global brand «ExampleCorp», core might be `examplecorp.com`, `examplecorp.io`, and a documented JV domain such as `partner-example.co.jp` — **not** `.gov` filings, banks, or SaaS vendors that merely appeared in allowed_domains.

2. **personal_emails**: consumer hosts (gmail, etc.), or obvious named individuals **even if** on an identity domain (e.g. firstname.lastname@corp.com).

3. **registered_domains_same_identity_from_emails**: From the **distinct registrable domains that actually appear** in the email addresses (the part after @), list **every** domain whose mail you judge as **first-party / same commercial identity** as the Organization — including regional sites (e.g. *.co.tz), subsidiaries (e.g. citibanamex.com), hyphenated JV domains (e.g. sbi-ripple.co.jp), alternate branding (e.g. rppl.app). **Only include domains that occur in at least one email in this list.** Exclude unrelated third parties (government regulators, transfer agents, generic vendors, consumer mail hosts) even if they appear in emails.
   - **Exclude standalone community / grants / accelerator program portals** whose registrable domain exists mainly for that program (e.g. apex ending in **grants.org**, such as xrplgrants.org) when the Organization name is the **corporate group / product brand** (e.g. Ripple Labs) — those are ancillary program ops mail, not primary corporate customer-contact domains. If the Organization itself is the grants program, include them.

Do NOT output official_emails — downstream code keeps every non-personal email whose domain falls under the merged allow-set.

Respond ONLY with valid JSON (no markdown fences):
{
  "identity_core_domains": ["example.com", "brand.example"],
  "registered_domains_same_identity_from_emails": ["subsidiary.example", "partner-brand.co.jp"],
  "personal_emails": ["..."],
  "reasoning": "brief"
}"""

USER_FILTER_TMPL = """\
Organization (identity): {org}

Allowed domains (NOISY — hints only): {allowed_domains_json}

Emails (derive distinct @ domains for registered_domains_same_identity_from_emails; omit standalone *.grants.org program portals when Org is the corporate brand): {emails_json}

Return identity_core_domains, registered_domains_same_identity_from_emails (cover every first-party @ domain present), and personal_emails."""

SYSTEM_CLASSIFY = """\
You are an expert at identifying official domains for a given brand or organization.

Given an organization name and a list of candidate domains extracted from its official URLs,
return ONLY the domains that are genuinely operated by / branded as this organization
(corporate website, official alternate TLDs, clearly branded regional or JV domains).

EXCLUDE:
- Government portals (*.gov, *.go.jp, etc.) unless the identity IS a government entity
- Third-party SaaS vendors, news sites, regulators, unrelated partners
- Generic CDN or hosting domains not branded as the organization

Respond ONLY with valid JSON (no markdown fences):
{
  "official_domains": ["domain1.com", "domain2.org"],
  "reasoning": "brief"
}"""

USER_CLASSIFY_TMPL = """\
Organization (identity): {org}

Candidate domains (extracted from official_urls, may include third-party noise): {domains_json}

Return the subset that are genuinely operated by / branded as this organization."""


# ---------- API key ----------
def _ensure_api_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    for p in (
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "openai_key.txt"),
        "./datasets/openai_key.txt",
    ):
        try:
            if os.path.isfile(p):
                os.environ["OPENAI_API_KEY"] = open(p, encoding="utf-8").read().strip()
                break
        except OSError:
            continue
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set and could not be read from datasets/openai_key.txt"
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


def _extract_web_search_sources(resp: Any) -> List[str]:
    sources: List[str] = []

    def _collect(src_list):
        if not isinstance(src_list, list):
            return
        for s in src_list:
            url = _get_attr(s, "url") or _get_attr(s, "source") or str(s)
            if url:
                sources.append(url)

    ws = _get_attr(resp, "web_search_call")
    if ws is not None:
        _collect(_get_attr(_get_attr(ws, "action", {}), "sources", []))

    out = _get_attr(resp, "output", [])
    if isinstance(out, list):
        for item in out:
            if _get_attr(item, "type") == "web_search_call":
                _collect(_get_attr(_get_attr(item, "action", {}), "sources", []))

    seen, uniq = set(), []
    for u in sources:
        if u and u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


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


def _fetch_url_text(url: str) -> Optional[str]:
    if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
        return None
    req = urllib.request.Request(url, headers={"User-Agent": _FETCH_UA})
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_SEC) as resp:
            raw = resp.read(FETCH_MAX_BODY_BYTES + 1)
            if len(raw) > FETCH_MAX_BODY_BYTES:
                raw = raw[:FETCH_MAX_BODY_BYTES]
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return raw.decode("latin-1", errors="ignore")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _url_visit_key(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    u = u.split("#", 1)[0]
    return u


def _parse_sitemap_locs(xml_text: str) -> Tuple[bool, List[str]]:
    is_index = bool(re.search(r"<sitemapindex[\s>]", xml_text, re.I))
    locs = re.findall(r"<loc>\s*(https?://[^\s<]+)\s*</loc>", xml_text, re.I)
    return is_index, locs


async def _fetch_sitemap_urls(
    roots: List[str],
    allowed_set: Set[str],
) -> Tuple[List[str], Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "roots_checked": 0,
        "sitemaps_fetched": 0,
        "urls_extracted": 0,
        "urls_added": 0,
    }
    all_candidates: List[str] = []
    seen: Set[str] = set()

    async def _try_fetch(url: str) -> Optional[str]:
        return await asyncio.to_thread(_fetch_url_text, url)

    async def _expand_sitemap(sm_url: str) -> List[str]:
        text = await _try_fetch(sm_url)
        if not text or "<loc>" not in text.lower():
            return []
        stats["sitemaps_fetched"] += 1
        is_index, locs = _parse_sitemap_locs(text)
        page_locs: List[str] = []
        if is_index:
            child_sms = [u for u in locs if u.startswith(("http://", "https://"))]
            for child_sm in child_sms[:SITEMAP_MAX_CHILD_SITEMAPS]:
                child_text = await _try_fetch(child_sm)
                if child_text and "<loc>" in child_text.lower():
                    stats["sitemaps_fetched"] += 1
                    _, child_locs = _parse_sitemap_locs(child_text)
                    page_locs.extend(child_locs)
        else:
            page_locs.extend(locs)
        return page_locs

    for root in roots:
        if not root or not root.startswith(("http://", "https://")):
            continue
        root = root.rstrip("/")
        stats["roots_checked"] += 1

        root_locs: List[str] = []

        for sm_path in ("/sitemap.xml", "/sitemap_index.xml"):
            locs = await _expand_sitemap(root + sm_path)
            if locs:
                root_locs.extend(locs)
                break

        if not root_locs:
            robots_text = await _try_fetch(root + "/robots.txt")
            if robots_text:
                declared = re.findall(r"(?i)^Sitemap:\s*(\S+)", robots_text, re.M)
                declared = [u.strip() for u in declared if u.strip().startswith(("http://", "https://"))]
                declared.sort(key=lambda u: -_link_priority_for_email_hunt(u))
                high_score = [u for u in declared if _link_priority_for_email_hunt(u) > 2]
                to_follow = (high_score if high_score else declared)[:SITEMAP_MAX_CHILD_SITEMAPS]
                for sm_url in to_follow:
                    locs = await _expand_sitemap(sm_url)
                    root_locs.extend(locs)

        stats["urls_extracted"] += len(root_locs)

        for u in root_locs:
            if not isinstance(u, str) or not u.startswith(("http://", "https://")):
                continue
            if not _host_matches_allowed(_hostname_from_url(u), allowed_set):
                continue
            try:
                if _ATTACHMENT_EXT_RE.search(urlparse(u).path):
                    continue
            except Exception:
                continue
            k = _url_visit_key(u)
            if k and k not in seen:
                seen.add(k)
                all_candidates.append(u)

    def _sitemap_sort_key(u: str):
        try:
            segs = [s for s in urlparse(u).path.split("/") if s]
            depth = len(segs)
            leaf_penalty = (
                1
                if segs
                and len(segs[-1]) <= 4
                and re.fullmatch(r"[A-Za-z0-9]{1,4}", segs[-1])
                else 0
            )
        except Exception:
            depth, leaf_penalty = 99, 0
        return (-_link_priority_for_email_hunt(u), leaf_penalty, depth)

    all_candidates.sort(key=_sitemap_sort_key)
    top = all_candidates[:SITEMAP_TOPK]
    stats["urls_added"] = len(top)
    return top, stats


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
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in (
        "com",
        "co",
        "gov",
        "ac",
        "org",
        "net",
        "edu",
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
    if (
        len(parts) >= 3
        and parts[-2] == "co"
        and len(parts[-1]) == 2
        and parts[-1].isalpha()
    ):
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
    allowed_norm = [
        _normalize_domain(a)
        for a in (allowed_domains or [])
        if isinstance(a, str) and a.strip()
    ]
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


# ---------- OpenAI sync (threaded timeout) ----------
def _web_tools(allowed_domains: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    tool: Dict[str, Any] = {"type": "web_search"}
    if allowed_domains:
        tool["filters"] = {"allowed_domains": allowed_domains}
    return [tool]


def _responses_create_sync(
    client: OpenAI,
    *,
    system: str,
    user: str,
    tools: Optional[List[Dict[str, Any]]],
    timeout_s: float,
    reasoning_effort: str,
) -> Any:
    payload: Dict[str, Any] = {
        "model": API_MODEL,
        "reasoning": {"effort": reasoning_effort},
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        payload["include"] = ["web_search_call.action.sources"]

    def _call():
        return client.responses.create(**payload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call)
        return fut.result(timeout=timeout_s)


def _retry_sync(client: OpenAI, name: str, fn):
    last_exc: Optional[BaseException] = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except (
            APITimeoutError,
            APIStatusError,
            OpenAIError,
            concurrent.futures.TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as e:
            last_exc = e
            time.sleep(_jitter_delay(attempt))
    raise RuntimeError(f"{name} failed after retries: {last_exc!r}")


def _phase1_plan(org: str, client: OpenAI) -> Dict[str, Any]:
    def _run():
        resp = _responses_create_sync(
            client,
            system=SYSTEM_JSON_PHASE1,
            user=PHASE1_USER.format(org=org),
            tools=_web_tools(None),
            timeout_s=PER_PHASE_TIMEOUT,
            reasoning_effort=REASONING_EFFORT_PLAN_SEARCH,
        )
        raw = getattr(resp, "output_text", "") or ""
        data = _parse_json_strict(raw)
        if not isinstance(data, dict):
            raise ValueError("plan not dict")
        p1_src = _extract_web_search_sources(resp)
        if p1_src:
            data["phase1_web_search_source_urls"] = p1_src
        return data

    return _retry_sync(client, "phase1_plan", _run)


def _phase2_search(
    org: str,
    plan: Dict[str, Any],
    allowed_domains: List[str],
    client: OpenAI,
) -> Tuple[Dict[str, Any], None]:
    plan_json = json.dumps(plan, ensure_ascii=False)
    ad_json = json.dumps(allowed_domains, ensure_ascii=False)

    def _run() -> Dict[str, Any]:
        tools = _web_tools(allowed_domains if allowed_domains else None)
        resp = _responses_create_sync(
            client,
            system=SYSTEM_JSON_PHASE2_URLS,
            user=PHASE2_URL_USER.format(org=org, plan_json=plan_json, allowed_domains_json=ad_json),
            tools=tools,
            timeout_s=PER_PHASE_TIMEOUT,
            reasoning_effort=REASONING_EFFORT_SEARCH,
        )
        raw = getattr(resp, "output_text", "") or ""
        data = _parse_json_strict(raw)
        if not isinstance(data, dict):
            raise ValueError("phase2: response not dict")
        data.setdefault("top_urls", [])
        data.setdefault("retrieval_evidence", [])
        if not isinstance(data["top_urls"], list):
            data["top_urls"] = []
        data["top_urls"] = [
            u for u in data["top_urls"] if isinstance(u, str) and u.strip().startswith(("http://", "https://"))
        ]
        data["tool_source_urls"] = _extract_web_search_sources(resp)
        data.setdefault("candidate_emails", [])
        return data

    retrieval = _retry_sync(client, "phase2_search", _run)
    return retrieval, None


# ---------- DNS ----------
def _dns_resolver_for_server(timeout_s: float, lifetime_s: float, nameservers: List[str]):
    r = dns.resolver.Resolver(configure=False)
    r.nameservers = nameservers
    r.timeout = timeout_s
    r.lifetime = lifetime_s
    return r


def _mail_domain_exists_in_dns_sync(
    domain: str,
    *,
    timeout_s: float = DNS_VERIFY_TIMEOUT,
    lifetime_s: float = DNS_VERIFY_LIFETIME,
    dns_servers: Optional[List[str]] = None,
) -> bool:
    if not _DNSPYTHON_AVAILABLE:
        return True
    d = domain.strip().lower().rstrip(".")
    if not d:
        return False
    servers = dns_servers or _DEFAULT_PUBLIC_DNS_SERVERS

    def _resolve_rdtype(rdtype: str):
        for dns_server in servers:
            res = _dns_resolver_for_server(timeout_s, lifetime_s, [dns_server])
            try:
                return res.resolve(d, rdtype)
            except dns.resolver.NXDOMAIN:
                pass
            except dns.resolver.NoAnswer:
                pass
            except dns.resolver.NoNameservers:
                pass
            except dns.resolver.LifetimeTimeout:
                pass
            except dns.exception.DNSException:
                pass
        return None

    if _resolve_rdtype("MX") is not None:
        return True
    if _resolve_rdtype("A") is not None:
        return True
    if _resolve_rdtype("AAAA") is not None:
        return True
    if _resolve_rdtype("NS") is not None:
        return True
    return False


async def _filter_final_emails_by_dns_async(
    emails: List[str], cand_src: List[Dict[str, str]]
) -> Tuple[List[str], List[Dict[str, str]]]:
    if not emails:
        return [], cand_src
    if not _DNSPYTHON_AVAILABLE:

        def _mail_dom(e: str) -> str:
            return e.rsplit("@", 1)[1].lower().strip().rstrip(".")

        by_l: Dict[str, str] = {}
        for r in cand_src:
            if isinstance(r, dict) and r.get("email"):
                by_l[str(r["email"]).strip().lower()] = (r.get("source_url") or "").strip()
        new_cand_src = [{"email": e, "source_url": by_l.get(e.lower(), "")} for e in emails]
        return emails, new_cand_src

    def _mail_dom(e: str) -> str:
        return e.rsplit("@", 1)[1].lower().strip().rstrip(".")

    unique_doms = sorted({_mail_dom(e) for e in emails if "@" in e})
    ok_by_dom: Dict[str, bool] = {}
    sem = asyncio.Semaphore(DNS_VERIFY_CONCURRENCY)

    async def _check(dom: str) -> None:
        async with sem:
            ok = await asyncio.to_thread(_mail_domain_exists_in_dns_sync, dom)
        ok_by_dom[dom] = ok

    await asyncio.gather(*[_check(d) for d in unique_doms])

    kept = [e for e in emails if "@" in e and ok_by_dom.get(_mail_dom(e), False)]

    by_l: Dict[str, str] = {}
    for r in cand_src:
        if isinstance(r, dict) and r.get("email"):
            by_l[str(r["email"]).strip().lower()] = (r.get("source_url") or "").strip()
    new_cand_src = [{"email": e, "source_url": by_l.get(e.lower(), "")} for e in kept]

    return kept, new_cand_src


def _filter_final_emails_by_dns(
    emails: List[str], cand_src: List[Dict[str, str]]
) -> Tuple[List[str], List[Dict[str, str]]]:
    return asyncio.run(_filter_final_emails_by_dns_async(emails, cand_src))


# ---------- BFS fetch ----------
async def _fetch_and_bfs_urllib(
    seeds: List[str],
    allowed_set: Set[str],
) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "mode": "urllib",
        "seeds": len(seeds),
        "fetches": 0,
        "bfs_queued": 0,
        "emails_found": 0,
        "depth_max_seen": 0,
    }
    if not seeds or RECALL_LINK_MAX_FETCH <= 0:
        return {}, stats

    email_map: Dict[str, Tuple[str, str]] = {}
    visited: Set[str] = set()
    q: deque[Tuple[str, int]] = deque()

    for su in seeds:
        if not su or not su.startswith(("http://", "https://")):
            continue
        if not _host_matches_allowed(_hostname_from_url(su), allowed_set):
            continue
        uk = _url_visit_key(su)
        if not uk or uk in visited:
            continue
        visited.add(uk)
        q.append((su, 0))
        stats["bfs_queued"] += 1

    sem = asyncio.Semaphore(FETCH_CONCURRENCY)

    async def one_fetch(u: str) -> Tuple[str, Optional[str]]:
        async with sem:
            h = await asyncio.to_thread(_fetch_url_text, u)
        return u, h

    fetches = 0
    while q and fetches < RECALL_LINK_MAX_FETCH:
        url, hop = q.popleft()
        if hop > stats["depth_max_seen"]:
            stats["depth_max_seen"] = hop
        fetches += 1
        _, page_html = await one_fetch(url)
        if not page_html:
            continue

        http_links, mto_raws = _parse_page_links(page_html, url)
        for mraw in mto_raws:
            mraw = mraw.split("?", 1)[0]
            e = _strip_trailing_junk(mraw)
            if _is_email(e):
                k = e.lower()
                if k not in email_map:
                    email_map[k] = (e, url)
        for em in _extract_emails_from_html_page(page_html):
            k = em.lower()
            if k not in email_map:
                email_map[k] = (em, url)

        if USE_RECALL_LINK_TREE and hop < RECALL_LINK_TREE_MAX_DEPTH:
            scored: List[Tuple[float, str]] = []
            seen_child: Set[str] = set()
            for nxt in http_links:
                nxt_list = _filter_candidate_urls([nxt], allowed_set)
                if not nxt_list:
                    continue
                nxt = nxt_list[0]
                nk = _url_visit_key(nxt)
                if not nk or nk in visited or nk in seen_child:
                    continue
                seen_child.add(nk)
                scored.append((_link_priority_for_email_hunt(nxt), nxt))
            scored.sort(key=lambda t: (-t[0], t[1]))
            topk = RECALL_LINK_TOPK_L0 if hop == 0 else RECALL_LINK_TOPK
            for _sc, nxt in scored[:topk]:
                nk = _url_visit_key(nxt)
                if nk not in visited:
                    visited.add(nk)
                    q.append((nxt, hop + 1))
                    stats["bfs_queued"] += 1

    stats["fetches"] = fetches
    stats["emails_found"] = len(email_map)
    return email_map, stats


async def _fetch_and_bfs_playwright(
    seeds: List[str],
    allowed_set: Set[str],
) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Any]]:
    try:
        from playwright.async_api import async_playwright
        from playwright.async_api import TimeoutError as PWTimeout
    except ImportError as exc:
        raise ImportError(
            "playwright 未安装。请运行：\n"
            "  pip install playwright\n"
            "  playwright install chromium"
        ) from exc

    stats: Dict[str, Any] = {
        "mode": "playwright",
        "seeds": len(seeds),
        "fetches": 0,
        "bfs_queued": 0,
        "emails_found": 0,
        "depth_max_seen": 0,
    }
    if not seeds or RECALL_LINK_MAX_FETCH <= 0:
        return {}, stats

    email_map: Dict[str, Tuple[str, str]] = {}
    visited: Set[str] = set()
    q: deque[Tuple[str, int]] = deque()

    for su in seeds:
        if not su or not su.startswith(("http://", "https://")):
            continue
        if not _host_matches_allowed(_hostname_from_url(su), allowed_set):
            continue
        uk = _url_visit_key(su)
        if not uk or uk in visited:
            continue
        visited.add(uk)
        q.append((su, 0))
        stats["bfs_queued"] += 1

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent=_FETCH_UA,
            java_script_enabled=True,
            ignore_https_errors=True,
        )
        sem = asyncio.Semaphore(PLAYWRIGHT_CONCURRENCY)

        async def fetch_one_pw(url: str) -> Optional[str]:
            async with sem:
                page = await ctx.new_page()
                content: Optional[str] = None
                try:
                    try:
                        await page.goto(url, wait_until="networkidle", timeout=PLAYWRIGHT_PAGE_TIMEOUT_MS)
                    except PWTimeout:
                        pass
                    except Exception:
                        pass
                    content = await page.content()
                except Exception:
                    pass
                finally:
                    await page.close()
                return content

        fetches = 0
        while q and fetches < RECALL_LINK_MAX_FETCH:
            batch: List[Tuple[str, int]] = []
            while q and len(batch) < PLAYWRIGHT_CONCURRENCY and fetches + len(batch) < RECALL_LINK_MAX_FETCH:
                batch.append(q.popleft())
            if not batch:
                break

            contents = await asyncio.gather(*[fetch_one_pw(url) for url, _ in batch])
            fetches += len(batch)

            for (url, hop), content in zip(batch, contents):
                if hop > stats["depth_max_seen"]:
                    stats["depth_max_seen"] = hop
                if not content:
                    continue

                http_links, mto_raws = _parse_page_links(content, url)
                for mraw in mto_raws:
                    mraw = mraw.split("?", 1)[0]
                    e = _strip_trailing_junk(mraw)
                    if _is_email(e):
                        k = e.lower()
                        if k not in email_map:
                            email_map[k] = (e, url)
                for em in _extract_emails_from_html_page(content):
                    k = em.lower()
                    if k not in email_map:
                        email_map[k] = (em, url)

                if USE_RECALL_LINK_TREE and hop < RECALL_LINK_TREE_MAX_DEPTH and fetches < RECALL_LINK_MAX_FETCH:
                    scored: List[Tuple[float, str]] = []
                    seen_child: Set[str] = set()
                    for nxt in http_links:
                        nxt_list = _filter_candidate_urls([nxt], allowed_set)
                        if not nxt_list:
                            continue
                        nxt = nxt_list[0]
                        nk = _url_visit_key(nxt)
                        if not nk or nk in visited or nk in seen_child:
                            continue
                        seen_child.add(nk)
                        scored.append((_link_priority_for_email_hunt(nxt), nxt))
                    scored.sort(key=lambda t: (-t[0], t[1]))
                    topk = RECALL_LINK_TOPK_L0 if hop == 0 else RECALL_LINK_TOPK
                    for _sc, nxt in scored[:topk]:
                        nk = _url_visit_key(nxt)
                        if nk not in visited:
                            visited.add(nk)
                            q.append((nxt, hop + 1))
                            stats["bfs_queued"] += 1

        await browser.close()

    stats["fetches"] = fetches
    stats["emails_found"] = len(email_map)
    return email_map, stats


async def _run_fetch_merge_async(
    plan: Dict[str, Any],
    retrieval: Dict[str, Any],
    allowed: List[str],
    llm_candidates: List[str],
) -> Tuple[List[str], List[Dict[str, str]]]:
    plan = dict(plan)
    plan["official_urls"] = _normalize_official_urls_to_site_roots(plan.get("official_urls"))
    allowed_set = _allowed_hosts_set(allowed)

    raw_top_urls: List[str] = retrieval.get("top_urls") or []
    filtered_top_urls = _filter_candidate_urls(raw_top_urls, allowed_set)

    sitemap_extra, _ = await _fetch_sitemap_urls(plan.get("official_urls") or [], allowed_set)

    seed_seen: Set[str] = set()
    bfs_seeds: List[str] = []
    for u in list(plan.get("official_urls") or []) + filtered_top_urls + sitemap_extra:
        k = _url_visit_key(u)
        if k and k not in seed_seen:
            seed_seen.add(k)
            bfs_seeds.append(u)

    if USE_PLAYWRIGHT:
        email_src_map, _ = await _fetch_and_bfs_playwright(bfs_seeds, allowed_set)
    else:
        email_src_map, _ = await _fetch_and_bfs_urllib(bfs_seeds, allowed_set)

    merged_order: List[str] = []
    seen_lower: Set[str] = set()
    for k, (em, _) in email_src_map.items():
        if k not in seen_lower:
            seen_lower.add(k)
            merged_order.append(em)
    for e in llm_candidates:
        k = e.lower()
        if k not in seen_lower:
            seen_lower.add(k)
            email_src_map[k] = (e, "")
            merged_order.append(e)

    merged_src = [
        {"email": em, "source_url": email_src_map.get(em.lower(), ("", ""))[1]} for em in merged_order
    ]
    return merged_order, merged_src


def _fetch_merge_page_emails(
    plan: Dict[str, Any],
    retrieval: Dict[str, Any],
    allowed: List[str],
    llm_candidates: List[str],
) -> Tuple[List[str], List[Dict[str, str]]]:
    return asyncio.run(_run_fetch_merge_async(plan, retrieval, allowed, llm_candidates))


def _responses_create_plain_sync(
    client: OpenAI,
    *,
    system: str,
    user: str,
    timeout_s: float,
    reasoning_effort: str,
) -> Any:
    payload: Dict[str, Any] = {
        "model": API_MODEL,
        "reasoning": {"effort": reasoning_effort},
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    def _call():
        return client.responses.create(**payload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call)
        return fut.result(timeout=timeout_s)


def _sources_map_from_rows(rows: List[Dict[str, str]]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for r in rows or []:
        if isinstance(r, dict) and r.get("email"):
            m[str(r["email"]).lower().strip()] = (r.get("source_url") or "").strip()
    return m


def _identity_llm_filter_sync_fixed(
    client: OpenAI,
    org: str,
    allowed_domains: List[str],
    emails: List[str],
    merged_src: List[Dict[str, str]],
) -> Tuple[List[str], List[Dict[str, str]]]:
    if not emails:
        return [], []

    def _run():
        resp = _responses_create_plain_sync(
            client,
            system=SYSTEM_FILTER,
            user=USER_FILTER_TMPL.format(
                org=org,
                allowed_domains_json=json.dumps(allowed_domains, ensure_ascii=False),
                emails_json=json.dumps(emails, ensure_ascii=False),
            ),
            timeout_s=PER_PHASE_TIMEOUT,
            reasoning_effort=REASONING_EFFORT_FILTER,
        )
        raw = getattr(resp, "output_text", "") or ""
        data = _parse_json_strict(raw)
        if not isinstance(data, dict):
            raise ValueError("filter response is not a dict")
        personal = data.get("personal_emails")
        identity_core = data.get("identity_core_domains")
        keep_raw = data.get("registered_domains_same_identity_from_emails") or []
        if not isinstance(keep_raw, list):
            keep_raw = []
        if not isinstance(personal, list):
            raise ValueError("missing or invalid personal_emails")
        if not isinstance(identity_core, list) or not identity_core:
            raise ValueError("missing or empty identity_core_domains")

        emails_lower = {e.lower(): e for e in emails}

        def _norm_from_input(lst: List[Any]) -> List[str]:
            out: List[str] = []
            for x in lst:
                if not isinstance(x, str):
                    continue
                lo = x.lower()
                if lo in emails_lower:
                    out.append(emails_lower[lo])
            return out

        personal_clean = _norm_from_input(personal)
        cores_raw = [c for c in identity_core if isinstance(c, str) and c.strip()]
        if not cores_raw:
            raise ValueError("identity_core_domains has no valid strings")

        keep_validated = _validate_llm_keep_domains(keep_raw, emails)

        official_emails, _tp, _ce = _apply_identity_core_filter(
            emails,
            personal_clean,
            cores_raw,
            allowed_domains,
            keep_validated,
            org,
        )
        return official_emails

    official = _retry_sync(client, "identity_llm_filter", _run)
    smap = _sources_map_from_rows(merged_src)
    return official, [{"email": e, "source_url": smap.get(e.lower().strip(), "")} for e in official]


def _classify_official_domains_from_urls_sync(
    client: OpenAI,
    org: str,
    candidate_domains: List[str],
) -> List[str]:
    if not candidate_domains:
        return []

    def _run():
        resp = _responses_create_plain_sync(
            client,
            system=SYSTEM_CLASSIFY,
            user=USER_CLASSIFY_TMPL.format(
                org=org,
                domains_json=json.dumps(candidate_domains, ensure_ascii=False),
            ),
            timeout_s=PER_PHASE_TIMEOUT,
            reasoning_effort=REASONING_EFFORT_CLASSIFY,
        )
        raw = getattr(resp, "output_text", "") or ""
        data = _parse_json_strict(raw)
        if not isinstance(data, dict):
            raise ValueError("classify response is not a dict")
        official = data.get("official_domains")
        if not isinstance(official, list):
            raise ValueError("missing official_domains")
        return [d for d in official if isinstance(d, str) and d.strip()]

    raw_list = _retry_sync(client, "classify_official_domains", _run)
    cand_set = set(candidate_domains)
    kept: List[str] = []
    seen_k: Set[str] = set()
    for d in raw_list:
        if not isinstance(d, str):
            continue
        dn = d.strip().lower().rstrip(".")
        rd = _registered_domain_etld1(dn) or dn
        key = rd if rd in cand_set else (dn if dn in cand_set else "")
        if not key or key in seen_k:
            continue
        if rd in cand_set or dn in cand_set:
            seen_k.add(key)
            kept.append(rd if rd in cand_set else dn)
    return kept


def _merge_official_email_domains(
    plan: Dict[str, Any],
    filtered_emails: List[str],
    valid_from_urls: List[str],
) -> List[str]:
    merged: List[str] = list(valid_from_urls)
    merged_set: Set[str] = set(valid_from_urls)
    for email in filtered_emails:
        d = _extract_etld1_from_email(email)
        if d and d not in merged_set:
            merged_set.add(d)
            merged.append(d)
    return sorted(merged)


def _extract_official_email_domains_post_filter(
    client: OpenAI,
    org: str,
    plan: Dict[str, Any],
    filtered_emails: List[str],
) -> List[str]:
    """
    Mirrors extract_official_domains_from_urls.py: LLM-clean URL registrable domains,
    then union with eTLD+1 from the (already filtered) mailbox list. Batch jsonl uses
    this domain list as a separate artifact; it does not change 5.6 email rows.
    """
    urls: List[str] = (plan or {}).get("official_urls") or []
    seen_cands: Set[str] = set()
    candidate_domains: List[str] = []
    for url in urls:
        d = _extract_etld1_from_url(url)
        if d and d not in seen_cands:
            seen_cands.add(d)
            candidate_domains.append(d)

    valid_from_urls = _classify_official_domains_from_urls_sync(client, org, candidate_domains)
    return _merge_official_email_domains(plan, filtered_emails, valid_from_urls)

