"""Configuration, regexes, and LLM prompt strings for the KB expansion agent."""

from __future__ import annotations

import re

import tldextract

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
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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
