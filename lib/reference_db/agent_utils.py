from __future__ import annotations

import os
import re
import json
import asyncio
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

# ---------- Config ----------
API_MODEL = "o4-mini"
PER_PHASE_TIMEOUT = 120.0
MAX_RETRIES = 3
BASE_BACKOFF = 0.6
REASONING_EFFORT_PLAN = "low"
REASONING_EFFORT_SEARCH = "medium"


# ---------- Regex ----------
_EMAIL_RE = re.compile(r"(?i)^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$")
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I | re.MULTILINE)


# ---------- Prompts (unchanged) ----------
SYSTEM_JSON = (
    "You are a careful analyst. Reply with JSON only (no markdown fences, no commentary). "
    "Use UTF-8. Obey the schema described in the user message."
)

SYSTEM_JSON_PHASE2 = (
    "You are a careful analyst for official contact-channel discovery. "
    "Reply with JSON only (no markdown fences, no commentary). Use UTF-8. "
    "Obey the schema in the user message. "
    "Hard rules: (1) List ONLY email addresses whose mail domain is the brand’s official "
    "registrable domain or a hostname under it, matching the tool’s allowed domains / the plan. "
    "(2) Prefer an empty candidate_emails list over wrong or unverified addresses—precision over recall. "
    "(3) Do NOT list addresses on Zendesk, Intercom, Freshdesk, generic ticketing/ESP hosts, or other "
    "third-party mail domains, or other user-posted forums, unless the retrieved allowed-domain page explicitly presents that exact "
    "address as this brand’s public official contact (not a generic “contact us at support@...” on a help portal). "
    "When in doubt, omit the email. "
    "(4) For every email in candidate_emails you MUST include a matching entry in candidate_email_sources: "
    'each item is {"email": string, "source_url": string} with source_url a full http(s) URL of the page where that address appeared; the URL must exist, be accessible, and the email must exactly appear on that URL.'
)

PHASE1_USER = """Entity (identity/brand to investigate): {org!r}

You are finding the LEGITIMATE brand's official customer-contact channels (for anti-phishing ground truth), not the scammer's domain.

STEP 1 — DISAMBIGUATE THE ENTITY (think before planning):
Before listing URLs, decide:
- What does {org!r} most likely refer to? A specific company? A sub-service (e.g. "MyJCB" = JCB card's online portal)? A regional subsidiary (e.g. "Amazon JP" vs "Amazon US")? A product line of a larger corp?
- If the name is ambiguous, choose the SINGLE most likely interpretation for an anti-phishing ground-truth database (i.e. the target that phishers would impersonate), and state it in `notes`.
- If the name clearly refers to a SUB-SERVICE of a larger company (e.g. "MyJCB" under JCB, "Google Pay" under Google, "Apple ID" under Apple), the official domains of the parent ARE in scope — include the parent's apex domain in target_site_hints, because contact emails are almost always centralized at the parent.

STEP 2 — PLAN URLS AGGRESSIVELY:
Emails appear on more pages than just /contact. Plan URLs covering ALL of these channel types when plausible:
- Customer support: /contact, /support, /help, /help-center, /customer-service, /feedback
- Corporate / legal: /privacy, /privacy-policy, /terms, /legal, /cookies (these almost always list a data-protection / legal email)
- Security / abuse: /security, /security.txt, /abuse, /report, /responsible-disclosure, /bug-bounty, /phishing
- Press / investor: /press, /media, /newsroom, /investor-relations, /about/contact
- Regional TLDs: if the brand operates in multiple countries, include both the global apex and regional sites (e.g. brand.com AND brand.co.jp AND brand.com.sg) when relevant.

STEP 3 — RETURN JSON with this exact shape:
{{
  "entity": string,                     // your disambiguated interpretation (e.g. "MyJCB (JCB Card's online member portal)")
  "official_urls": string[],            // 6–15 full https URLs; MUST cover multiple channel types above, not just /contact
  "queries": string[],                  // 6–12 concrete web search queries
  "target_site_hints": string[],        // apex/hostnames, INCLUDING parent brand's apex if entity is a sub-service
  "channels_to_prioritize": string[],   // subset of: support, contact, helpdesk, security, abuse, privacy, legal, press
  "notes": string                       // explicit disambiguation reasoning + any ambiguity risks
}}

QUERY DESIGN RULES:
- Mix direct ("<brand> contact email", "<brand> customer service email") with structural ("site:brand.com contact", "<brand> privacy policy email").
- For sub-services, include the parent: 'JCB customer service email' as well as 'MyJCB support'.
- For brands that rarely publish emails (lotteries, government, social platforms), still include privacy/legal/abuse queries — those pages usually DO list an email for compliance reasons.

RULE: Focus ONLY on the entity (or its parent if the entity is a sub-service). DO NOT drift into unrelated partners or acquirers.
"""

PHASE2_USER = """Entity: {org!r}

Retrieval plan (JSON):
{plan_json}

Web search is RESTRICTED to these allowed registrable domains / hosts (tool filter): {allowed_domains_json}

GOAL: find AT LEAST ONE official email address for this entity when one plausibly exists. Empty output is acceptable only when you have genuinely searched the plan and nothing qualifies.

ACCEPTABLE EMAIL TYPES (any of these count as official contact channels):
- Customer support / contact: support@, contact@, help@, info@, customerservice@, customercare@
- Security / abuse: security@, abuse@, phishing@, report@, soc@
- Privacy / legal: privacy@, dpo@, gdpr@, legal@, compliance@
- Press / PR / investor: press@, media@, pr@, ir@, investor@
- Technical / service-specific: admin@, postmaster@, noreply@ (only if shown as brand's own)

ALL are valid for anti-phishing ground truth — you don't need a "customer-service-only" email.

WHERE TO LOOK (not just /contact):
- Privacy policy / terms of service pages (contain privacy@ / dpo@ / legal@)
- /security.txt, /.well-known/security.txt (RFC 9116; contains security contact)
- Footer of homepage
- About / company info pages
- Regional subsites (if allowed domains include regional TLDs)

HARD RULES (violating these is worse than returning empty):
1. Output ONLY email addresses whose domains belong to the brand's official registrable domain OR a subdomain of it, per the allowed list. Reject any address outside that scope.
2. Do NOT output addresses on third-party help-desk domains (zendesk.com, freshdesk.com, intercom.com, hubspot.com, salesforce.com, servicenow.com, etc.) UNLESS an allowed-domain page explicitly presents that exact address as the brand's official public contact.
3. Do NOT fabricate. Every email must literally appear on an allowed-domain page you retrieved. If in doubt, omit.
4. Do NOT output emails that look like user-generated content (forum posts, support tickets quoted on help pages, examples in documentation).

TASKS:
1) Use web_search aggressively — run multiple queries from the plan. If the first few return no email-bearing pages, try:
   - Privacy / legal page queries: 'site:<brand_domain> privacy policy email', 'site:<brand_domain> contact us'
   - Security queries: 'site:<brand_domain> security.txt', 'site:<brand_domain> report abuse'
   - Footer / about queries: 'site:<brand_domain> about contact'
2) Retrieve pages and extract every email address literally present (not inferred, not composed from text).
3) Filter strictly: apply the HARD RULES above.
4) Deduplicate.

FALLBACK POLICY — only if after honest searching NO email is found:
- Return empty candidate_emails
- In retrieval_evidence, include the queries you tried and the URLs you retrieved, with a note explaining WHY no email could be extracted (e.g. "brand uses web form only", "all results were 404", "all emails found were on third-party domains and not confirmed as official")

Final JSON must include:
- retrieval_evidence: list of {{ "query": string, "urls": string[], "notes": string }}
- candidate_emails: deduplicated list of email strings (may be empty only per the Fallback Policy)
- candidate_email_sources: list of {{ "email": string, "source_url": string }}, one per candidate_emails, same order; source_url is the http(s) URL on which that exact address appeared (use "" only if tool citations were lost).

Return ONLY JSON with keys: retrieval_evidence, candidate_emails, candidate_email_sources."""


# ---------- Helpers ----------
def _jitter_delay(attempt: int) -> float:
    base = BASE_BACKOFF * (2 ** attempt)
    return random.uniform(base * 0.5, base * 1.5)


def _parse_json_strict(text: str) -> Any:
    if not text or not isinstance(text, str):
        raise ValueError("empty")
    s = _JSON_FENCE.sub("", text.strip()).strip()
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


def _naive_registrable_domain(hostname: str) -> str:
    parts = hostname.lower().strip().split(".")
    if len(parts) < 2:
        return hostname.lower()
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in (
        "com", "co", "gov", "ac", "org", "net", "edu",
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
            add(_naive_registrable_domain(h))

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
                add(_naive_registrable_domain(hh))
        else:
            add(h)
            if "." in h:
                add(_naive_registrable_domain(h))
    return out


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


def _normalize_sources(
    data: Dict[str, Any], candidate_emails: List[str]
) -> List[Dict[str, str]]:
    raw = data.get("candidate_email_sources")
    m: Dict[str, str] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            e = (item.get("email") or item.get("e") or "").strip()
            u = (item.get("source_url") or item.get("url") or item.get("source") or "").strip()
            if not _is_email(e):
                continue
            if u.startswith(("http://", "https://")):
                m[e.lower()] = u
    return [{"email": em, "source_url": m.get(em.lower(), "")} for em in candidate_emails]


# ---------- OpenAI calls ----------
def _web_tools(allowed_domains: Optional[List[str]]) -> List[Dict[str, Any]]:
    tool: Dict[str, Any] = {"type": "web_search"}
    if allowed_domains:
        tool["filters"] = {"allowed_domains": allowed_domains}
    return [tool]
