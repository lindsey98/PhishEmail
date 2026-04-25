from __future__ import annotations

import os
import re
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from openai import OpenAI, APIStatusError, APITimeoutError, OpenAIError

try:
    import dns.exception
    import dns.resolver

    _DNSPYTHON_AVAILABLE = True
except ImportError:
    _DNSPYTHON_AVAILABLE = False

from lib.reference_db.agent_utils import (
    _jitter_delay,
    _clean_emails,
    _web_tools,
    _parse_json_strict,
    _normalize_sources,
    _allowed_domains_from_plan,
    _extract_web_search_sources,
    API_MODEL,
    MAX_RETRIES,
    SYSTEM_JSON,
    SYSTEM_JSON_PHASE2,
    PHASE1_USER,
    PHASE2_USER,
    REASONING_EFFORT_PLAN,
    REASONING_EFFORT_SEARCH,
    PER_PHASE_TIMEOUT,
)


# ---------- HTML fetch config (Phase 2 后：抓引用页确定性抽取邮箱) ----------
FETCH_MAX_URLS_PER_IDENTITY = 16
FETCH_CONCURRENCY = 4
FETCH_TIMEOUT_SEC = 20
FETCH_MAX_BODY_BYTES = 2_000_000
_FETCH_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# ---------- DNS 验证 config (与 validate_knowphish_domains_dns_mx 对齐) ----------
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


# ---------- Email / host regex ----------
_EMAIL_RE = re.compile(r"(?i)^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$")
_EMAIL_FIND_RE = re.compile(r"[A-Z0-9._%+-]{1,64}@[A-Z0-9.-]{1,253}\.[A-Z]{2,63}", re.I)


def _is_email(s: str) -> bool:
    return isinstance(s, str) and bool(_EMAIL_RE.match(s.strip()))


def _strip_trailing_junk(addr: str) -> str:
    s = addr.strip()
    while s and s[-1] in ".,;:)]}>\"'":
        s = s[:-1]
    return s


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
    return out


# ---------- Allowed-host matching (跟 Phase 2 tool filter 一致) ----------
def _allowed_hosts_set(allowed_domains: List[str]) -> Set[str]:
    s: Set[str] = set()
    for x in allowed_domains or []:
        x = (x or "").strip().lower().rstrip(".")
        if x:
            s.add(x)
    return s


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


def _email_domain_matches_allowed(email: str, allowed_set: Set[str]) -> bool:
    if "@" not in email:
        return False
    ed = email.rsplit("@", 1)[1].lower().strip().rstrip(".")
    return _host_matches_allowed(ed, allowed_set)


# ---------- HTML fetch + email merge ----------
def _collect_urls_for_fetch(
    retrieval: Dict[str, Any], plan: Dict[str, Any], max_urls: int
) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []

    def add(u: Any) -> None:
        if not isinstance(u, str):
            return
        u = u.strip()
        if not u.startswith(("http://", "https://")):
            return
        if u in seen:
            return
        seen.add(u)
        ordered.append(u)

    for u in retrieval.get("tool_source_urls") or []:
        add(u)
    for row in retrieval.get("retrieval_evidence") or []:
        if isinstance(row, dict):
            for u in row.get("urls") or []:
                add(u)
    for row in retrieval.get("candidate_email_sources") or []:
        if isinstance(row, dict):
            add(row.get("source_url") or "")
    for u in plan.get("official_urls") or []:
        add(u)

    return ordered[:max_urls]


def _fetch_url_text(url: str) -> Optional[str]:
    if not isinstance(url, str) or not (
        url.startswith("http://") or url.startswith("https://")
    ):
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


def _fetch_merge_page_emails(
    plan: Dict[str, Any],
    retrieval: Dict[str, Any],
    allowed: List[str],
    llm_candidates: List[str],
) -> Tuple[List[str], List[Dict[str, str]]]:
    allowed_set = _allowed_hosts_set(allowed)
    raw_urls = _collect_urls_for_fetch(retrieval, plan, FETCH_MAX_URLS_PER_IDENTITY)
    urls = [u for u in raw_urls if _host_matches_allowed(_hostname_from_url(u), allowed_set)]

    fetch_map: Dict[str, Tuple[str, str]] = {}  # email_lower -> (email_original_case, src_url)

    if urls:
        with ThreadPoolExecutor(max_workers=FETCH_CONCURRENCY) as ex:
            future_to_url = {ex.submit(_fetch_url_text, u): u for u in urls}
            for fut in as_completed(future_to_url):
                url = future_to_url[fut]
                try:
                    html = fut.result()
                except Exception:
                    html = None
                if not html:
                    continue
                for em in _extract_emails_from_text(html):
                    if not _email_domain_matches_allowed(em, allowed_set):
                        continue
                    k = em.lower()
                    if k not in fetch_map:
                        fetch_map[k] = (em, url)

    cand_src_raw = retrieval.get("candidate_email_sources") or []
    by_l: Dict[str, str] = {}
    for r in cand_src_raw:
        if isinstance(r, dict) and r.get("email"):
            ek = str(r["email"]).strip().lower()
            by_l[ek] = (r.get("source_url") or "").strip()

    for k, (_em, src_url) in fetch_map.items():
        if k not in by_l or not by_l[k]:
            by_l[k] = src_url

    merged: List[str] = []
    seen_m: Set[str] = set()
    for e in llm_candidates:
        k = e.lower()
        if k not in seen_m:
            seen_m.add(k)
            merged.append(e)
    for k, (em, _) in fetch_map.items():
        if k not in seen_m:
            seen_m.add(k)
            merged.append(em)

    new_cand_src = [
        {"email": e, "source_url": by_l.get(e.lower(), "")} for e in merged
    ]
    return merged, new_cand_src


def _dns_resolver_for_server(timeout_s: float, lifetime_s: float, nameservers: List[str]):
    r = dns.resolver.Resolver(configure=False)
    r.nameservers = nameservers
    r.timeout = timeout_s
    r.lifetime = lifetime_s
    return r


def _mail_domain_exists_in_dns(
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

    for rt in ("MX", "A", "AAAA", "NS"):
        if _resolve_rdtype(rt) is not None:
            return True
    return False


def _filter_final_emails_by_dns(
    emails: List[str], cand_src: List[Dict[str, str]]
) -> Tuple[List[str], List[Dict[str, str]]]:
    if not emails:
        return [], cand_src
    if not _DNSPYTHON_AVAILABLE:
        return emails, cand_src

    def _mail_dom(e: str) -> str:
        return e.rsplit("@", 1)[1].lower().strip().rstrip(".")

    unique_doms = sorted({_mail_dom(e) for e in emails if "@" in e})
    ok_by_dom: Dict[str, bool] = {}

    with ThreadPoolExecutor(max_workers=DNS_VERIFY_CONCURRENCY) as ex:
        future_to_dom = {
            ex.submit(_mail_domain_exists_in_dns, dom): dom for dom in unique_doms
        }
        for fut in as_completed(future_to_dom):
            dom = future_to_dom[fut]
            try:
                ok_by_dom[dom] = bool(fut.result())
            except Exception:
                ok_by_dom[dom] = False

    kept = [e for e in emails if "@" in e and ok_by_dom.get(_mail_dom(e), False)]

    by_l: Dict[str, str] = {}
    for r in cand_src:
        if isinstance(r, dict) and r.get("email"):
            by_l[str(r["email"]).strip().lower()] = (r.get("source_url") or "").strip()
    new_cand_src = [{"email": e, "source_url": by_l.get(e.lower(), "")} for e in kept]
    return kept, new_cand_src


# ---------- API key ----------
def _ensure_api_key() -> None:
    """Load OPENAI_API_KEY lazily; env var wins, file is fallback."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    key_path = "./datasets/openai_key.txt"
    if os.path.exists(key_path):
        with open(key_path, encoding="utf-8") as f:
            k = f.read().strip()
        if k:
            os.environ["OPENAI_API_KEY"] = k


# ---------- Low-level request ----------
def _responses_create(
    *,
    client: OpenAI,
    system: str,
    user: str,
    tools: Optional[List[Dict[str, Any]]],
    timeout_s: float,
    reasoning_effort: str = "low",
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

    # The OpenAI SDK accepts a per-request `timeout` kwarg.
    return client.with_options(timeout=timeout_s).responses.create(**payload)


def _retry_phase(name: str, fn):
    """Run `fn()` with exponential-jitter backoff on transient errors."""
    last_exc: Optional[BaseException] = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except (
            APITimeoutError,
            APIStatusError,
            OpenAIError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as e:
            last_exc = e
            time.sleep(_jitter_delay(attempt))
    raise RuntimeError(f"{name} failed after retries: {last_exc!r}")


# ---------- Phases ----------
def _phase1_plan(org: str, client: OpenAI) -> Dict[str, Any]:
    def _run():
        resp = _responses_create(
            client=client,
            system=SYSTEM_JSON,
            user=PHASE1_USER.format(org=org),
            tools=None,
            timeout_s=PER_PHASE_TIMEOUT,
            reasoning_effort=REASONING_EFFORT_PLAN,
        )
        raw = getattr(resp, "output_text", "") or ""
        data = _parse_json_strict(raw)
        if not isinstance(data, dict):
            raise ValueError("plan not dict")
        return data

    return _retry_phase("phase1_plan", _run)


def _phase2_search(
    org: str,
    plan: Dict[str, Any],
    allowed_domains: List[str],
    client: OpenAI,
) -> Tuple[Dict[str, Any], List[str]]:
    plan_json = json.dumps(plan, ensure_ascii=False)
    ad_json = json.dumps(allowed_domains, ensure_ascii=False)
    user_msg = PHASE2_USER.format(
        org=org, plan_json=plan_json, allowed_domains_json=ad_json
    )

    def _run():
        tools = _web_tools(allowed_domains if allowed_domains else None)
        resp = _responses_create(
            client=client,
            system=SYSTEM_JSON_PHASE2,
            user=user_msg,
            tools=tools,
            timeout_s=PER_PHASE_TIMEOUT,
            reasoning_effort=REASONING_EFFORT_SEARCH,
        )
        raw = getattr(resp, "output_text", "") or ""
        data = _parse_json_strict(raw)
        if not isinstance(data, dict):
            raise ValueError("retrieval not dict")
        sources = _extract_web_search_sources(resp)
        data.setdefault("retrieval_evidence", [])
        data.setdefault("candidate_emails", [])
        cand = _clean_emails([str(x) for x in (data.get("candidate_emails") or [])])
        data["candidate_emails"] = cand
        data["candidate_email_sources"] = _normalize_sources(data, cand)
        data["tool_source_urls"] = sources
        return data, sources

    return _retry_phase("phase2_search", _run)


# ---------- Public API ----------
def lookup_emails(
    org: str,
    *,
    with_sources: bool = False,
    raise_on_error: bool = False,
) -> Union[List[str], List[Dict[str, str]]]:
    if not isinstance(org, str) or not org.strip():
        return []

    _ensure_api_key()

    # Fresh client per call. Cheap — mostly just an httpx session — and
    # avoids any cross-call state (httpx transports, connection pools).
    # `with` guarantees the session is closed even if an error is raised.
    with OpenAI() as client:
        try:
            plan = _phase1_plan(org, client)
            allowed = _allowed_domains_from_plan(plan)
            retrieval, _ = _phase2_search(org, plan, allowed, client)
        except Exception:
            if raise_on_error:
                raise
            return []

    llm_candidates = _clean_emails(list(retrieval.get("candidate_emails") or []))

    # Fetch allowed-domain pages, deterministically extract emails, merge with LLM candidates.
    try:
        merged_emails, merged_src = _fetch_merge_page_emails(
            plan, retrieval, allowed, llm_candidates
        )
    except Exception:
        if raise_on_error:
            raise
        merged_emails = llm_candidates
        merged_src = retrieval.get("candidate_email_sources") or []

    merged_emails = _clean_emails(merged_emails)

    # DNS filter: drop mail domains that resolve on *no* public resolver.
    try:
        final_emails, final_src = _filter_final_emails_by_dns(merged_emails, merged_src)
    except Exception:
        if raise_on_error:
            raise
        final_emails, final_src = merged_emails, merged_src

    if not with_sources:
        return final_emails

    by_email = {
        (r.get("email") or "").lower(): (r.get("source_url") or "")
        for r in final_src
        if isinstance(r, dict)
    }
    return [
        {"email": e, "source_url": by_email.get(e.lower(), "")} for e in final_emails
    ]


