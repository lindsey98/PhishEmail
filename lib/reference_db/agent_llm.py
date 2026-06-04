"""OpenAI-backed phases and filters for the KB expansion agent."""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import APIStatusError, APITimeoutError, OpenAI, OpenAIError

from .agent_constants import (
    API_MODEL,
    MAX_RETRIES,
    PER_PHASE_TIMEOUT,
    PHASE1_USER,
    PHASE2_URL_USER,
    REASONING_EFFORT_CLASSIFY,
    REASONING_EFFORT_FILTER,
    REASONING_EFFORT_PLAN_SEARCH,
    REASONING_EFFORT_SEARCH,
    SYSTEM_CLASSIFY,
    SYSTEM_FILTER,
    SYSTEM_JSON_PHASE1,
    SYSTEM_JSON_PHASE2_URLS,
    USER_CLASSIFY_TMPL,
    USER_FILTER_TMPL,
)
from .agent_helpers import (
    _apply_identity_core_filter,
    _extract_etld1_from_email,
    _extract_etld1_from_url,
    _get_attr,
    _jitter_delay,
    _parse_json_strict,
    _registered_domain_etld1,
    _validate_llm_keep_domains,
)


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
        raise RuntimeError("OPENAI_API_KEY is not set and could not be read from datasets/openai_key.txt")


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
