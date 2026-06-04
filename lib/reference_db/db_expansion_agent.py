"""
Knowledge-base agent: end-to-end official email discovery for one identity.

Orchestration only; helpers live in agent_utils.py.
"""

from __future__ import annotations

from typing import Dict, List, Union

from openai import OpenAI

from .agent_utils import (
    _allowed_domains_from_plan,
    _clean_emails,
    _ensure_api_key,
    _extract_official_email_domains_post_filter,
    _fetch_merge_page_emails,
    _filter_final_emails_by_dns,
    _identity_llm_filter_sync_fixed,
    _normalize_official_urls_to_site_roots,
    _phase1_plan,
    _phase2_search,
)


def lookup_emails(
    org: str,
    *,
    with_sources: bool = False,
    raise_on_error: bool = False,
) -> Union[List[str], List[Dict[str, str]]]:
    if not isinstance(org, str) or not org.strip():
        return []

    _ensure_api_key()

    with OpenAI() as client:
        try:
            plan = _phase1_plan(org.strip(), client)
            plan["official_urls"] = _normalize_official_urls_to_site_roots(plan.get("official_urls"))
            allowed = _allowed_domains_from_plan(plan)
            retrieval, _ = _phase2_search(org.strip(), plan, allowed, client)
        except Exception:
            if raise_on_error:
                raise
            return []

        llm_candidates = _clean_emails(list(retrieval.get("candidate_emails") or []))

        try:
            merged_emails, merged_src = _fetch_merge_page_emails(plan, retrieval, allowed, llm_candidates)
        except Exception:
            if raise_on_error:
                raise
            merged_emails = llm_candidates
            merged_src = retrieval.get("candidate_email_sources") or []

        merged_emails = _clean_emails(merged_emails)

        try:
            final_emails, final_src = _filter_final_emails_by_dns(merged_emails, merged_src)
        except Exception:
            if raise_on_error:
                raise
            final_emails, final_src = merged_emails, merged_src

        try:
            final_emails, final_src = _identity_llm_filter_sync_fixed(
                client, org.strip(), allowed, final_emails, final_src
            )
        except Exception:
            if raise_on_error:
                raise

        try:
            _ = _extract_official_email_domains_post_filter(client, org.strip(), plan, final_emails)
        except Exception:
            if raise_on_error:
                raise

    if not with_sources:
        return final_emails

    by_email = {(r.get("email") or "").lower(): (r.get("source_url") or "") for r in final_src if isinstance(r, dict)}
    return [{"email": e, "source_url": by_email.get(e.lower(), "")} for e in final_emails]
