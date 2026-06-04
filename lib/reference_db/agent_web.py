"""Network layer for the KB expansion agent: HTTP fetch, sitemap/link BFS crawling
(urllib + optional Playwright) and DNS MX checks."""

from __future__ import annotations

import asyncio
import re
import urllib.error
import urllib.request
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

try:
    import dns.exception
    import dns.resolver

    _DNSPYTHON_AVAILABLE = True
except ImportError:
    _DNSPYTHON_AVAILABLE = False

from .agent_constants import (
    _ATTACHMENT_EXT_RE,
    _DEFAULT_PUBLIC_DNS_SERVERS,
    _FETCH_UA,
    DNS_VERIFY_CONCURRENCY,
    DNS_VERIFY_LIFETIME,
    DNS_VERIFY_TIMEOUT,
    FETCH_CONCURRENCY,
    FETCH_MAX_BODY_BYTES,
    FETCH_TIMEOUT_SEC,
    PLAYWRIGHT_CONCURRENCY,
    PLAYWRIGHT_PAGE_TIMEOUT_MS,
    RECALL_LINK_MAX_FETCH,
    RECALL_LINK_TOPK,
    RECALL_LINK_TOPK_L0,
    RECALL_LINK_TREE_MAX_DEPTH,
    SITEMAP_MAX_CHILD_SITEMAPS,
    SITEMAP_TOPK,
    USE_PLAYWRIGHT,
    USE_RECALL_LINK_TREE,
)
from .agent_helpers import (
    _allowed_hosts_set,
    _extract_emails_from_html_page,
    _filter_candidate_urls,
    _host_matches_allowed,
    _hostname_from_url,
    _is_email,
    _link_priority_for_email_hunt,
    _normalize_official_urls_to_site_roots,
    _parse_page_links,
    _strip_trailing_junk,
    _url_visit_key,
)


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
            leaf_penalty = 1 if segs and len(segs[-1]) <= 4 and re.fullmatch(r"[A-Za-z0-9]{1,4}", segs[-1]) else 0
        except Exception:
            depth, leaf_penalty = 99, 0
        return (-_link_priority_for_email_hunt(u), leaf_penalty, depth)

    all_candidates.sort(key=_sitemap_sort_key)
    top = all_candidates[:SITEMAP_TOPK]
    stats["urls_added"] = len(top)
    return top, stats


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
        from playwright.async_api import TimeoutError as PWTimeout
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise ImportError(
            "playwright 未安装。请运行：\n  pip install playwright\n  playwright install chromium"
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

    merged_src = [{"email": em, "source_url": email_src_map.get(em.lower(), ("", ""))[1]} for em in merged_order]
    return merged_order, merged_src


def _fetch_merge_page_emails(
    plan: Dict[str, Any],
    retrieval: Dict[str, Any],
    allowed: List[str],
    llm_candidates: List[str],
) -> Tuple[List[str], List[Dict[str, str]]]:
    return asyncio.run(_run_fetch_merge_async(plan, retrieval, allowed, llm_candidates))
