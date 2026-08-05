"""Rebuttal experiment RA-Q1/Q2.

Generate benign emails that adopt SpearMail's *surface presentation* (clean HTML,
a courteous greeting, a simple professional signature) by asking an LLM to rewrite
genuine CSDMC-2010 Ham benign emails in that style — while staying strictly
faithful to each original's meaning.

Crucially, the rewrite does NOT fabricate intent. SpearMail's template always
injects a call-to-action pseudo-link; a faithful benign rewrite must not. So:

  - the real sender identity (name + address) is preserved, no impersonation;
  - if the original does not ask the recipient to do anything, the rewrite adds
    no call-to-action, invitation, or link;
  - links are preserved only if present in the original, using the original URLs
    verbatim (never invented, never relocated to the sender's domain);
  - no invented organizations, titles, phone numbers, deadlines, or padding.

The point of the experiment is to show PiMRef does not merely pattern-match the
SpearMail template: benign emails restyled to look like SpearMail still yield 0
false positives, because PiMRef checks counterfactual identity-domain
consistency rather than surface style.

Source emails are read directly from the CSDMC-2010 Ham folder (raw .eml files);
anonymized/aggregator/masked senders are filtered out (see is_real_sender).

Usage:
    python -m lib.adversary.benign_style_rewrite_gen \
        --ham_dir ./datasets/CSDMC2010/Ham \
        --output_dir ./datasets/spearmail_benign_rewrite \
        --num 500
"""

import os
import re
import json
import email
import email.policy
import random
from email.utils import parseaddr, getaddresses
from email.header import decode_header
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
from tqdm import tqdm
from tldextract import tldextract
from bs4 import BeautifulSoup
from openai import OpenAI

try:  # optional language filter, mirrors clean_benchmark.py
    from langdetect import detect_langs
except Exception:  # pragma: no cover
    detect_langs = None

# Anonymized / placeholder domains used by the SpamAssassin public corpus.
# Emails whose From is one of these carry a third-party publication name in the
# display name (e.g. "guardian" <rssfeeds@spamassassin.taint.org>) but an
# aggregator domain, so a SpearMail-style rewrite would fabricate a brand
# identity that the domain does not back. Such sources are excluded.
ANON_DOMAINS = {'taint.org', 'example.com', 'example.org', 'example.net',
                'example.edu', 'localhost', 'sneakemail.com'}
# Role local-parts that indicate an RSS/feed aggregator rather than a real sender.
ROLE_LOCALPARTS = ('rssfeeds@', 'rss@', 'feed@', 'feeds@', 'noreply@', 'no-reply@')

SYSTEM_CONTEXT = (
    "You are helping build a benign email dataset to stress-test a phishing "
    "detector. You take a genuine, legitimate email and rewrite it into a more "
    "polished, professionally formatted (SpearMail-style) version WITHOUT "
    "changing what it actually says, asks, or offers. You are faithful to the "
    "original's intent and facts: you never add a call-to-action, link, "
    "invitation, organization, title, or urgency that the original did not have."
)


def build_prompt(sender_name: str, sender_address: str, to_name: str,
                 subject: str, body: str, orig_urls) -> str:
    """Faithful, style-only benign rewrite prompt.

    Adopts SpearMail's *surface presentation* (clean HTML, a courteous greeting,
    a simple professional signature) but preserves the original email's
    communicative intent exactly. In particular it does NOT fabricate a
    call-to-action or link: SpearMail's template always injects a pseudo-link,
    which is precisely the thing we must not add to a benign email that never
    asked the recipient to do anything.

    Link handling: the caller passes the URLs found in the *original* email. If
    there are none, the rewrite adds none. If there are some, only those exact
    URLs may appear (never invented, never relocated to the sender's domain).
    """
    recipient = to_name.strip() if to_name and to_name.strip() else "the recipient"

    if orig_urls:
        links_rule = (
            "The original email contains the following link(s). If (and only if) "
            "the original meaning calls for keeping a link, use ONLY these exact "
            "URLs, unchanged — never invent a new link, never edit these, and "
            "never replace them with the sender's own domain:\n"
            + "\n".join(f"  - {u}" for u in orig_urls)
        )
    else:
        links_rule = (
            "The original email contains NO links and does not ask the recipient "
            "to do anything. Do NOT add any link, button, or call-to-action of "
            "any kind."
        )

    return f"""I want to rewrite a legitimate (benign) email into a more polished, professionally formatted version, WITHOUT changing what it actually says or asks.

The email is sent by: {sender_name} <{sender_address}>. Keep this exact sender identity and email domain.
The recipient is: {recipient}.

Rewrite the message below as clean HTML with a courteous greeting and a simple professional signature (the sender's real name and email address only). Improve the wording and formatting; do not change the substance.

Faithfulness rules (critical):
- Preserve the original's communicative intent exactly. If the original does NOT invite or ask the recipient to do anything, the rewrite must NOT invite or ask them to do anything either — no call-to-action, no "click here", no invitation, no request.
- Do not invent any content, facts, events, deadlines, phone numbers, organizations, publications, teams, titles, or affiliations that are not in the original. If the sender looks like an individual, write purely as that individual.
- {links_rule}
- Keep roughly the same length and scope as the original; do not pad it with new material.
- The <title> of the HTML must be the subject line of the email.

Original email:
Subject: {subject}
Body: {body}

Return ONLY JSON in exactly this format:
{{"subject": "<email subject>", "body": "<html>...email body as a single line...</html>"}}
The body must be valid single-line HTML. Do not include markdown, notes, or disclaimers."""


def _safe(text: str, maxlen: int = 60) -> str:
    text = re.sub(r'[^A-Za-z0-9]+', '_', str(text)).strip('_')
    return text[:maxlen] or 'x'


def is_real_sender(address) -> bool:
    """Reject anonymized / aggregator / malformed senders.

    Keeps only addresses that plausibly belong to a real individual or
    organization, so the rewrite preserves the true sender identity instead of
    fabricating a brand for a placeholder/mailing-list domain.
    """
    if not isinstance(address, str) or '@' not in address:
        return False
    if address.lower().startswith(ROLE_LOCALPARTS):
        return False
    dom = address.split('@')[-1].lower()
    if dom.startswith(('lists.', 'list.', 'listserv')) or 'mailman' in dom:
        return False
    ext = tldextract.extract(dom)
    if not ext.suffix:  # malformed / no registrable domain
        return False
    if f"{ext.domain}.{ext.suffix}" in ANON_DOMAINS:
        return False
    return True


def valid_record(name, address, subject, body) -> bool:
    if not isinstance(body, str) or not isinstance(address, str):
        return False
    if not is_real_sender(address):
        return False
    body = body.strip()
    if not (80 <= len(body) <= 6000):
        return False
    if not isinstance(subject, str) or not subject.strip():
        return False
    if detect_langs is not None:
        try:
            langs = detect_langs(body)
            if not any(l.lang == 'en' for l in langs):
                return False
        except Exception:
            return False
    return True


def generate_one(client, model, rec):
    name = rec['name']
    address = rec['address']
    subject = rec['subject']
    body = rec['body']
    to_name = rec['to_name']
    to_addr = rec['to_addr']

    prompt = build_prompt(name, address, to_name, subject, body, rec.get('orig_urls', []))

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_CONTEXT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
    )
    data = json.loads(resp.choices[0].message.content)
    out_subject = str(data['subject']).replace('\n', ' ').strip()
    out_body = str(data['body']).strip()

    from_hdr = f"{name} <{address}>"
    to_hdr = f"{to_name} <{to_addr}>" if to_name else to_addr
    eml = (
        f"From: {from_hdr}\n"
        f"To: {to_hdr}\n"
        f"Subject: {out_subject}\n"
        f"Content-Type: text/html\n\n"
        f"{out_body}\n"
    )
    return eml


def _decode_hdr(value) -> str:
    if not value:
        return ''
    try:
        parts = decode_header(value)
        return ''.join(p.decode(cs or 'utf-8', 'replace') if isinstance(p, bytes) else p
                       for p, cs in parts).strip()
    except Exception:
        return str(value).strip()


# Common quoted-reply markers; drop everything from the first one onward so the
# rewrite is seeded with the new message, not the quoted history.
_REPLY_MARKERS = [
    re.compile(r'^.{0,80}\bwrote:\s*$', re.MULTILINE),  # "On/At ... wrote:"
    re.compile(r'^-{2,}\s*Original Message\s*-{2,}', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^_{5,}\s*$', re.MULTILINE),
    re.compile(r'^From:\s.*$', re.MULTILINE),
]


_URL_RE = re.compile(r'https?://[^\s"\'<>)\]]+', re.I)


def _extract_links(msg, cap: int = 12):
    """URLs present in the *original* email (inline text + <a href>), so the
    rewrite can preserve real links verbatim and never invent one."""
    urls = []
    for part in msg.walk():
        ctype = part.get_content_type()
        if ctype not in ('text/plain', 'text/html'):
            continue
        try:
            payload = part.get_content()
        except Exception:
            continue
        if not isinstance(payload, str):
            continue
        if ctype == 'text/html':
            urls += re.findall(r'href=["\']([^"\']+)["\']', payload, re.I)
        urls += _URL_RE.findall(payload)
    seen = []
    for u in urls:
        u = u.strip().rstrip('.,);]>')
        if not u.lower().startswith(('http://', 'https://')):
            continue
        if u not in seen:
            seen.append(u)
    return seen[:cap]


def _extract_body(msg) -> str:
    body = msg.get_body(preferencelist=('plain', 'html'))
    if body is None:
        return ''
    try:
        content = body.get_content()
    except Exception:
        return ''
    if body.get_content_type() == 'text/html':
        content = BeautifulSoup(content, 'html.parser').get_text(separator='\n')
    if not isinstance(content, str):
        return ''
    # strip quoted reply history
    cut = len(content)
    for pat in _REPLY_MARKERS:
        m = pat.search(content)
        if m:
            cut = min(cut, m.start())
    content = content[:cut]
    # drop lingering quote lines and collapse blank runs
    lines = [ln for ln in content.splitlines() if not ln.lstrip().startswith('>')]
    content = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines))
    return content.strip()


def parse_ham_folder(ham_dir):
    """Parse raw CSDMC-2010 Ham .eml files into benign source records.

    Reads the raw .eml files with the stdlib email parser (+ BeautifulSoup for
    HTML bodies), mirroring the lightweight approach in
    spearmail_structural_perturb.py so the whole experiment runs without the
    heavy inference dependency chain.
    """
    paths = sorted(Path(ham_dir).rglob('*.eml'))
    records = []
    for path in tqdm(paths, desc='parsing ham'):
        try:
            with open(path, 'rb') as f:
                msg = email.message_from_binary_file(f, policy=email.policy.default)
            sender_name, sender_address = parseaddr(msg.get('From', ''))
            sender_name = _decode_hdr(sender_name)
            # A display name containing '@' signals a masked/forwarded header
            # (e.g. "brand@realco.com [mothlight/..]" <ticket@sneakemail.com>),
            # where the name carries a third-party brand the domain does not back.
            if '@' in sender_name:
                continue
            sender_name = sender_name or (sender_address.split('@')[0] if sender_address else 'Sender')
            subject = _decode_hdr(msg.get('Subject', ''))
            body = _extract_body(msg)
            recips = getaddresses(msg.get_all('To', []))
            to_name, to_addr = (recips[0] if recips else ('', ''))
            to_name = _decode_hdr(to_name)
            orig_urls = _extract_links(msg)
        except Exception:
            continue
        if not valid_record(sender_name, sender_address, subject, body):
            continue
        records.append({
            'key': path.stem,
            'name': sender_name,
            'address': sender_address,
            'subject': subject,
            'body': body,
            'to_name': to_name,
            'to_addr': to_addr or 'recipient@example.com',
            'orig_urls': orig_urls,
        })
    return records


@click.command()
@click.option('--ham_dir', default='./datasets/CSDMC2010/Ham', show_default=True,
              help='Folder of raw CSDMC-2010 Ham benign .eml files.')
@click.option('--output_dir', default='./datasets/spearmail_benign_rewrite',
              show_default=True, help='Where to write the rewritten .eml files.')
@click.option('--num', default=500, show_default=True, type=int,
              help='Number of benign emails to generate.')
@click.option('--model', default='gpt-4o', show_default=True,
              help='OpenAI chat model to use for rewriting.')
@click.option('--seed', default=42, show_default=True, type=int)
@click.option('--workers', default=8, show_default=True, type=int,
              help='Concurrent API calls.')
@click.option('--proxy', default=None, help='Optional http(s) proxy, e.g. http://127.0.0.1:7890')
@click.option('--key_file', default='./datasets/openai_key.txt', show_default=True)
@click.option('--exclude_keys', default='', help='Comma-separated source stems (e.g. '
              'TRAIN_03706) to permanently exclude; top-up then draws fresh emails '
              'from beyond the original sample instead of regenerating these.')
def main(ham_dir, output_dir, num, model, seed, workers, proxy, key_file, exclude_keys):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    exclude = {k.strip().replace('benign_', '').replace('.eml', '')
               for k in exclude_keys.split(',') if k.strip()}

    if not os.environ.get('OPENAI_API_KEY') and os.path.exists(key_file):
        os.environ['OPENAI_API_KEY'] = open(key_file).read().strip()

    client_kwargs = {}
    if proxy:
        import httpx
        client_kwargs['http_client'] = httpx.Client(proxy=proxy)
    client = OpenAI(**client_kwargs)

    records = parse_ham_folder(ham_dir)
    random.shuffle(records)
    if exclude:
        # Drop excluded keys AFTER the shuffle so the remaining sample order (and
        # thus the already-generated files) is preserved; top-up pulls the next
        # non-excluded, not-yet-generated records.
        records = [r for r in records if r['key'] not in exclude]
        print(f"excluded {len(exclude)} keys: {sorted(exclude)}")
    print(f"{len(records)} valid candidate benign emails; sampling up to {num}.")

    # Two-pass selection so top-up is deterministic and never overshoots:
    # keep every already-generated file first, then fill the remainder with
    # fresh (not-yet-generated) records up to `num`.
    for rec in records:
        rec['out_path'] = os.path.join(output_dir, f"benign_{_safe(rec['key'])}.eml")
    written = sum(1 for rec in records if os.path.exists(rec['out_path']))
    selected = []
    for rec in records:
        if written + len(selected) >= num:
            break
        if os.path.exists(rec['out_path']):
            continue
        selected.append((rec, rec['out_path']))

    print(f"{written} already present; generating {len(selected)} new emails.")

    def work(item):
        rec, out_path = item
        try:
            eml = generate_one(client, model, rec)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(eml)
            return True
        except Exception as e:  # keep going on individual failures
            print(f"[skip {rec['key']}] {type(e).__name__}: {e}")
            return False

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, item) for item in selected]
        for fut in tqdm(as_completed(futures), total=len(futures), desc='generating'):
            if fut.result():
                written += 1

    print(f"Done. {written} benign SpearMail-style emails in {output_dir}")


if __name__ == '__main__':
    main()
