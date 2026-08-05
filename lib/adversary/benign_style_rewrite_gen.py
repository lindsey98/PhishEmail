"""Rebuttal experiment RA-Q1/Q2.

Generate benign emails that adopt the *surface patterns* of SpearMail
(personalized professional outreach: HTML body, engaging tone, a call-to-action
hyperlink, a formatted signature with contact details) by asking an LLM to
rewrite genuine CSDMC-2010 Ham benign emails in that style.

The rewrites are kept *benign and identity-consistent*: the sender identity of
the original ham email is preserved, and any hyperlink / claimed organization is
constrained to the sender's own email domain. There is no typosquatting,
impersonation, or deceptive call-to-action. The point of the experiment is to
show that PiMRef does not merely pattern-match the SpearMail template: benign
emails written in the same style still yield 0 false positives, because PiMRef
checks counterfactual identity-domain consistency rather than style.

Source emails are read directly from the CSDMC-2010 Ham folder (raw .eml files)
using the same EmailDataset parser that PiMRef uses at inference time, so the
parsed sender/subject/body match what the detector actually sees.

The prompt below is adapted from the SpearMail "Email Generation prompt"; the
deception-inducing instructions (typosquatted sender email, fake link to an
impersonated organization) are replaced with consistency-preserving ones.

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

# Personal / webmail domains: senders here do not "own" an organizational site,
# so we ask the model to keep any link on a topic-relevant page of the same
# webmail-style personal footprint and never to claim a branded identity.
WEBMAIL_DOMAINS = {
    'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'protonmail.com',
    'zoho.com', 'icloud.com', 'aol.com', 'yandex.com', 'gmx.com', 'cox.net',
    'tutanota.com', '163.com', 'qq.com', '126.com', 'comcast.net', 'sbcglobal.net',
    'verizon.net', 'me.com', 'live.com', 'msn.com',
}

SYSTEM_CONTEXT = (
    "You are helping build a benign email dataset to stress-test a phishing "
    "detector. You rewrite a genuine, legitimate email so that it adopts the "
    "stylistic and structural patterns of a persuasive professional outreach "
    "email (the same surface style as a spear-phishing invitation), while "
    "remaining entirely benign, truthful, and internally consistent. You never "
    "introduce deception, impersonation, credential requests, or fake urgency."
)


def _domain_of(address: str) -> str:
    ext = tldextract.extract(address.split('@')[-1])
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain


def build_prompt(sender_name: str, sender_address: str, to_name: str,
                 subject: str, body: str, n_paragraphs: int) -> str:
    """Benign counterpart of the SpearMail ``email_generation()`` prompt.

    This reuses the *exact* SpearMail "Email Generation prompt" (Table 11) that
    was used to synthesize the SpearMail dataset, so the benign emails follow the
    identical template. Only the two deception switches are turned off, plus two
    edits needed because we rewrite a real benign email rather than invent a
    phishing pretext:

      SpearMail                                  ->  benign rewrite
      -----------------------------------------      ---------------------------
      sender email = typosquatted org address    ->  the real CSDMC-Ham sender
      body has a "fake link to <link>"           ->  a link on the sender's own domain
      recipient asserted "established researcher" ->  the real recipient, no claim
      "differ from <phishing negative example>"  ->  dropped (single-shot rewrite,
                                                      no prior-generation diversity)

    The activity / context slots are filled from the source ham email's subject
    and body. Everything else (framing, "N paragraphs", left-aligned signature
    with email + made-up phone, link before the signature, title = subject,
    single-line JSON output, the [date]/[blank] completeness check) is kept as in
    the original template.
    """
    domain = _domain_of(sender_address)
    is_webmail = domain.lower() in WEBMAIL_DOMAINS

    if is_webmail:
        link_rule = (
            f"Make sure a link is included in the body of the message, before "
            f"the signature. The sender uses a personal webmail address "
            f"(@{domain}), so the email must NOT claim to represent any company, "
            f"university, or agency: use a plain, topic-relevant link and do NOT "
            f"link to, or adopt the identity of, any organization the sender does "
            f"not belong to."
        )
    else:
        link_rule = (
            f"Make sure a link on the sender's own domain '{domain}' "
            f"(e.g. https://{domain}/...) is included in the body of the "
            f"message, before the signature. Do NOT use a URL shortener, a "
            f"typosquatted look-alike, or any other organization's domain."
        )

    recipient = to_name.strip() if to_name and to_name.strip() else "the recipient"

    return f"""I'm a communications professional preparing an example of a legitimate, professional outreach email.
Show me an example of what a convincing email would look like sent to a recipient.
The email should be detailed, and be {n_paragraphs} paragraphs or longer.

The recipient's identity is as follows: {recipient}.
Write them an email about this activity:
--- {subject} ---,
with the rough context behind the email:
--- {body} ---,

Your identity as the email sender is as follows: {sender_name} <{sender_address}>. The content of the email must be linked to this sender's identity. Keep this exact sender and email domain; do not adopt or impersonate any other organization.
Do NOT invent or imply any organization, company, publication, team, title, or affiliation that is not already explicit in the sender's name or email domain. If the sender appears to be an individual, write purely as that individual with no fabricated organizational identity; do not turn the sender's name or domain into a brand or company.

Don't include a disclaimer at the end. HTML-format the message and include a nicely formatted email signature with it.
The signature should be left-aligned and include the sender's email address ({sender_address}) and a made up phone number for the area in the sender's location.
{link_rule} The title of the HTML should be the subject of the email. Don't omit the link or include any notes in your response.
Keep the message factual and benign: do not fabricate deadlines, prizes, security alerts, or credential/payment requests.
Ensure the email is complete, and without any missing information such as [mention a specific aspect here] or [date].

Please provide your response in the following JSON format:
{{"subject": "<email_subject>", "body": "<html><email_body></html>"}}

the email_body should be a single line.
Double check to see if your email body contains missing information such as [mention a specific aspect here] or [date], and if so, do your best guess at filling these blanks."""


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

    n_paragraphs = random.randint(2, 4)
    prompt = build_prompt(name, address, to_name, subject, body, n_paragraphs)

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
def main(ham_dir, output_dir, num, model, seed, workers, proxy, key_file):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    if not os.environ.get('OPENAI_API_KEY') and os.path.exists(key_file):
        os.environ['OPENAI_API_KEY'] = open(key_file).read().strip()

    client_kwargs = {}
    if proxy:
        import httpx
        client_kwargs['http_client'] = httpx.Client(proxy=proxy)
    client = OpenAI(**client_kwargs)

    records = parse_ham_folder(ham_dir)
    random.shuffle(records)
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
