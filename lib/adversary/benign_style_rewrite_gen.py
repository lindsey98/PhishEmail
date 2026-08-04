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

The prompt below is adapted from the SpearMail "Email Generation prompt"; the
deception-inducing instructions (typosquatted sender email, fake link to an
impersonated organization) are replaced with consistency-preserving ones.

Usage:
    python -m lib.adversary.benign_style_rewrite_gen \
        --input_csv ./datasets/CSDMC2010_benign_results_augmented.csv \
        --output_dir ./datasets/spearmail_benign_rewrite \
        --num 500
"""

import os
import re
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import pandas as pd
from tqdm import tqdm
from tldextract import tldextract
from openai import OpenAI

try:  # optional language filter, mirrors clean_benchmark.py
    from langdetect import detect_langs
    import langdetect as _ld
except Exception:  # pragma: no cover
    detect_langs = None

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


def build_prompt(sender_name: str, sender_address: str, subject: str,
                 body: str, n_paragraphs: int) -> str:
    domain = _domain_of(sender_address)
    is_webmail = domain.lower() in WEBMAIL_DOMAINS

    if is_webmail:
        link_rule = (
            f"The sender uses a personal webmail address (@{domain}), so the "
            f"email must NOT claim to represent any company, university, or "
            f"agency. Include exactly one plain, topic-relevant hyperlink before "
            f"the signature (e.g. a public event page or shared document). Do "
            f"NOT link to, or claim the identity of, any well-known organization "
            f"that the sender does not actually belong to."
        )
    else:
        link_rule = (
            f"Any organizational identity claimed in the email, and the one "
            f"call-to-action hyperlink you include before the signature, MUST "
            f"stay consistent with the sender's own domain '{domain}' "
            f"(e.g. https://{domain}/...). Do NOT use a URL shortener, a "
            f"typosquatted look-alike domain, or any domain other than "
            f"'{domain}'. Never impersonate a different organization."
        )

    return f"""I want to rewrite a legitimate (benign) email so it reads like a polished, persuasive professional outreach message, while keeping it completely benign and consistent.

The email is sent by: {sender_name} <{sender_address}>.
Preserve this exact sender identity. Do not change who the sender is or the domain they send from.

Rewrite the message below into a detailed HTML email of {n_paragraphs} paragraphs or longer. Adopt these stylistic patterns:
- an engaging, personalized, professional tone directed at the recipient;
- a clear (but honest) call-to-action;
- one inline hyperlink placed in the body before the signature;
- a nicely formatted, left-aligned email signature that includes the sender's email address ({sender_address}) and a plausible phone number.

Consistency / benignity requirements (critical):
- {link_rule}
- Keep the factual topic and intent of the original message. Do not fabricate deadlines, prizes, security alerts, credential/payment requests, or any deceptive claim.
- The <title> of the HTML must be the subject line of the email.

Original benign email to rewrite:
Subject: {subject}
Body: {body}

Return ONLY JSON in exactly this format:
{{"subject": "<email subject>", "body": "<html>...email body as a single line...</html>"}}
The body must be valid single-line HTML. Do not include markdown, notes, or disclaimers."""


def _safe(text: str, maxlen: int = 40) -> str:
    text = re.sub(r'[^A-Za-z0-9]+', '_', str(text)).strip('_')
    return text[:maxlen] or 'x'


def valid_row(name, address, subject, body) -> bool:
    if not isinstance(body, str) or not isinstance(address, str):
        return False
    if '@' not in address:
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


def generate_one(client, model, row):
    name = str(row['sender_name']) if pd.notna(row['sender_name']) else 'Sender'
    address = str(row['sender_address'])
    subject = str(row['subject'])
    body = str(row['email_body_text'])
    to_name = str(row['to_names']) if pd.notna(row.get('to_names')) else ''
    to_addr = str(row['to_addresses']) if pd.notna(row.get('to_addresses')) else 'recipient@example.com'
    # to_addresses may be a list-like string; take the first address
    m = re.search(r'[\w.+-]+@[\w.-]+', to_addr)
    to_addr = m.group(0) if m else 'recipient@example.com'

    n_paragraphs = random.randint(2, 4)
    prompt = build_prompt(name, address, subject, body, n_paragraphs)

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


@click.command()
@click.option('--input_csv', default='./datasets/CSDMC2010_benign_results_augmented.csv',
              show_default=True, help='Parsed CSDMC-2010 Ham benign emails.')
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
def main(input_csv, output_dir, num, model, seed, workers, proxy, key_file):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    if not os.environ.get('OPENAI_API_KEY') and os.path.exists(key_file):
        os.environ['OPENAI_API_KEY'] = open(key_file).read().strip()

    client_kwargs = {}
    if proxy:
        import httpx
        client_kwargs['http_client'] = httpx.Client(proxy=proxy)
    client = OpenAI(**client_kwargs)

    df = pd.read_csv(input_csv)
    candidates = [i for i in df.index
                  if valid_row(df.loc[i, 'sender_name'], df.loc[i, 'sender_address'],
                               df.loc[i, 'subject'], df.loc[i, 'email_body_text'])]
    random.shuffle(candidates)
    print(f"{len(candidates)} valid candidate benign emails; sampling up to {num}.")

    written = 0
    selected = []
    for i in candidates:
        if written + len(selected) >= num:
            break
        out_path = os.path.join(output_dir, f"benign_{i}_{_safe(df.loc[i, 'sender_name'])}.eml")
        if os.path.exists(out_path):  # resume: already generated
            written += 1
            continue
        selected.append((i, out_path))

    print(f"{written} already present; generating {min(len(selected), num - written)} new emails.")
    selected = selected[:max(0, num - written)]

    def work(item):
        i, out_path = item
        try:
            eml = generate_one(client, model, df.loc[i])
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(eml)
            return True
        except Exception as e:  # keep going on individual failures
            print(f"[skip idx={i}] {type(e).__name__}: {e}")
            return False

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, item) for item in selected]
        for fut in tqdm(as_completed(futures), total=len(futures), desc='generating'):
            if fut.result():
                written += 1

    print(f"Done. {written} benign SpearMail-style emails in {output_dir}")


if __name__ == '__main__':
    main()
