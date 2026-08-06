"""LLM-based selection of meta-review Requirement #2 emails.

Requirement #2 wants *normal legitimate* emails where the message says it is from
one person/org but arrives through some other service or domain. We select them
from the real field-study inboxes with an LLM judging each email's CONTENT
against the sender address. Independent of PiMRef's verdict.

An email is selected if EITHER:
  (A) it is sent from a personal webmail account AND its content presents itself
      as an organization or an internal role; OR
  (B) its content presents itself as an organization but the sender address is
      NOT that organization's official domain.

Usage:
    python -m lib.pipeline.req2_llm_filter \
        --email-dir datasets/field \
        --eval-dir datasets/field_req2_eval \
        --out datasets/req2_llm_classified.csv \
        --model gpt-4o-mini
"""

import os
import re
import csv as _csv
import json
import email
import email.policy
import shutil
from email.utils import parseaddr
from email.header import decode_header
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
from tqdm import tqdm
from tldextract import tldextract
from bs4 import BeautifulSoup
from openai import OpenAI

_csv.field_size_limit(10_000_000)

WEBMAIL_DOMAINS = {
    'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com', 'msn.com',
    'live.com', 'me.com', 'icloud.com', 'protonmail.com', 'gmx.com', 'foxmail.com',
    '163.com', '126.com', 'qq.com', 'sina.com', '139.com', 'aliyun.com', 'yeah.net',
    'sbcglobal.net', 'earthlink.net', 'comcast.net',
}

SYSTEM = (
    "You read an email's From address, subject and body. Identify the "
    "organization the email presents itself as being from (via the display name, "
    "the signature, or wording like 'from the X team' — not an organization only "
    "mentioned in passing), and decide whether the From address belongs to that "
    "organization's own domain."
)

_REPLY = re.compile(r'^.{0,80}\bwrote:\s*$|^-{2,}\s*Original Message|^_{5,}\s*$',
                    re.IGNORECASE | re.MULTILINE)


def _reg(addr):
    d = str(addr).split('@')[-1].strip().lower()
    e = tldextract.extract(d)
    return f"{e.domain}.{e.suffix}" if e.suffix else d


def _dec(v):
    if not v:
        return ''
    try:
        return ''.join(p.decode(c or 'utf-8', 'replace') if isinstance(p, bytes) else p
                       for p, c in decode_header(v)).strip()
    except Exception:
        return str(v).strip()


def _body(msg):
    b = msg.get_body(preferencelist=('plain', 'html'))
    if b is None:
        return ''
    try:
        content = b.get_content()
    except Exception:
        return ''
    if b.get_content_type() == 'text/html':
        content = BeautifulSoup(content, 'html.parser').get_text(separator='\n')
    if not isinstance(content, str):
        return ''
    m = _REPLY.search(content)
    if m:
        content = content[:m.start()]
    content = '\n'.join(ln for ln in content.splitlines() if not ln.lstrip().startswith('>'))
    return re.sub(r'\n{3,}', '\n\n', content).strip()


def build_prompt(name, addr, domain, subject, body):
    return f"""From display name: {name!r}
From address: {addr}   (sender domain: {domain})
Subject: {subject}
Body (may be truncated):
\"\"\"
{body[:2500]}
\"\"\"

Using ONLY the above:
- claimed_organization: the organization the email presents itself as being from (display name, signature, or "the X team"). "" if it presents no organization (e.g. just a personal message).
- from_address_is_official: does the From address belong to that organization's own/official domain? true / false / "n/a" (n/a when there is no claimed_organization).
- label: 1 if the body presents an organization AND the From address is NOT that organization's own domain; otherwise 0.

Return ONLY JSON:
{{"claimed_organization": "...", "from_address_is_official": true|false|"n/a", "label": 0|1, "reason": "one short sentence"}}"""


def classify(client, model, rec):
    prompt = build_prompt(rec['name'], rec['addr'], rec['domain'], rec['subject'], rec['body'])
    resp = client.chat.completions.create(
        model=model, response_format={"type": "json_object"}, temperature=0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
    )
    d = json.loads(resp.choices[0].message.content)
    label = 1 if str(d.get('label', 0)).strip().lower() in ('1', 'true', 'yes') else 0
    return {
        'file': rec['file'], 'from_name': rec['name'], 'from_addr': rec['addr'],
        'sender_domain': rec['domain'],
        'is_personal_webmail': rec['domain'] in WEBMAIL_DOMAINS,
        'subject': rec['subject'][:120],
        'claimed_organization': str(d.get('claimed_organization', ''))[:80],
        'from_address_is_official': d.get('from_address_is_official', ''),
        'label': label,
        'reason': str(d.get('reason', ''))[:200],
    }


def parse_dir(email_dir, exclude):
    exc = re.compile(exclude, re.I) if exclude else None
    recs = []
    for root, _, files in os.walk(email_dir):
        if 'inbox' not in root.lower() or (exc and exc.search(root)):
            continue
        for fn in files:
            if not fn.lower().endswith('.eml'):
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, 'rb') as fh:
                    msg = email.message_from_binary_file(fh, policy=email.policy.default)
                name, addr = parseaddr(msg.get('From', ''))
                if '@' not in addr:
                    continue
                recs.append({'file': fp, 'name': _dec(name), 'addr': addr,
                             'domain': _reg(addr), 'subject': _dec(msg.get('Subject', '')),
                             'body': _body(msg)})
            except Exception:
                continue
    return recs


FIELDS = ['file', 'from_name', 'from_addr', 'sender_domain', 'is_personal_webmail',
          'subject', 'claimed_organization', 'from_address_is_official', 'label', 'reason']


@click.command()
@click.option('--email-dir', default='datasets/field', show_default=True)
@click.option('--eval-dir', default='datasets/field_req2_eval', show_default=True,
              help='label=1 emails (org claimed, From domain differs) copied here (+ _manifest.csv).')
@click.option('--out', default='datasets/req2_llm_classified.csv', show_default=True,
              help='All classifications (also used to resume).')
@click.option('--exclude', default=r'phish|junk|spam|honeypot|deleted|/sent', show_default=True)
@click.option('--model', default='gpt-4o-mini', show_default=True)
@click.option('--workers', default=8, show_default=True, type=int)
@click.option('--limit', default=0, type=int, help='Classify at most N (0=all).')
@click.option('--key-file', default='./datasets/openai_key.txt', show_default=True)
def main(email_dir, eval_dir, out, exclude, model, workers, limit, key_file):
    if not os.environ.get('OPENAI_API_KEY') and os.path.exists(key_file):
        os.environ['OPENAI_API_KEY'] = open(key_file).read().strip()
    client = OpenAI()

    recs = parse_dir(email_dir, exclude)
    print(f"{len(recs)} emails parsed from {email_dir}")

    done = set()
    if os.path.exists(out):
        with open(out, newline='', encoding='utf-8') as f:
            for row in _csv.DictReader(f):
                done.add(row['file'])
        print(f"resuming: {len(done)} already classified")
    todo = [r for r in recs if r['file'] not in done]
    if limit:
        todo = todo[:limit]
    print(f"classifying {len(todo)} with {model}")

    new = not (os.path.exists(out) and os.path.getsize(out) > 0)
    fout = open(out, 'a', newline='', encoding='utf-8')
    w = _csv.DictWriter(fout, fieldnames=FIELDS)
    if new:
        w.writeheader()

    def work(r):
        try:
            return classify(client, model, r)
        except Exception as e:
            print(f"[skip] {r['file']}: {type(e).__name__}: {e}")
            return None

    n_match = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in tqdm(as_completed([ex.submit(work, r) for r in todo]), total=len(todo)):
            row = fut.result()
            if not row:
                continue
            w.writerow(row)
            fout.flush()
            if row['label'] == 1:
                n_match += 1
    fout.close()
    print(f"newly classified: {len(todo)}   label=1 among them: {n_match}")

    # rebuild eval dir from ALL positives in the classified CSV
    _export_positives(out, eval_dir)


def _export_positives(out, eval_dir):
    if not os.path.exists(out):
        return
    os.makedirs(eval_dir, exist_ok=True)
    rows = [r for r in _csv.DictReader(open(out, newline='', encoding='utf-8'))
            if str(r.get('label')).strip() in ('1', 'True', 'true')]
    man, used, n = [], set(), 0
    for r in rows:
        src = r['file']
        if not os.path.exists(src):
            continue
        vol = src.split('field/')[1].split('/')[0] if 'field/' in src else 'src'
        base = f"{vol}_{os.path.basename(src)}"
        name, i = base, 1
        while name in used:
            name = base.replace('.eml', f'_{i}.eml'); i += 1
        used.add(name)
        shutil.copy2(src, os.path.join(eval_dir, name))
        rr = dict(r); rr['eval_file'] = name; man.append(rr); n += 1
    import pandas as pd
    df = pd.DataFrame(man)
    df.to_csv(os.path.join(eval_dir, '_manifest.csv'), index=False)
    print(f"exported {n} positives -> {eval_dir}")
    if man:
        wm = df['is_personal_webmail'].astype(str).str.lower().eq('true').sum()
        print(f"  of which personal-webmail senders: {int(wm)}")


if __name__ == '__main__':
    main()
