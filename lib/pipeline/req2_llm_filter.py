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
    plain, html_parts = [], []
    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        if ctype not in ('text/plain', 'text/html'):
            continue
        raw = part.get_payload(decode=True)   # 自动处理 base64/quoted-printable
        if not raw:
            continue
        charset = part.get_content_charset() or 'utf-8'
        try:
            text = raw.decode(charset, 'replace')
        except LookupError:
            text = raw.decode('utf-8', 'replace')
        (plain if ctype == 'text/plain' else html_parts).append(text)

    if plain:
        content = '\n'.join(plain)
    elif html_parts:
        content = BeautifulSoup('\n'.join(html_parts), 'html.parser').get_text(separator='\n')
    else:
        p = msg.get_payload(decode=True)
        if p is None:
            p = msg.get_payload()
            p = p.encode('utf-8', 'replace') if isinstance(p, str) else p
        if p:
            charset = msg.get_content_charset() or 'utf-8'
            try:
                text = p.decode(charset, 'replace')
            except (LookupError, AttributeError):
                text = str(p)
            if '<' in text and '>' in text:
                text = BeautifulSoup(text, 'html.parser').get_text(separator='\n')
            plain.append(text)

    content = content.replace('\r', '')
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
        model=model, temperature=0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
    )
    content = resp.choices[0].message.content.strip()
    m = re.search(r'\{.*\}', content, re.DOTALL)
    if not m:
        raise ValueError(f'no JSON in response: {content[:100]}')
    d = json.loads(m.group(0))
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


def _parse_one(fp):
    try:
        with open(fp, 'rb') as fh:
            msg = email.message_from_binary_file(fh, policy=email.policy.default)
        name, addr = parseaddr(msg.get('From', ''))
        if '@' not in addr:
            return None
        return {'file': fp, 'name': _dec(name), 'addr': addr,
                'domain': _reg(addr), 'subject': _dec(msg.get('Subject', '')),
                'body': _body(msg)}
    except Exception:
        return None


def parse_dir(email_dir, exclude, folder_filter='inbox', workers=None):
    exc = re.compile(exclude, re.I) if exclude else None
    ff = (folder_filter or '').lower()   # empty -> scan every folder
    paths = []
    for root, dirs, files in os.walk(email_dir):
        # 剪枝：被排除的子树直接不进
        dirs[:] = [d for d in dirs if not (exc and exc.search(os.path.join(root, d)))]
        if ff and ff not in root.lower():
            continue
        paths.extend(os.path.join(root, fn) for fn in files
                     if fn.lower().endswith('.eml'))

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        recs = [r for r in tqdm(ex.map(_parse_one, paths, chunksize=32),
                                total=len(paths), desc='parsing') if r]
    return recs


def parse_hf(name, split, sender_col, subject_col, body_col, label_col, benign_label, limit):
    """Read benign rows (label == benign_label) with `sender` = 'Name <addr>'.

    `name` is a LOCAL path — a .parquet file, or a directory of .parquet files
    (each file is a split; all of them are read). No `datasets` lib / network.
    """
    import glob
    import pandas as pd
    if os.path.isdir(name):
        files = sorted(glob.glob(os.path.join(name, '*.parquet')))
    elif os.path.isfile(name):
        files = [name]
    else:
        raise click.ClickException(f"not a local parquet file/dir: {name}")

    recs = []
    for fp in files:
        cols = [c for c in {sender_col, subject_col, body_col, label_col} if c]
        df = pd.read_parquet(fp, columns=cols)
        if benign_label != '' and label_col in df.columns:
            df = df[df[label_col].astype(str).str.strip() == str(benign_label)]
        tag = os.path.splitext(os.path.basename(fp))[0]

        def _s(v):
            return '' if pd.isna(v) else str(v)

        for idx, r in df.iterrows():
            nm, addr = parseaddr(_s(r.get(sender_col)))
            if '@' not in addr:
                continue
            recs.append({'file': f"{tag}:{idx}", 'name': nm or '',
                         'addr': addr, 'domain': _reg(addr),
                         'subject': _s(r.get(subject_col)),
                         'body': _s(r.get(body_col))})
            if limit and len(recs) >= limit:
                return recs
    return recs


FIELDS = ['file', 'from_name', 'from_addr', 'sender_domain', 'is_personal_webmail',
          'subject', 'claimed_organization', 'from_address_is_official', 'label', 'reason']


@click.command()
@click.option('--email-dir', default='datasets/field', show_default=True)
@click.option('--hf-dataset', default=None, help='Local .parquet file OR directory of parquet splits to classify (all *.parquet in the dir are read) instead of a local --email-dir.')
@click.option('--hf-split', default='all', show_default=True, help='Unused for a local parquet directory (every *.parquet is read).')
@click.option('--hf-sender-col', default='sender', show_default=True)
@click.option('--hf-subject-col', default='subject', show_default=True)
@click.option('--hf-body-col', default='text', show_default=True)
@click.option('--hf-label-col', default='label', show_default=True)
@click.option('--hf-benign-label', default='0', show_default=True,
              help="Dataset label value that means benign/ham (kept). '' keeps all rows.")
@click.option('--eval-dir', default='datasets/field_req2_eval', show_default=True,
              help='label=1 emails (org claimed, From domain differs) copied here (+ _manifest.csv).')
@click.option('--out', default='datasets/req2_llm_classified.csv', show_default=True,
              help='All classifications (also used to resume).')
@click.option('--exclude', default=r'phish|junk|spam|honeypot|deleted|/sent', show_default=True)
@click.option('--folder-filter', default='inbox', show_default=True,
              help="Only scan folders whose path contains this substring. Empty "
                   "string ('') scans every folder — use it for CSDMC-Ham etc.")
@click.option('--workers', default=8, show_default=True, type=int)
@click.option('--limit', default=0, type=int, help='Classify at most N (0=all).')
@click.option('--model', default='glm-5.2', show_default=True)
@click.option('--base-url', default='https://api.modelarts-maas.com/openai/v1', show_default=True)
@click.option('--key-file', default='./datasets/maas_key.txt', show_default=True)
def main(email_dir, hf_dataset, hf_split, hf_sender_col, hf_subject_col, hf_body_col,
         hf_label_col, hf_benign_label, eval_dir, out, exclude, folder_filter,
         model, workers, limit, key_file, base_url):
    api_key = os.environ.get('MAAS_API_KEY') or (
        open(key_file).read().strip() if os.path.exists(key_file) else None)
    if not api_key:
        raise click.ClickException('no API key: set MAAS_API_KEY or provide --key-file')
    client = OpenAI(api_key=api_key, base_url=base_url)

    if hf_dataset:
        recs = parse_hf(hf_dataset, hf_split, hf_sender_col, hf_subject_col,
                        hf_body_col, hf_label_col, hf_benign_label, limit)
        print(f"{len(recs)} benign rows from HF {hf_dataset} [{hf_split}]")
    else:
        recs = parse_dir(email_dir, exclude, folder_filter)
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

    if hf_dataset:
        # HF rows have no source .eml to copy; positives are label=1 rows in `out`.
        n = sum(1 for r in _csv.DictReader(open(out, newline='', encoding='utf-8'))
                if str(r.get('label')).strip() in ('1', 'True', 'true'))
        print(f"HF mode: {n} Req#2 positives (label=1) in {out} (no .eml export)")
    else:
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
