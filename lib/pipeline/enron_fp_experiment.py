"""Enron inbox false-positive experiment (rebuttal RC-Q1 / meta-review Req. #2).

Measures PiMRef's false-positive rate on a large corpus of *legitimate* mail
(the Enron inbox is all benign, so any "phishing" verdict is a false positive),
and shows how an adaptive, per-user whitelist plus a DMARC exemption suppress the
false positives that come from legitimate identity-domain inconsistencies
(personal webmail used for work; mail relayed through a third-party service).

Deployment story this supports: PiMRef does not delete mail. A first-contact
sender whose claimed identity does not match its domain raises a one-time
inconsistency alert; if the user marks it a false positive, the
(sender_address, claimed_identity) pair is added to that user's whitelist and
never alerts again. On the all-benign Enron corpus, "the user always confirms the
false positive" is a clean simulation of an ideal user (every flag there IS a
false positive); a real deployment needs the human in the loop because a first
contact could genuinely be phishing.

Pipeline (run the middle step yourself, e.g. on a remote box):

    # 1) build the inbox .eml set + metadata from the raw CSV
    python -m lib.pipeline.enron_fp_experiment prep \
        --csv ./datasets/enron_emails.csv \
        --out-dir ./datasets/enron_inbox_eml \
        --metadata ./datasets/enron_inbox_meta.csv

    # 2) run PiMRef over the produced folder (its normal inference entrypoint)
    python inference.py --email_dir ./datasets/enron_inbox_eml \
        --output_csv ./datasets/enron_inbox_results.csv

    # 3) join + streaming whitelist/DMARC evaluation + plots
    python -m lib.pipeline.enron_fp_experiment eval \
        --results ./datasets/enron_inbox_results.csv \
        --metadata ./datasets/enron_inbox_meta.csv \
        --out-dir ./datasets/enron_fp_out
"""

import os
import re
import ast
import csv as _csv
import json
import email
import email.policy
from email.utils import parseaddr, parsedate_to_datetime
from collections import defaultdict

import click
import pandas as pd
from tldextract import tldextract

csv_field_limit = 10_000_000
_csv.field_size_limit(csv_field_limit)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def reg_domain(domain_or_addr: str) -> str:
    """Registrable domain (example.com) from a domain or an email address."""
    if not domain_or_addr:
        return ''
    d = domain_or_addr.split('@')[-1].strip().lower()
    ext = tldextract.extract(d)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ''


def join_key(path: str) -> str:
    """Stable join key = '<user>/<filename>', robust to path prefixes so the
    PiMRef results CSV and the metadata CSV line up regardless of how
    --email_dir was spelled when inference ran."""
    parts = str(path).replace('\\', '/').rstrip('/').split('/')
    return '/'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]


# --------------------------------------------------------------------------- #
# auth / DMARC extraction  (works on real Authentication-Results headers; on the
# header-stripped Enron dump these are absent, so auth_status == 'none')
# --------------------------------------------------------------------------- #
def extract_auth(msg) -> dict:
    """Return {'dmarc','spf','dkim','dkim_domain','spf_domain','auth_domain'}.

    Parsed from Authentication-Results / ARC-Authentication-Results /
    Received-SPF / DKIM-Signature headers when present.
    """
    out = {'dmarc': 'none', 'spf': 'none', 'dkim': 'none',
           'dkim_domain': '', 'spf_domain': '', 'auth_domain': ''}
    ar = ' '.join(msg.get_all('Authentication-Results', []) +
                  msg.get_all('ARC-Authentication-Results', []))
    if ar:
        for mech in ('dmarc', 'spf', 'dkim'):
            m = re.search(mech + r'=(\w+)', ar, re.I)
            if m:
                out[mech] = m.group(1).lower()
        m = re.search(r'header\.d=([\w.-]+)', ar, re.I)
        if m:
            out['dkim_domain'] = reg_domain(m.group(1))
        m = re.search(r'smtp\.mailfrom=([\w.@-]+)', ar, re.I)
        if m:
            out['spf_domain'] = reg_domain(m.group(1))
    rspf = ' '.join(msg.get_all('Received-SPF', []))
    if rspf and out['spf'] == 'none':
        m = re.match(r'\s*(\w+)', rspf)
        if m:
            out['spf'] = m.group(1).lower()
        m = re.search(r'domain of ([\w.@-]+)', rspf, re.I)
        if m:
            out['spf_domain'] = reg_domain(m.group(1))
    dkim_sig = msg.get_all('DKIM-Signature', [])
    if dkim_sig and not out['dkim_domain']:
        m = re.search(r'd=([\w.-]+)', ' '.join(dkim_sig))
        if m:
            out['dkim_domain'] = reg_domain(m.group(1))
    out['auth_domain'] = out['dkim_domain'] or out['spf_domain']
    return out


def auth_status(row) -> str:
    """Coarse 'pass' / 'fail' / 'none' from the parsed auth fields."""
    if str(row.get('dmarc', 'none')) in ('pass', 'fail'):
        return str(row['dmarc'])
    if 'pass' in (str(row.get('dkim')), str(row.get('spf'))):
        return 'pass'
    if 'fail' in (str(row.get('dkim')), str(row.get('spf'))):
        return 'fail'
    return 'none'


# --------------------------------------------------------------------------- #
# 1) PREP
# --------------------------------------------------------------------------- #
@click.group()
def cli():
    pass


@cli.command()
@click.option('--csv', 'csv_path', default='./datasets/enron_emails.csv', show_default=True)
@click.option('--out-dir', default='./datasets/enron_inbox_eml', show_default=True,
              help='Directory of .eml files to feed to PiMRef inference.')
@click.option('--metadata', default='./datasets/enron_inbox_meta.csv', show_default=True,
              help='Per-email metadata (user, sender, timestamp, auth) for eval.')
@click.option('--limit', default=0, type=int, help='Cap total inbox emails (0 = all).')
@click.option('--max-per-user', default=0, type=int, help='Cap emails per user (0 = all).')
@click.option('--chunksize', default=50000, show_default=True, type=int)
def prep(csv_path, out_dir, metadata, limit, max_per_user, chunksize):
    """Filter the raw Enron CSV to inbox mail and write .eml files + metadata."""
    os.makedirs(out_dir, exist_ok=True)
    inbox_re = re.compile(r'/inbox/', re.I)
    per_user = defaultdict(int)
    meta_rows = []
    total = 0

    reader = pd.read_csv(csv_path, chunksize=chunksize)
    for chunk in reader:
        mask = chunk['file'].astype(str).str.contains(inbox_re)
        for _, r in chunk[mask].iterrows():
            if limit and total >= limit:
                break
            raw = str(r['message'])
            path = str(r['file'])
            user = path.replace('\\', '/').split('/')[0].strip()
            if max_per_user and per_user[user] >= max_per_user:
                continue
            try:
                msg = email.message_from_string(raw, policy=email.policy.default)
            except Exception:
                continue
            _, sender_addr = parseaddr(msg.get('From', ''))
            if '@' not in sender_addr:
                continue
            # timestamp -> epoch for chronological ordering
            try:
                dt = parsedate_to_datetime(msg.get('Date'))
                epoch = dt.timestamp()
                date_iso = dt.isoformat()
            except Exception:
                continue  # unusable timestamp -> cannot place in the stream
            auth = extract_auth(msg)
            seq = per_user[user]
            rel = f"{user}/{seq}.eml"
            fp = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(raw)
            meta_rows.append({
                'join_key': rel,
                'user': user,
                'sender_address': sender_addr.lower(),
                'from_domain': reg_domain(sender_addr),
                'date_iso': date_iso,
                'epoch': epoch,
                'auth_status': auth_status(auth),
                'auth_domain': auth['auth_domain'],
            })
            per_user[user] += 1
            total += 1
        if limit and total >= limit:
            break

    pd.DataFrame(meta_rows).to_csv(metadata, index=False)
    print(f"wrote {total} inbox emails to {out_dir}")
    print(f"users: {len(per_user)}  metadata: {metadata}")
    if meta_rows:
        na = sum(1 for m in meta_rows if m['auth_status'] != 'none')
        print(f"emails with usable auth headers: {na} "
              f"({100*na/len(meta_rows):.2f}%)  <- ~0 expected on the Enron dump")


# --------------------------------------------------------------------------- #
# 2) EVAL
# --------------------------------------------------------------------------- #
def _is_phish(val) -> bool:
    s = str(val).strip().lower()
    if s in ('1', 'phish', 'phishing', 'malicious', 'true', 'yes', 'positive'):
        return True
    try:
        return float(s) >= 0.5
    except ValueError:
        return False


def _norm_identity(matched, identities) -> str:
    """Claimed identity for the whitelist key: prefer matched_identity, else the
    extracted sender_identities set. Normalised (lowercased, de-braced)."""
    for cand in (matched, identities):
        s = str(cand).strip()
        if not s or s.lower() in ('nan', 'none', 'set()', '{}', '[]'):
            continue
        try:  # set/list literal e.g. "{'paypal'}"
            v = ast.literal_eval(s)
            if isinstance(v, (set, list, tuple)):
                s = ' '.join(sorted(str(x) for x in v))
        except (ValueError, SyntaxError):
            pass
        return re.sub(r'\s+', ' ', s.strip().strip('{}\'"').lower())
    return ''


def _load_identity_domains(path):
    """claimed-identity (lowercased) -> set of registrable domains, from the KB."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        db = json.load(f)
    out = {}
    for brand, domains in db.items():
        out[brand.strip().lower()] = {reg_domain(d) for d in domains if d}
    return out


@cli.command()
@click.option('--results', required=True, help='PiMRef inference output CSV.')
@click.option('--metadata', default='./datasets/enron_inbox_meta.csv', show_default=True)
@click.option('--out-dir', default='./datasets/enron_fp_out', show_default=True)
@click.option('--brand-db', default='./datasets/company_database.json', show_default=True,
              help='KB used to check DMARC alignment against the claimed identity.')
@click.option('--dmarc-mode', type=click.Choice(['header', 'off']), default='header',
              show_default=True,
              help="'header': exempt a flag when the message is DMARC/auth "
                   "authenticated for a domain aligned with the claimed identity "
                   "(inert on Enron, which has no auth headers). 'off': disable.")
def eval(results, metadata, out_dir, brand_db, dmarc_mode):
    """Streaming per-user whitelist + DMARC evaluation over PiMRef's verdicts."""
    os.makedirs(out_dir, exist_ok=True)
    res = pd.read_csv(results, engine='python')
    meta = pd.read_csv(metadata)

    # locate PiMRef columns defensively
    def col(df, *names):
        for n in names:
            if n in df.columns:
                return n
        return None
    c_path = col(res, 'email_file_path', 'file', 'path')
    c_pred = col(res, 'our_pred', 'pred', 'prediction')
    c_send = col(res, 'sender_address', 'from')
    c_match = col(res, 'matched_identity')
    c_ids = col(res, 'sender_identities', 'sender_identity')
    if not (c_path and c_pred):
        raise click.ClickException(
            f"results CSV missing required columns; found {list(res.columns)}")

    res['join_key'] = res[c_path].map(join_key)
    meta['join_key'] = meta['join_key'].map(join_key)
    df = res.merge(meta, on='join_key', how='inner', suffixes=('', '_meta'))
    if df.empty:
        raise click.ClickException("no rows joined between results and metadata "
                                   "(check that both refer to the same .eml set)")

    id_domains = _load_identity_domains(brand_db) if dmarc_mode == 'header' else {}

    df['pred_phish'] = df[c_pred].map(_is_phish)
    df['claimed_identity'] = [
        _norm_identity(df.iloc[i][c_match] if c_match else '',
                       df.iloc[i][c_ids] if c_ids else '')
        for i in range(len(df))]
    sender_col = c_send if c_send else 'sender_address'
    if c_send:
        df['sender_key'] = df[c_send].astype(str).str.lower()
    else:
        df['sender_key'] = df['sender_address'].astype(str).str.lower()

    def dmarc_exempt(row) -> bool:
        if dmarc_mode == 'off':
            return False
        if str(row.get('auth_status')) != 'pass':
            return False  # DMARC not passing -> no exemption (fall through)
        auth_dom = reg_domain(str(row.get('auth_domain', '')))
        if not auth_dom:
            return False
        # authenticated domain must align with the claimed identity's known domains
        doms = id_domains.get(row['claimed_identity'], set())
        return auth_dom in doms

    df['dmarc_exempt'] = df.apply(dmarc_exempt, axis=1)

    # chronological stream, per-user whitelist
    df = df.sort_values('epoch', kind='mergesort').reset_index(drop=True)
    whitelist = defaultdict(set)            # user -> {(sender, identity)}
    static_fp = adaptive_fp = interactions = dmarc_saves = 0
    per_user_interactions = defaultdict(int)
    timeline = []                            # (epoch, cum_static, cum_adaptive, cum_interactions)
    decisions = []

    for _, row in df.iterrows():
        user = row['user']
        flagged = bool(row['pred_phish'])
        key = (row['sender_key'], row['claimed_identity'])
        is_static_fp = flagged
        is_adaptive_fp = False
        action = 'ok'
        if flagged:
            if row['dmarc_exempt']:
                action = 'dmarc_exempt'
                dmarc_saves += 1
            elif key in whitelist[user]:
                action = 'whitelist_suppressed'
            else:
                action = 'flag_first_contact'
                is_adaptive_fp = True
                whitelist[user].add(key)      # simulate user confirming the FP
                interactions += 1
                per_user_interactions[user] += 1
        static_fp += int(is_static_fp)
        adaptive_fp += int(is_adaptive_fp)
        timeline.append((row['epoch'], static_fp, adaptive_fp, interactions))
        decisions.append({
            'join_key': row['join_key'], 'user': user, 'date_iso': row.get('date_iso'),
            'sender': row['sender_key'], 'claimed_identity': row['claimed_identity'],
            'pred_phish': flagged, 'action': action,
            'static_fp': is_static_fp, 'adaptive_fp': is_adaptive_fp,
        })

    n = len(df)
    print(f"\n=== Enron inbox FP experiment ===")
    print(f"emails evaluated (joined): {n}   users: {df['user'].nunique()}")
    print(f"static  FPR (raw PiMRef):        {static_fp}/{n} = {100*static_fp/n:.3f}%")
    print(f"adaptive FPR (whitelist+DMARC):  {adaptive_fp}/{n} = {100*adaptive_fp/n:.3f}%")
    if static_fp:
        print(f"reduction:                       "
              f"{100*(static_fp-adaptive_fp)/static_fp:.1f}% of FPs suppressed")
    print(f"DMARC exemptions:                {dmarc_saves} "
          f"({'inert on Enron - no auth headers' if dmarc_saves == 0 else ''})")
    print(f"human interactions (whitelist adds): {interactions}")
    if per_user_interactions:
        vals = sorted(per_user_interactions.values())
        print(f"  per-user interactions: mean={sum(vals)/len(vals):.2f} "
              f"median={vals[len(vals)//2]} max={vals[-1]}")

    dec_path = os.path.join(out_dir, 'per_email_decisions.csv')
    pd.DataFrame(decisions).to_csv(dec_path, index=False)
    print(f"per-email decisions -> {dec_path}")

    _plot_convergence(timeline, out_dir)


def _plot_convergence(timeline, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping plot: matplotlib unavailable: {e})")
        return
    idx = list(range(1, len(timeline) + 1))
    cum_static = [t[1] for t in timeline]
    cum_adaptive = [t[2] for t in timeline]
    cum_inter = [t[3] for t in timeline]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(idx, cum_static, label='cumulative FP (static PiMRef)', color='#c0392b')
    ax1.plot(idx, cum_adaptive, label='cumulative FP (adaptive)', color='#27ae60')
    ax1.plot(idx, cum_inter, label='cumulative whitelist adds (interactions)',
             color='#2980b9', linestyle='--')
    ax1.set_xlabel('emails processed (chronological)')
    ax1.set_ylabel('cumulative count')
    ax1.set_title('Enron inbox: FP suppression & human-interaction convergence')
    ax1.legend(loc='upper left')
    fig.tight_layout()
    out = os.path.join(out_dir, 'convergence.png')
    fig.savefig(out, dpi=150)
    print(f"convergence plot -> {out}")


if __name__ == '__main__':
    cli()
