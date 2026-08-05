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
from email.header import decode_header
from collections import defaultdict

import click
import pandas as pd
from tldextract import tldextract

csv_field_limit = 10_000_000
_csv.field_size_limit(csv_field_limit)

WEBMAIL_DOMAINS = {
    'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com', 'msn.com',
    'live.com', 'me.com', 'icloud.com', 'protonmail.com', 'gmx.com',
    '163.com', '126.com', 'qq.com', 'foxmail.com', 'sina.com', '139.com',
    'aliyun.com', 'yeah.net', 'sbcglobal.net', 'earthlink.net', 'comcast.net',
    'excite.com', 'juno.com', 'netscape.net', 'email.com',
}


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
@click.option('--metadata-only', is_flag=True, default=False,
              help='Rebuild ONLY the metadata CSV (skip writing .eml files). Use '
                   'to recover metadata for an already-run results CSV — the '
                   'deterministic <user>/<seq>.eml keys line up with a prior prep.')
def prep(csv_path, out_dir, metadata, limit, max_per_user, chunksize, metadata_only):
    """Filter the raw Enron CSV to inbox mail and write .eml files + metadata."""
    if not metadata_only:
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
            if not metadata_only:
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
    print(f"joined {len(df)}/{len(res)} results rows against metadata")
    # Alignment sanity check: the metadata is rebuilt deterministically, so the
    # <user>/<seq>.eml key should map to the same email. Verify via sender address.
    if c_send and 'sender_address' in meta.columns:
        a = df[c_send].astype(str).str.lower().str.strip()
        b = df['sender_address'].astype(str).str.lower().str.strip()
        match = (a == b).mean()
        print(f"alignment check (sender_address match): {100*match:.1f}%")
        if match < 0.95:
            print("  WARNING: low alignment — metadata seq numbering may not match "
                  "the results' .eml set; regenerate prep with the same code/input.")

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


@cli.command()
@click.option('--results', required=True, help='PiMRef inference output CSV over a benign corpus.')
@click.option('--brand-db', default='./datasets/company_database.json', show_default=True)
@click.option('--out', default=None, help='Write the isolated subset CSV here (for manual confirmation).')
def subset(results, brand_db, out):
    """Isolate the meta-review Requirement #2 hard subset and report FPR.

    Requirement #2 asks for the false-positive rate specifically on *legitimate*
    emails that CLAIM one organization but ARRIVE from a different domain
    (Figure 13: org identity + personal/webmail/third-party sender). Testing a
    whole benign corpus does not answer this — most mail is identity-consistent.

    This keeps only rows where PiMRef's own identity module matched a KNOWN brand
    whose official domain(s) do NOT contain the sender's registrable domain, i.e.
    exactly "org X claimed, sent from non-X domain". FPR = fraction PiMRef flags.

    Caveats to apply before reporting:
      - Confirm the rows are genuinely legitimate (spot-check / manual pass).
      - Re-run inference with strict matching (default now) to drop fuzzy
        mis-matches such as "American Airlines" -> "American Express".
      - Reuse this on the 2025 researcher corpus and on collected modern-ESP
        newsletters (Mailchimp/SendGrid/Substack), which are better modern
        representatives than the 2001 Enron dump.
    """
    db = json.load(open(brand_db))
    brand_dom = {k.strip().lower(): {reg_domain(x) for x in v if x} for k, v in db.items()}
    df = pd.read_csv(results, engine='python')

    def col(*names):
        for n in names:
            if n in df.columns:
                return n
        return None
    c_pred = col('our_pred', 'pred')
    c_send = col('sender_address', 'from')
    c_match = col('matched_identity')
    if not (c_pred and c_send and c_match):
        raise click.ClickException(f"results CSV missing columns; have {list(df.columns)}")

    def brand_of(v):
        s = str(v).strip()
        if not (s.startswith('{') or s.startswith('[')):
            return None
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (set, list, tuple)) and val:
                return str(list(val)[0]).strip().lower()
        except (ValueError, SyntaxError):
            pass
        return None

    rows = []
    for _, r in df.iterrows():
        b = brand_of(r[c_match])
        if not b or b not in brand_dom:
            continue
        sd = reg_domain(r[c_send])
        if not sd or sd in brand_dom[b]:
            continue  # consistent (or unpar%seable) — not a Req#2 case
        rows.append({
            'sender': r[c_send],
            'sender_domain': sd,
            'claimed_brand': b,
            'brand_official_domains': ';'.join(sorted(brand_dom[b])),
            'subject': r.get('subject', ''),
            'flagged_phishing': bool(_is_phish(r[c_pred])),
        })
    sub = pd.DataFrame(rows)
    n = len(sub)
    print(f"\n=== Requirement #2 subset (org claimed, sent from other domain) ===")
    print(f"corpus rows: {len(df)}   subset: {n}")
    if n:
        fp = int(sub['flagged_phishing'].sum())
        print(f"FPR on this hard subset: {fp}/{n} = {100*fp/n:.2f}%")
        print("(report honestly; pair with the RC-Q1 knowledge-base / user-interaction "
              "mitigation. Re-run with strict matching to drop fuzzy mis-matches.)")
    out = out or (os.path.splitext(results)[0] + '_req2_subset.csv')
    sub.to_csv(out, index=False)
    print(f"subset -> {out}  (spot-check these are genuinely legitimate)")


def _decode_hdr(value) -> str:
    if not value:
        return ''
    try:
        return ''.join(p.decode(c or 'utf-8', 'replace') if isinstance(p, bytes) else p
                       for p, c in decode_header(value)).strip()
    except Exception:
        return str(value).strip()


@cli.command(name='scan-req2')
@click.option('--email-dir', required=True,
              help='Folder scanned recursively for .eml (e.g. a volunteer inbox).')
@click.option('--brand-db', default='./datasets/company_database.json', show_default=True)
@click.option('--out', default='./datasets/req2_candidates.csv', show_default=True)
@click.option('--exclude', default=r'phish|junk|spam|honeypot|deleted|/sent', show_default=True,
              help='Regex on the path to skip (non-benign folders).')
@click.option('--min-brand-len', default=5, show_default=True, type=int,
              help='Ignore KB brand names shorter than this (cuts short-name noise).')
@click.option('--impersonator-brands', default=3, show_default=True, type=int,
              help='Drop any sender address that claims >= this many distinct brands '
                   '(synthetic/impersonation, not a legitimate org sender).')
def scan_req2(email_dir, brand_db, out, exclude, min_brand_len, impersonator_brands):
    """Pre-filter raw .eml for meta-review Requirement #2 candidates.

    Heuristic (matches how the field/ set was built): keep emails whose From
    DISPLAY NAME names a known organization (KB brand) while the sender's
    registrable domain is NOT that org's official domain — i.e. "claims org X,
    sent from a non-X domain". Splits into (A) third-party/relay domain and
    (B) personal-webmail sender. Drops impersonators (one address claiming many
    brands).

    This is a SUBSTRING pre-filter over display names, not PiMRef's matcher, so:
      - it has some false hits (e.g. 'intel' in 'IntelliSys'); confirm manually;
      - the reportable FPR should come from running PiMRef (strict) on the
        confirmed set and then the `subset` command.
    """
    db = json.load(open(brand_db))
    brands = {k.strip().lower(): {reg_domain(x) for x in v if x}
              for k, v in db.items() if len(k) >= min_brand_len and any(v)}
    pat = re.compile('|'.join(re.escape(b) for b in sorted(brands, key=len, reverse=True)))
    exc = re.compile(exclude, re.I) if exclude else None

    rows = []
    scanned = 0
    for root, _, files in os.walk(email_dir):
        if 'inbox' not in root.lower():
            continue
        if exc and exc.search(root):
            continue
        for fn in files:
            if not fn.lower().endswith('.eml'):
                continue
            fp = os.path.join(root, fn)
            scanned += 1
            try:
                with open(fp, 'rb') as fh:
                    msg = email.message_from_binary_file(fh, policy=email.policy.default)
            except Exception:
                continue
            nm, addr = parseaddr(msg.get('From', ''))
            nm = _decode_hdr(nm)
            if '@' not in addr or '@' in nm or not nm:
                continue
            sd = reg_domain(addr)
            if not sd:
                continue
            m = pat.search(nm.lower())
            if not m:
                continue
            b = m.group(0)
            if b not in brands or sd in brands[b]:
                continue
            cat = 'B: personal-webmail claims org' if sd in WEBMAIL_DOMAINS \
                else 'A: org via third-party domain'
            rows.append({
                'file': fp, 'from_name': nm, 'from_addr': addr, 'sender_domain': sd,
                'matched_brand': b, 'brand_domains': ';'.join(sorted(brands[b])),
                'category': cat, 'subject': _decode_hdr(msg.get('Subject', ''))[:80],
            })

    df = pd.DataFrame(rows)
    if not df.empty and impersonator_brands:
        by = df.groupby('from_addr')['matched_brand'].nunique()
        drop = set(by[by >= impersonator_brands].index)
        n_imp = int(df['from_addr'].isin(drop).sum())
        df = df[~df['from_addr'].isin(drop)]
    else:
        n_imp = 0

    print(f"\n=== scan-req2 ({email_dir}) ===")
    print(f"benign inbox .eml scanned: {scanned}")
    print(f"dropped as impersonators (>= {impersonator_brands} brands/sender): {n_imp}")
    print(f"Req#2 candidates: {len(df)}")
    if not df.empty:
        print(df['category'].value_counts().to_string())
        print(f"distinct (sender, brand): {df.groupby(['from_addr','matched_brand']).ngroups}")
    df.to_csv(out, index=False)
    print(f"candidates -> {out}  (manually confirm legitimacy; drop substring false hits)")


@cli.command()
@click.option('--results', required=True, help='PiMRef results CSV to review.')
@click.option('--whitelist', default='./datasets/whitelist.json', show_default=True,
              help='Persistent (sender, claimed-identity) whitelist (JSON).')
@click.option('--out', default=None, help='Reviewed results CSV (adds whitelisted / pred_after).')
@click.option('--apply-only', is_flag=True, default=False,
              help='Apply the existing whitelist without prompting (batch/re-run).')
def review(results, whitelist, out, apply_only):
    """Interactive human-in-the-loop whitelist (RC-Q1 mitigation).

    For each email PiMRef flagged as phishing, show it and ask whether to
    whitelist the (sender_address, claimed_identity) pair. Whitelisted pairs are
    suppressed now and in every future run (the whitelist persists to JSON), so a
    legitimate sender only ever prompts once. Answer y=whitelist (false positive),
    N=keep flagged, s=skip for now, q=quit.
    """
    df = pd.read_csv(results, engine='python')

    def col(*names):
        for n in names:
            if n in df.columns:
                return n
        return None
    c_pred = col('our_pred', 'pred')
    c_send = col('sender_address', 'from')
    c_match = col('matched_identity')
    c_ids = col('sender_identities', 'sender_identity')
    c_subj = col('subject')
    if not (c_pred and c_send):
        raise click.ClickException(f"results CSV missing columns; have {list(df.columns)}")

    wl = set()
    if os.path.exists(whitelist):
        try:
            for pair in json.load(open(whitelist)):
                wl.add((str(pair[0]).lower(), str(pair[1])))
        except Exception:
            pass

    def key(r):
        ident = _norm_identity(r[c_match] if c_match else '', r[c_ids] if c_ids else '')
        return (str(r[c_send]).lower().strip(), ident)

    def save():
        json.dump(sorted([list(x) for x in wl]), open(whitelist, 'w'), ensure_ascii=False, indent=0)

    flagged = df[df[c_pred].map(_is_phish)]
    print(f"{len(flagged)} flagged email(s); {len(wl)} whitelist pair(s) loaded.")
    newly = 0
    for _, r in flagged.iterrows():
        k = key(r)
        if k in wl:
            continue  # already whitelisted -> auto-suppressed
        if apply_only:
            continue
        print("-" * 64)
        print(f"  sender : {r[c_send]}")
        print(f"  claims : {r[c_match] if c_match else '(n/a)'}")
        if c_subj:
            print(f"  subject: {str(r[c_subj])[:70]}")
        try:
            ans = input("  Whitelist this sender? [y/N/s/q] ").strip().lower()
        except EOFError:
            break
        if ans == 'q':
            break
        if ans == 'y':
            wl.add(k)
            newly += 1
            save()  # persist immediately (crash-safe)

    df['whitelisted'] = df.apply(lambda r: key(r) in wl, axis=1)
    df['pred_after_whitelist'] = df[c_pred].map(_is_phish) & (~df['whitelisted'])
    before = int(df[c_pred].map(_is_phish).sum())
    after = int(df['pred_after_whitelist'].sum())
    print("=" * 64)
    print(f"flagged before: {before}   after whitelist: {after}   "
          f"(suppressed {before - after}; {newly} newly whitelisted this run)")
    save()
    out = out or (os.path.splitext(results)[0] + '_reviewed.csv')
    df.to_csv(out, index=False)
    print(f"whitelist -> {whitelist}   reviewed results -> {out}")


if __name__ == '__main__':
    cli()
