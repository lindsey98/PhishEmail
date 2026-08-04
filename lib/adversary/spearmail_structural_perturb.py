"""Rebuttal experiment RA-Q1/Q2 (structural-perturbation half).

Randomly sample 500 SpearMail emails and ask GPT-4o-mini to restructure each one
(reorder the functional segments and relocate the invitation link) while
preserving the phishing intent and the recipient-specific content. Only the body
is regenerated; the original headers are kept.

This is the companion to benign_style_rewrite_gen.py: together they show PiMRef
does not merely pattern-match the SpearMail template. Reordering the segments and
moving the link shifts the call-to-action detector's input distribution (recall
86% -> 81% in the rebuttal), yet PiMRef still far outperforms every baseline
because it checks counterfactual identity-domain consistency, not surface layout.

Output: new .eml files written to --out-folder.

Usage:
  export OPENAI_API_KEY=...   # or ./datasets/openai_key.txt
  python -m lib.adversary.spearmail_structural_perturb \
      --in-folder ./datasets/v6 \
      --out-folder ./datasets/GPT_V6/v6_perturbed
Deps: pip install beautifulsoup4 openai
"""

import argparse
import email
import email.policy
import os
import random
from email.message import EmailMessage
from pathlib import Path

from bs4 import BeautifulSoup

SEED = 42

PERTURB_PROMPT = """You are rewriting a spear-phishing email to vary its DISCOURSE STRUCTURE while keeping the exact same intent, target, and factual content.

Apply ALL of the following transformations:
1. Reorder the functional segments (greeting, self-introduction, pretext/event, call-to-action, link, urgency, sign-off). Do NOT keep the original order. For example, lead with the call-to-action or the pretext instead of the greeting.
2. Relocate the link to a different position than in the original (e.g. early in the body, or mid-sentence).

Preserve exactly:
- The sender identity and any claimed organization.
- The recipient-specific details (name, research interest, activity).
- The pseudo-link URL itself (keep the same URL string, only move its position).
- The phishing intent.

Output ONLY the rewritten email body as plain text. No subject line, no explanations, no markdown.

Original email body:
{body}"""


def extract_body(msg) -> str:
    body = msg.get_body(preferencelist=("plain", "html"))
    if body is None:
        return ""
    content = body.get_content()
    if body.get_content_type() == "text/html":
        content = BeautifulSoup(content, "html.parser").get_text(separator="\n")
    if not isinstance(content, str):
        return ""
    return content.encode("utf-8", errors="ignore").decode("utf-8").strip()


def load_messages(folder: str, n_sample: int):
    items = []
    for p in sorted(Path(folder).rglob("*.eml")):
        try:
            with open(p, "rb") as f:
                msg = email.message_from_binary_file(f, policy=email.policy.default)
            body = extract_body(msg)
            if len(body.split()) >= 10:
                items.append((p, msg, body))
        except Exception as e:
            print(f"[skip] {p.name}: {e}")
    random.seed(SEED)
    if len(items) > n_sample:
        items = random.sample(items, n_sample)
    print(f"Loaded {len(items)} emails from {folder}")
    return items


def perturb_body(client, body: str, model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PERTURB_PROMPT.format(body=body)}],
        temperature=0.9,
    )
    return resp.choices[0].message.content.strip()


def rebuild_eml(orig_msg, new_body: str) -> bytes:
    """Keep original headers, replace body with new plain-text body."""
    out = EmailMessage()
    skip = {"content-type", "content-transfer-encoding",
            "content-disposition", "mime-version"}
    for k, v in orig_msg.items():
        if k.lower() not in skip:
            out[k] = v
    out.set_content(new_body)
    return out.as_bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-folder", default="./datasets/GPT_V6/v6")
    ap.add_argument("--out-folder", default="./datasets/GPT_V6/v6_perturbed")
    ap.add_argument("--n-sample", type=int, default=500)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--key-file", default="./datasets/openai_key.txt")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY") and os.path.exists(args.key_file):
        os.environ["OPENAI_API_KEY"] = open(args.key_file).read().strip()

    from openai import OpenAI
    client = OpenAI()

    out_dir = Path(args.out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_messages(args.in_folder, args.n_sample)
    ok, fail = 0, 0
    for i, (path, msg, body) in enumerate(items):
        out_path = out_dir / f"{path.stem}_perturbed.eml"
        if out_path.exists():
            ok += 1
            continue
        try:
            new_body = perturb_body(client, body, args.model)
            if len(new_body.split()) < 10:
                raise ValueError("perturbed body too short")
            out_path.write_bytes(rebuild_eml(msg, new_body))
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[fail] {path.name}: {e}")
        if (i + 1) % 25 == 0:
            print(f"progress {i + 1}/{len(items)}  ok={ok} fail={fail}")

    print(f"\nDone. wrote {ok} emails to {out_dir}, {fail} failed.")


if __name__ == "__main__":
    main()