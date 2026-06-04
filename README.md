# PiMRef

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](pixi.toml)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)

> **PiMRef** — *Detecting and Explaining Ever-evolving Spear-Phishing Emails with Knowledge-Base Invariants.*

PiMRef is a lightweight, explainable, and generalizable anti-phishing framework.
Instead of classifying text as "spam vs. ham" — which attackers evade simply by
rewriting their copy — PiMRef checks a single, hard-to-fake invariant:

> **Does the identity an email *claims* match the domain it actually comes from?**

It rests on two pillars:

- **Identity fact-checking** — verify claimed identities, whether internal
  (HR, CEO) or external (PayPal, Alibaba), and flag impostors.
- **Intent analysis** — detect engagement instructions (click links, open
  attachments, share credentials) to surface attacker intent early and suppress
  false positives.

> [!NOTE]
> Spear phishing is a moving target: as soon as attackers change tactics,
> content classifiers must be retrained. AI-generated content (LLM-written copy,
> deepfakes, voice cloning) makes the problem worse. PiMRef instead targets the
> one thing an attacker cannot easily fake — the verifiable link between a
> claimed identity and its official domain.

## Table of Contents

- [Threat Model](#threat-model)
- [How It Works](#how-it-works)
- [Environment](#environment)
- [Setup](#setup)
- [Usage](#usage)
  - [Input formats](#input-formats)
  - [Running inference](#running-inference)
  - [Output](#output)
- [Project Structure](#project-structure)
- [Email Best Practices](#email-best-practices)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Threat Model

**PiMRef targets content-based identity deception — not sender spoofing.**

Modern email-authentication protocols — **SPF**, **DKIM**, and **DMARC** —
already address *sender spoofing*: they verify that a message genuinely
originates from the domain in its `From:` header and was not forged in transit.
**PiMRef is complementary to these protocols, not a replacement.**

We assume an attacker who:

- **Uses their own, legitimately-owned email address** (e.g.
  `billing@account-verify.example`) and therefore **passes** SPF/DKIM/DMARC —
  no spoofing is involved; and
- **Impersonates a trusted identity through the email *content*** — claiming to
  be "PayPal", "your IT department", "the CEO", etc., while sending from a domain
  that has nothing to do with that identity.

This is precisely the gap left open by authentication protocols: a message can be
perfectly authenticated yet still be phishing, because authentication verifies
*the sending domain*, not *the identity the content claims to represent*. PiMRef
closes that gap by fact-checking the claimed identity against its known official
domain(s).

| Layer | Question it answers | Handled by |
| --- | --- | --- |
| Transport authentication | "Did this really come from the domain in `From:`?" | SPF / DKIM / DMARC |
| **Content identity** | "Does the identity the email *claims* match the domain it came from?" | **PiMRef** |

> [!IMPORTANT]
> Deploy PiMRef *alongside* SPF/DKIM/DMARC for defense in depth: authentication
> stops forged senders, and PiMRef catches authenticated-but-deceptive senders.

## How It Works

PiMRef frames phishing detection as an **identity consistency check** rather than
content classification — which is what makes it robust to ever-evolving,
AI-generated phishing. Each email flows through four stages:

1. **Identity & intent recognition (NER).** A fine-tuned token-classification
   model (`lib/encoder/IdentityBert.py`) reads the subject, sender name, and body
   and extracts three things:
   - **Claimed identities** — the brand/organization the email purports to be
     from (e.g. *PayPal*, *Alibaba*).
   - **Internal relations** — claimed internal roles (e.g. *IT Support*, *HR team*).
   - **Call-to-action instructions** — engagement prompts (click a link, open an
     attachment, reply with credentials), plus any URL that immediately follows a
     CTA (the "next step of engagement").

2. **Knowledge-base matching (+ expansion agent).** Each claimed identity is
   matched against a CharacterBERT identity knowledge base (`lib/reference_db/`)
   via FAISS nearest-neighbour search, yielding the identity's *official* email
   domains. If an identity is **not present** in the knowledge base, the
   **expansion agent** (`lib/reference_db/db_expansion_agent.py`) searches the web
   for the organization's official domains and caches them for reuse. *(The agent
   uses the OpenAI API — see [Running inference](#running-inference).)*

3. **Consistency check.** The sender domain (unioned with the Reply-To /
   call-to-action URL domains) is compared against the claimed identity's official
   domains. A mismatch means the email claims one identity but is sent/answered
   from an unrelated domain — the core impostor signal. Claimed internal roles are
   checked analogously: an internal role arriving from a domain outside the
   recipient's own organization is flagged.

4. **Intent gating (false-positive reduction).** An identity mismatch alone is
   **not** reported as phishing. The email is flagged only when it **also**
   contains at least one call-to-action — a legitimate-but-misconfigured sender
   with no actionable request is treated as benign. This `check_action` gate
   substantially reduces false positives.

Every prediction is explainable: it carries the matched identity, the recognized
call-to-action(s), and the domain mismatch that triggered the verdict.

## Environment

**Hardware**

| Resource | Recommended |
| --- | --- |
| GPU VRAM | ≥ 16 GB |
| System RAM | ≥ 8 GB |
| Disk | ≥ 16 GB free |

**Software**

- **OS:** Linux (tested on Ubuntu 20.04.6 LTS)
- **Pixi:** 0.49.0 ([install guide](https://pixi.sh/dev/installation))
- **CUDA:** 12.1 (optional, for GPU acceleration)

## Setup

```bash
# 1. System libraries
sudo apt-get update
sudo apt-get install -y poppler-utils

# 2. Python environment (managed by Pixi)
cd PhishEmail/
pixi install
pixi run playwright install chromium   # email rendering uses headless Chromium

# 3. Pre-trained model checkpoints
pixi run get-model                      # or: bash scripts/get_model.sh
```

> All commands are run **from the project root**, so the relative `./checkpoints`,
> `./datasets`, and `./lib` resource paths resolve correctly.

## Usage

### Input formats

`--email_dir` accepts any of the following — they are all normalized
automatically, so you never need to pre-convert anything:

| Input | Notes |
| --- | --- |
| **Folder** of `.eml` / `.txt` files | Scanned recursively. Any `.mbox`/`.pst`/`.msg` files found inside are expanded too. |
| A **single** `.eml` / `.txt` file | The full raw email (headers + body). |
| A **`.mbox`** mailbox export | Each message is extracted to its own `.eml`. |
| A **`.msg`** Outlook message | Supported via the `extract-msg` package (installed by default). |
| A **`.pst`** Outlook archive | Requires the optional `pypff` backend (`pip install libpff-python`). |

Unsupported file types raise a clear error listing the accepted formats.

### Running inference

```bash
pixi run inference --email_dir [folder | file.eml | mailbox.mbox | archive.pst | message.msg]
# equivalently: pixi run python apps/cli/inference.py --email_dir [...]
```

A ready-to-run example lives in [`examples/`](examples/).

| Flag | Default | Description |
| --- | --- | --- |
| `--email_dir` | *(required)* | A directory, a single `.eml`/`.txt`, or a `.mbox`/`.pst`/`.msg` container. |
| `--output_csv` | `<timestamp>_results.csv` | Where to write results. Re-running with the same path **resumes**, skipping emails already present. |
| `--save_vis` | off | Save per-email HTML visualizations of the recognized entities. |
| `--vis_dir` | `./datasets/vis` | Output directory for the visualizations. |
| `--auto_translate` | off | Translate non-English emails (subject + body) before analysis. |
| `--run_dfence` | off | Also run the D-Fence baseline for comparison. |
| `--run_helphed` | off | Also run the HelpHed baseline for comparison. |

> [!NOTE]
> **Knowledge-base expansion requires an OpenAI API key.** When an identity is
> not already in the knowledge base, the expansion agent calls the OpenAI API.
> Set `OPENAI_API_KEY` in your environment (or place the key in
> `datasets/openai_key.txt`). Expansion is toggled by `knowledge_expansion_on` in
> `lib/config.py` (default `True` for the CLI; the Outlook server runs with it
> disabled). If you don't need expansion, leave it off and no key is required.

### Output

Results are written to a CSV with the following columns:

| Column | Description |
| --- | --- |
| `email_file_path` | Path to the email file |
| `sender_name` | Sender's name |
| `sender_address` | Sender's email address |
| `to_names` | Recipient name(s) |
| `to_addresses` | Recipient email address(es) |
| `subject` | Email subject |
| `sender_identities` | Claimed brand/organization identities recognized by the NER |
| `sender_relation` | Claimed internal role(s) (e.g., IT Support, HR team) |
| `required_actions` | Call-to-action instructions extracted from the email |
| `next_step_of_engagement` | URL after a CTA, or the Reply-To address, used in the domain check |
| `matched_identity` | Imitated brand/role, or a status (e.g., "Consistent") |
| `our_pred` | `True` if predicted Phish |
| `our_runtime` | Time taken for identity extraction & matching (s) |

When `--run_dfence` and/or `--run_helphed` are passed, the corresponding baseline
columns are also populated: `dfence_pred`, `dfence_runtime`,
`helphed_stacking_pred`, `helphed_stacking_runtime`, `helphed_voting_pred`,
`helphed_voting_runtime` (otherwise left empty).

## Project Structure

```text
PhishEmail/
├─ apps/                 # Runnable entry points
│  ├─ cli/inference.py   # Batch inference CLI   (pixi run inference)
│  └─ server/app.py      # Flask analysis server (pixi run serve)
├─ lib/                  # Importable package
│  ├─ config.py          # Model/checkpoint configuration & knowledge-base loading
│  ├─ data/              # Email parsing, rendering & OCR
│  ├─ encoder/           # Identity NER model (IdentityBert)
│  ├─ decoder/           # LLaMA-based components
│  ├─ reference_db/      # CharacterBERT matcher & identity knowledge base
│  ├─ labeling/          # HTML/header annotation helpers
│  ├─ baselines/         # Baseline detectors (D-Fence, HelpHed, …)
│  ├─ adversary/         # Adversarial attack generation/evaluation
│  └─ utilities/         # Shared helpers (logging, email I/O, data utils)
├─ addin/                # Outlook task-pane add-in (frontend)
├─ examples/             # Synthetic sample emails + quickstart
├─ scripts/              # Operational scripts (get_model.sh)
├─ tests/                # Test suite (pixi run test)
├─ pixi.toml             # Environment & task definitions
└─ README.md
```

## Email Best Practices

PiMRef works best in an ecosystem of good email hygiene. These habits make
legitimate mail easier to verify — by humans **and** by tools like PiMRef — and
make phishing easier to spot.

**When sending email**

- Send important or action-requesting messages from your **official
  institutional address** (e.g. `name@university.edu`, `name@company.com`), not a
  personal or free-webmail account. This lets recipients — and automated checks —
  confirm that the claimed identity matches the sending domain.
- Configure **SPF, DKIM, and DMARC** for your domain so your mail authenticates
  correctly.
- Avoid asking recipients to click links or share credentials unexpectedly; when
  a link is necessary, point to an official domain they can recognize and verify.

**When receiving email**

- Be cautious with messages from **external organizations**, especially those
  that create urgency or request actions (clicking links, opening attachments,
  sharing credentials, making payments).
- Check that the **sender's domain** actually belongs to the organization it
  claims to represent — an email "from PayPal" sent from `paypal-secure.example`
  is a red flag.
- **Hover over links** to inspect the real destination before clicking, and watch
  for mismatches between the displayed text and the actual target.
- Don't let authority or urgency ("the CEO", "IT department", "act within 24
  hours") rush you. When in doubt, **verify through an independent channel** — the
  organization's official website or phone number — rather than replying or using
  contact details supplied in the email itself.

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup and pull-request guidelines, and note our
[Code of Conduct](CODE_OF_CONDUCT.md). For security issues, see
[SECURITY.md](SECURITY.md).

## Citation

If you use PiMRef in your research, please cite it. A machine-readable
[`CITATION.cff`](CITATION.cff) is provided (powering GitHub's "Cite this
repository" button), and a BibTeX entry is below.

> **Note:** the citation metadata is currently a scaffold — replace the `TODO`
> fields with the final authors, year, venue, and arXiv/DOI before publishing.

```bibtex
@article{pimref,
  title   = {PiMRef: Detecting and Explaining Ever-evolving Spear-Phishing Emails with Knowledge-Base Invariants},
  author  = {TODO-LastName, TODO-FirstName and others},
  year    = {TODO-YYYY},
  journal = {TODO (arXiv / venue)},
  url     = {https://github.com/your-org/PhishEmail}
}
```

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial
4.0 International (CC BY-NC 4.0)** license — see [LICENSE](LICENSE) for the full
text. In short: you may share and adapt the material with attribution, but **not
for commercial purposes**. For commercial licensing, please contact the authors.

[//]: # (## PiMRef as Outlook Add-in)
[//]: # ()
[//]: # (Integrate PiMRef's phishing detection into Outlook with a two-part setup:)
[//]: # ()
[//]: # (1. Outlook Task Pane Add-in - a client-side add-in you sideload into Outlook.)
[//]: # (2. PiMRef Server - a back-end service that handles phishing analysis requests.)
[//]: # ()
[//]: # (### Step 1: Install the Outlook Add-in)
[//]: # (Scaffold an Office Add-in project with `yo office` - Project type: Office Add-in Task Pane; Script type: TypeScript; Host: Outlook - then overwrite manifest.json, src/taskpane/, and assets/logo.png with the versions in addin/, and run `npm install && npm start`.)
[//]: # ()
[//]: # (### Step 2: Start the PiMRef server)
[//]: # (`pixi run serve`  # or: python apps/server/app.py  -- listens on port 5000 by default.)
[//]: # ()
[//]: # (### Step 3: Use PiMRef in Outlook)
[//]: # (Open Outlook desktop, select an email, click the PiMRef Add-in in the ribbon, and the task pane analyzes the selected email in real time.)
