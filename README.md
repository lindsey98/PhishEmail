# PiMRef

[![CI](https://github.com/your-org/PhishEmail/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/PhishEmail/actions/workflows/ci.yml)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](pixi.toml)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)

Welcome to **PiMRef**: **Detecting and Explaining Ever‑evolving Spear‑Phishing Emails with Knowledge‑Base Invariants**.

> [!NOTE]
> Spear phishing is a moving target. As soon as attackers change tactics, defenders often have to retrain or rebuild from scratch. At the same time, AIGC makes phishing content far more convincing, from LLM‑written copy to deepfakes and voice cloning.

> [!TIP]
> PiMRef is a lightweight, explainable, and generalizable anti‑phishing framework built on two pillars:
>
> - **Identity Fact‑Checking:** Verify claimed identities, whether internal (HR, CEO) or external (PayPal, Alibaba), and flag impostors.
> - **Intent Analysis via Engagement Instructions:** Detect action prompts (click links, open attachments, share credentials) to surface attacker intent early.


## ⚙️ Environment

**Hardware Requirements:**
- **GPU VRAM:** ≥ 16 GB  
- **System RAM:** ≥ 8 GB  
- **Disk Space:** ≥ 16 GB available  

**Software Requirements:**
- **OS:** Linux Ubuntu 20.04.6 LTS (tested)  
- **Pixi:** 0.49.0  
- **CUDA:** 12.1 (optional)  


## 🛠️ Setup

1. **Install Pixi**  
   Follow the official instructions at https://pixi.sh/dev/installation  

2. **Clone & Install Dependencies**  
   ```bash
   sudo apt-get update
   sudo apt-get install -y poppler-utils
   cd PhishEmail/
   pixi install
   pixi run playwright install chromium  # Email rendering uses headless Chromium
   bash scripts/get_model.sh  # or: pixi run get-model — downloads & extracts checkpoints
   ```

## 📁 Project Structure

```text
PhishEmail/
├─ apps/                 # Runnable entry points
│  ├─ cli/inference.py   # Batch inference CLI  (pixi run inference)
│  └─ server/app.py      # Flask analysis server (pixi run serve)
├─ lib/                  # Importable package
│  ├─ config.py          # Model/checkpoint configuration & knowledge base loading
│  ├─ data/              # Email parsing, rendering & OCR
│  ├─ encoder/           # Identity NER model (IdentityBert)
│  ├─ decoder/           # LLaMA-based components
│  ├─ reference_db/      # CharacterBERT matcher & identity knowledge base
│  ├─ labeling/          # HTML/header annotation helpers
│  ├─ baselines/         # Baseline detectors (D-Fence, HelpHed, …)
│  ├─ adversary/         # Adversarial attack generation/evaluation
│  └─ utilities/         # Shared helpers (logging, data utils)
├─ addin/                # Outlook task-pane add-in (frontend)
├─ scripts/              # Operational scripts (get_model.sh)
├─ tests/                # Test suite (pixi run test)
├─ pixi.toml             # Environment & task definitions
└─ README.md
```

> All commands are run from the project root so the relative `./checkpoints`,
> `./datasets`, and `./lib` resource paths resolve correctly.

## 🔎 How It Works

PiMRef frames phishing detection as an **identity consistency check** rather than
content classification — which is what makes it robust to ever‑evolving,
AIGC‑generated phishing. Each email flows through four stages:

1. **Identity & intent recognition (NER).** A fine‑tuned token‑classification
   model (`lib/encoder/IdentityBert.py`) reads the subject, sender name, and body
   and extracts three things:
   - **Claimed identities** — the brand/organization the email purports to be
     from (e.g. *PayPal*, *Alibaba*).
   - **Internal relations** — claimed internal roles (e.g. *IT Support*, *HR team*).
   - **Call‑to‑action instructions** — engagement prompts (click a link, open an
     attachment, reply with credentials), plus any URL that immediately follows a
     CTA (the "next step of engagement").

2. **Knowledge‑base matching (+ expansion agent).** Each claimed identity is
   matched against a CharacterBERT identity knowledge base (`lib/reference_db/`)
   via FAISS nearest‑neighbour search, yielding the identity's *official* email
   domains. If an identity is **not present** in the knowledge base, the
   **expansion agent** (`lib/reference_db/db_expansion_agent.py`) searches the web
   for the organization's official domains and caches them for reuse. *(The agent
   uses the OpenAI API — see [Run Inference](#run-inference).)*

3. **Consistency check.** The sender domain (unioned with the Reply‑To /
   call‑to‑action URL domains) is compared against the claimed identity's official
   domains. A mismatch means the email claims one identity but is sent/answered
   from an unrelated domain — the core impostor signal. Claimed internal roles are
   checked analogously: an internal role arriving from a domain outside the
   recipient's own organization is flagged.

4. **Intent gating (false‑positive reduction).** An identity mismatch alone is
   **not** reported as phishing. The email is flagged only when it **also**
   contains at least one call‑to‑action — a legitimate‑but‑misconfigured sender
   with no actionable request is treated as benign. This `check_action` gate
   substantially reduces false positives.

Every prediction is explainable: it carries the matched identity, the recognized
call‑to‑action(s), and the domain mismatch that triggered the verdict.

## Dataset Format

**Prepare your email data in one of two ways:**

### Option 1: Folder of `.eml` or `.txt` files
    ```text
    maildir/
    ├─ 1.eml
    ├─ 2.eml
    ├─ 3.txt
    └─ ...
    ```

> Each file contains the full raw email (headers + body).

### Option 2: Mailbox export

- Export your mailbox to a **`.mbox`** or **`.pst`** file.

## Run Inference

```bash
pixi run inference --email_dir [path/to/emails or .mbox/.pst file]
# equivalently: pixi run python apps/cli/inference.py --email_dir [...]
```

### Options

| Flag               | Default                   | Description                                                              |
| ------------------ | ------------------------- | ----------------------------------------------------------------------- |
| `--email_dir`      | *(required)*              | Folder of `.eml`/`.txt`, or a single `.mbox` / `.pst` file.             |
| `--output_csv`     | `<timestamp>_results.csv` | Where to write results. Re‑running with the same path **resumes**, skipping emails already in the file. |
| `--save_vis`       | off                       | Save per‑email HTML visualizations of the recognized entities.          |
| `--vis_dir`        | `./datasets/vis`          | Output directory for the visualizations.                                |
| `--auto_translate` | off                       | Translate non‑English emails (subject + body) before analysis.          |
| `--run_dfence`     | off                       | Also run the D‑Fence baseline for comparison.                           |
| `--run_helphed`    | off                       | Also run the HelpHed baseline for comparison.                           |

> **Knowledge‑base expansion requires an OpenAI API key.** When an identity is
> not already in the knowledge base, the expansion agent calls the OpenAI API.
> Set `OPENAI_API_KEY` in your environment (or place the key in
> `datasets/openai_key.txt`). Expansion is toggled by `knowledge_expansion_on` in
> `lib/config.py` (default `True` for the CLI; the Outlook server runs with it
> disabled). If you don't need expansion, leave it off and no key is required.

## Output Format

The results are saved as a CSV with the following columns:

| Column                    | Description                                                  |
| ------------------------- | ----------------------------------------------------------- |
| email\_file\_path         | Path to the email file                                      |
| sender\_name              | Sender’s name                                                |
| sender\_address           | Sender’s email address                                      |
| to\_names                 | Recipient name(s)                                            |
| to\_addresses             | Recipient email address(es)                                 |
| subject                   | Email subject                                                |
| sender\_identities        | Claimed brand/organization identities recognized by the NER |
| sender\_relation          | Claimed internal role(s) (e.g., IT Support, HR team)        |
| required\_actions         | Call‑to‑action instructions extracted from the email        |
| next\_step\_of\_engagement | URL after a CTA, or the Reply‑To address, used in the domain check |
| matched\_identity         | Imitated brand/role, or a status (e.g., “Consistent”)       |
| our\_pred                 | `True` if predicted Phish                                   |
| our\_runtime              | Time taken for identity extraction & matching (s)           |

When `--run_dfence` and/or `--run_helphed` are passed, the corresponding baseline
columns are also populated: `dfence_pred`, `dfence_runtime`,
`helphed_stacking_pred`, `helphed_stacking_runtime`, `helphed_voting_pred`,
`helphed_voting_runtime` (otherwise left empty).

----

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup and pull-request guidelines, and note our
[Code of Conduct](CODE_OF_CONDUCT.md). For security issues, see
[SECURITY.md](SECURITY.md).

## 📜 Citation

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

## ⚖️ License

This project is licensed under the **Creative Commons Attribution-NonCommercial
4.0 International (CC BY-NC 4.0)** license — see [LICENSE](LICENSE) for the full
text. In short: you may share and adapt the material with attribution, but **not
for commercial purposes**. For commercial licensing, please contact the authors.

[//]: # (## PiMRef as Outlook Add‑in)

[//]: # ()
[//]: # (Integrate PiMRef’s phishing detection into Outlook with a two‑part setup:)

[//]: # ()
[//]: # (1. **Outlook Task Pane Add-in** – A client-side add-in that you sideload into Outlook.  )

[//]: # (2. **PiMRef Server** – A back-end service that handles phishing analysis requests.)

[//]: # ()
[//]: # (### Step 1: Install the Outlook Add‑in)

[//]: # ()
[//]: # (#### a. Scaffold the Office Add‑in Project)

[//]: # ()
[//]: # (   1. **Install Yeoman and the Office generator**  )

[//]: # (       ```bash)

[//]: # (       npm install -g yo generator-office)

[//]: # (       ```)

[//]: # (   )
[//]: # (   2. **Create a new project**)

[//]: # (       ```bash)

[//]: # (       yo office)

[//]: # (       ```)

[//]: # (   )
[//]: # (   3. **When prompted, select:**)

[//]: # (    )
[//]: # (       **Project type:** `Office Add‑in Task Pane`)

[//]: # (       **Script type:** `TypeScript`)

[//]: # (       **Host:** `Outlook`)

[//]: # (    )
[//]: # (       _This creates a skeleton Outlook add‑in in a new directory._)

[//]: # ()
[//]: # ()
[//]: # (#### b. Replace with PiMRef Files)

[//]: # ()
[//]: # (In the generated project directory, overwrite the following with our versions from the `addin/` folder:)

[//]: # ()
[//]: # (- `manifest.json`)

[//]: # (- All files under `src/taskpane/`)

[//]: # (- `assets/logo.png`)

[//]: # ()
[//]: # ()
[//]: # (#### c. Run the Add‑in Locally)

[//]: # (    )
[//]: # (1. **Install dependencies**)

[//]: # (    ```bash)

[//]: # (    npm install)

[//]: # (    ```)

[//]: # ()
[//]: # (2. **Start the dev server & sideload the add‑in**)

[//]: # (    ```bash)

[//]: # (    npm start)

[//]: # (    ```)

[//]: # (- Starts Webpack on port `3000`)

[//]: # (- Automatically sideloads the add‑in into your Outlook &#40;Office 365 login required&#41;)

[//]: # ()
[//]: # (### Step 2: Set Up the PiMRef Server)

[//]: # ()
[//]: # (**Launch the server**)

[//]: # (```bash)

[//]: # (pixi run serve  # or: python apps/server/app.py)

[//]: # (```)

[//]: # ()
[//]: # (_The server listens on port `5000` by default._)

[//]: # ()
[//]: # (### Step 3: Use PiMRef in Outlook)

[//]: # ()
[//]: # (1. Open **Microsoft Outlook &#40;Desktop&#41;**.)

[//]: # (2. Select any email.)

[//]: # (3. Click the **PiMRef Add‑in** button in the ribbon, then choose **Show Task Pane**.)

[//]: # (4. The PiMRef pane appears and analyzes the selected email in real time.)

[//]: # ()
[//]: # (# Citations)

[//]: # (If you run into any issues, please open an issue in this repository.)

[//]: # (Happy phishing defense! 🚀)
