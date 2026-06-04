# PiMRef

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
   bash get_model.sh  # Downloads and extracts pre-trained model checkpoints
   ```

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
pixi run python inference.py \
  --email_dir [path/to/emails or .mbox/.pst file]
```

## Output Format

The results are saved as a CSV with the following columns:

| Column             | Description                                       |
| ------------------ | ------------------------------------------------- |
| email\_file\_path  | Path to the email file                            |
| sender\_name       | Sender’s name                                     |
| sender\_address    | Sender’s email address                            |
| to\_names          | Recipient name(s)                                 |
| to\_addresses      | Recipient email address(es)                       |
| subject            | Email subject                                     |
| sender\_identities | Recognized sender identity                        |
| sender\_relations  | Recognized sender–recipient relations             |
| required\_actions  | Next-step instructions extracted from the email   |
| matched\_identity  | Imitated brand or status (e.g., “Consistent”)     |
| our\_pred          | True if predicted Phish                           |
| our\_runtime       | Time taken for identity extraction & matching (s) |

----

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

[//]: # (python app.py)

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
