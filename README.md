# PhishEmail

# Introduction

[//]: # ()
[//]: # (---)

[//]: # (There are two primary approaches in phishing email detection research:)

[//]: # ()
[//]: # (- Email Header Analysis: )

[//]: # (This line of research focuses on identifying spoofing signs within the email header. )

[//]: # (While such methods can detect certain phishing attempts, we believe the ultimate solution is to design robust email authentication protocols, such as SPF &#40;Sender Policy Framework&#41; and DKIM &#40;DomainKeys Identified Mail&#41;. )

[//]: # (These protocols address spoofing at its root, eliminating the need for complicated heuristics.)

[//]: # ()
[//]: # (- Content-Based Classification: )

[//]: # (The second approach involves classifying emails based on their content. )

[//]: # (Traditional methods rely on feature-engineering-based techniques which, despite achieving promising results on benchmark datasets, often lack explainability and transferability to new waves of phishing attacks. )

[//]: # (This limitation is especially significant in the current era of AI-generated content, where phishing emails can appear highly professional and personalized.)

[//]: # (In recent years, researchers have also explored the capability of Large Language Models &#40;LLMs&#41; in classifying phishing emails. )

[//]: # (However, decoder-based LLMs introduce impractical runtime overheads—approximately 8 seconds per email—and can produce hallucinated answers, reducing their effectiveness in real-world applications.)

[//]: # ()
[//]: # (In our work, we propose an explainable and efficient detection approach based on the concept of verifiable claims in phishing emails.)

[//]: # (We posit that phishing emails often imitate official organizations of which the recipient may be a customer or member, or they impersonate internal roles within the recipient's own organization.)

[//]: # (This imitation creates an information inconsistency between the claimed identity and the true identity &#40;as indicated by the sender's address&#41;. )

[//]: # (Such inconsistencies serve as strong indicators for phishing alerts and provide clear explanations for the detection.)

[//]: # ()
[//]: # (Furthermore, phishing emails typically include a call to action, such as prompting the recipient to visit a suspicious link, in an attempt to trick victims into financial loss or data breaches. )

[//]: # (Therefore, we also highlight potential instructions within the email when reporting a phishing alert.)

[//]: # ()
[//]: # (Building on these intuitions, we formulate phishing email detection as a Named Entity Recognition &#40;NER&#41; task. )

[//]: # (Our approach involves extracting the claimed identity and instructions from the email content. )

[//]: # (With the claimed identity identified, we cross-reference the sender's address against the official email addresses associated with this identity in our knowledge base. )

[//]: # (If an inconsistency is discovered and at least one instruction is present, we classify the email as phishing. )

[//]: # (Otherwise, we consider the email to be benign.)

# Setup

---

For **Ubuntu**

## Preparing the ingredients
1. Clone the repo
```commandline
git clone https://github.com/lindsey98/PhishEmail.git
```

2. Create a conda environment and install the requirements (pypff is not available via pip, so you need to build it from source.)
```commandline
conda create -n emailenv python=3.10
conda activate emailenv 
pip install -r requirements.txt
sudo apt-get update
sudo apt-get install libpff-dev
sudo apt install git autoconf automake autopoint libtool pkg-config
```

3. Go to the project dir, download the models
First download from Google drive
```commandline
gdown --id 10Gu1zlRiCS5ICglNJQGq-2wpPPLKqcZ8 -O checkpoints.zip
```

Then unzip the download file
```commandline
unzip checkpoints.zip -d checkpoints/
```

Make sure the directory structure is:
```
PhishEmail/
    |_ checkpoints/
        |_ characterbert-typos-st/
        |_ identity-model/
        
        |_ company_database_names.npy # this is a compact version of knowledge base, the brands inside have been manually cleaned
        |_ company_database_reps.npy
        |_ company_database_knowphish.json
        
        |_ internal_relation_names.npy 
        |_ internal_relation_reps.npy
        
        |_ company_database_names_v2.npy # this is an extended version of knowledge base, the brands have been partially verified
        |_ company_database_reps_v2.npy
        |_ company_database_knowphish_v2.json
        
        |_ dfence_models/
            |_ meta/
            |_ struct/
            |_ text/
            |_ url/
        
        |_ helphed_models/
            |_ dt_model.pkl
            |_ word2vec_model.kv
            |_ ......
```

4. (Optional) I am using Clash (to connect to VPN), which runs a proxy server on port 7890, 
so I need to set the http_proxy environment variable to http://127.0.0.1:7890.
```commandline
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
```

# Dataset format

---
Prepare a folder of emails in .eml or .txt format. 
The .eml/.txt contains the raw email with headers and content.
E.g.
```commandline
maildir/
 |_ 1.eml
 |_ 2.eml
 |_ 3.txt
 ....
```

Alternatively, you can also export your mailbox directly to the **.mbox** or **.pst** format.

# Run inference

---

For a email folder, e.g.:
```commandline
python inference.py --email_dir maildir/
```
or for .mbox, e.g.:
```commandline
python inference.py --email_dir inbox.mbox
```
or for .pst, e.g.:
```commandline
python inference.py --email_dir inbox.pst
```

To run the baselines as well, e.g.:
```commandline
python inference.py --email_dir maildir/ --run_dfence --run_helphed
```

# Output format

---

The output will be a CSV file saved as ``{today's date in YYYY-MM-DD}_results.csv``.

The CSV file has the following columns:

- **email_file_path**: Path to the email file

- **sender_name**: sender name

- **sender_address**: sender email address

- **to_names**: recipient names

- **to_addresses**: recipient email addresses

- **subject**: email subject

- **email_body_text**: email body in plain text

- **sender_identities**: recognized sender identity in email

- **sender_relations**: recognized sender-recipient potential relation 

- **required_actions**: next-step instruction from the email

- **matched_identity**: Imitated target brand | No Prediction | No Matched Brand | Consistent

- **our_pred**: if ``True`` => Found sender identity-address inconsistency => Phish

- **our_runtime**: Time taken for identities extraction and identity matching


Results interpretation:
- Case 1: **our_pred = True**: 
  - **Phishing**, and the matched_identity field returns the target brand, if matched_identity = 'Internal', the email is imitating an internal role such as colleague.

- Case 2: **our_pred = False, matched_identity = No Prediction**: 
  - Benign because we **didnt recognize any claimed identity** in the email.
  
- Case 3: **our_pred = False, matched_identity = No Matched Brand**: 
  - The email is reported as benign because the recognized sender identity is an **unknown brand**.
  
- Case 4: **our_pred = False, matched_identity = Consistent**: 
  - The email is reported as benign because the sender claimed identity and his sender email address are consistent.
  
- Case 5: **our_pred = False, matched_identity = target brand**: 
  - The email is reported as benign because there is **no required action** found in the email.

For example, if we have the following entry, it indicates that the email is imitating DHL Express because we have detected the sender identity as DHL, but its sender address is not from the official contacts of DHL. 
In addition, it has a required instruction for the recipients to "confirm the delivery details," which suggests the email is a phishing attempt.

| email_file_path | sender_name |            sender_address             | to_names | to_addresses | subject | email_body_text | sender_identities |        sender_relations                        |           required_actions           | our_pred |               matched_identity                | our_runtime |
|:---------------:|:-----------:|:-------------------------------------:|:--------:|:------------:|:-------:|:---------------:|:-----------------:|:----------------------------------------------:|:------------------------------------:|:---------------:|:---------------------------------------------:|:-----------:|
|       ...       |     DHL Shp     | sanjiv.bahl@rgnau.ac.in |   ...    |     ...      |   ...   |       ...       |       {'dhl shp'}       | set() | {'kindly find attached to track shp and confirm delivery details.'} |      True       | DHL Express |    0.023 |



