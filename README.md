# PimRef

# Introduction


---

Official repository for PimRef

# Setup

---

## Preparing the environment
1. Clone this repo

2. Run setup.sh, this will create a conda environment named **emailenv**.

```commandline
chmox +x setup.sh
./setup.sh
```

Make sure the directory structure is:
```
PhishEmail/
    |_ addin/ # for settup up outlook plugin
    |_ lib/
        |_ baselines/ # dfence, helphed, chatspamdetector, etc.
        |_ encoder/ # training and utils scripts for NER model
        |_ decoder/ # alternative Llama2 model for sender and call-to-action extraction
        |_ reference_db/ # utils scripts for the CharacterBERT model
    |_ checkpoints/
        |_ characterbert-typos-st-adv/ # this is the CharacterBERT model
        |_ identity-model/ # this is the NER model
        |_ company_database_names_field_study.json # this is a compact version of knowledge base, the brands inside have been manually cleaned
        |_ dfence_models/
        |_ helphed_models/
    |_ inference.py # main script
```

# Dataset format

---

- Option 1: Prepare a **folder of emails in .eml or .txt format.** 
The .eml/.txt contains the raw email with headers and content.
E.g.

  ```commandline
  maildir/
   |_ 1.eml
   |_ 2.eml
   |_ 3.txt
   ....
  ```

- Option 2: Alternatively, you can also **export your mailbox directly to the** **.mbox** or **.pst** format.

# Run inference

---

- Given a email folder, e.g.:
```commandline
python inference.py --email_dir maildir/
```

- For .mbox, e.g.:
```commandline
python inference.py --email_dir inbox.mbox
```

- For .pst, e.g.:
```commandline
python inference.py --email_dir inbox.pst
```

- To run the baselines as well, e.g.:
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

- **sender_identities**: recognized sender identity in email

- **sender_relations**: recognized sender-recipient potential relation 

- **required_actions**: next-step instruction from the email

- **matched_identity**: Imitated target brand | No Prediction | No Matched Brand | Consistent

- **our_pred**: if ``True`` => ``Phish``

- **our_runtime**: Time taken for identities extraction and identity matching


For example, if we have the following entry, it indicates that the email is 
- imitating DHL Express because we have detected the sender identity as DHL, but its sender address is not from the official contacts of DHL. 
- In addition, it has a required instruction for the recipients to "confirm the delivery details," which suggests the email is a phishing attempt.

| email_file_path | sender_name |            sender_address             | to_names | to_addresses | subject | email_body_text | sender_identities |        sender_relations                        |           required_actions           | our_pred |               matched_identity                | our_runtime |
|:---------------:|:-----------:|:-------------------------------------:|:--------:|:------------:|:-------:|:---------------:|:-----------------:|:----------------------------------------------:|:------------------------------------:|:---------------:|:---------------------------------------------:|:-----------:|
|       ...       |     DHL Shp     | sanjiv.bahl@rgnau.ac.in |   ...    |     ...      |   ...   |       ...       |       {'dhl shp'}       | set() | {'kindly find attached to track shp and confirm delivery details.'} |      True       | DHL Express |    0.023 |



# PimRef as Outlook Plugin

---

 
The PiMRef add-in consists of two main components:

1. **Outlook Add-in:** A task pane add-in to be sideloaded into an Outlook account (requires a valid Office 365 login).
2. **PiMRef Server:** A server responsible for processing phishing analysis requests from the add-in.

## Step 1: Install Outlook Add-in

1. **Set up the Office Add-in Project:**
   - Follow the official Microsoft documentation for setting up an Office Add-in using Yeoman: [Microsoft Office Add-ins Documentation](https://learn.microsoft.com/en-sg/office/dev/add-ins/develop/yeoman-generator-overview).
   - **Important**: During the setup ```yo office```, select the following options:
     - **Office Add-in Task Pane project**
     - **typescript**
     - **Outlook** 

2. **Replace Files in the Generated Directory** 
   - After running the Yeoman generator, replace the following files in the generated directory with the files from the PiMRef addin/ directory:
     - manifest.json 
     - src/taskpane/*
     - assets/logo.png

3. **Start the add-in locally by running the following command:**
    ```bash
    npm start
    ```
   - This will:
     - Start Webpack on the default port (3000). 
     - Sideload the add-in to the Outlook account you've configured.


## Step 2: Settup up PiMRef Server

1. **Install dependencies** 
```commandline
conda activate emailenv
pip install flask 
pip install flask-cors
```

2. **Run the server locally on the default port 5000**
```commandline
python app.py
```

## Step 3: Open your local Microsoft Outlook Desktop

- Select an email, **Toolbar** -> **PimRef Add-in** -> **Show Task Pane**
- Click **Check** to run detection
- Click **View Detailed Explanation** to show results explanations