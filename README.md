# PhishEmail

# Setup

---

## Preparing the ingredients
1. Clone the repo
```commandline
git clone https://github.com/lindsey98/PhishEmail.git
```

2. Create a conda environment and install the requirements
```commandline
conda create -n emailenv python=3.10
conda activate emailenv 
pip install -r requirements.txt
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

# Dataset format

---
Prepare a folder of emails in .eml or .txt format. The .eml/.txt contains the raw email with headers and content.
E.g.
```commandline
maildir/
 |_ 1.eml
 |_ 2.eml
 |_ 3.txt
 ....
```

# Run inference

---

```commandline
python inference.py --email_dir maildir/
```

# Output format

---

The output will be a CSV file saved as ``{today's date in YYYY-MM-DD}_results.csv``.

The CSV file has the following columns:

|    email_file_path     |           sender_name            |            sender_address             |               to_names               |               to_addresses               |    subject    |     email_body_text      | sender_identities |        sender_relations                        |           required_actions           |                                            is_inconsistent                                            |               matched_identity                |               identity_recog_runtime               |       identity_matching_runtime        |
|:----------------------:|:--------------------------------:|:-------------------------------------:|:------------------------------------:|:----------------------------------------:|:-------------:|:------------------------:|:-----------------:|:----------------------------------------------:|:------------------------------------:|:-----------------------------------------------------------------------------------------------------:|:---------------------------------------------:|:--------------------------------------------------:|:--------------------------------------:|
| Path to the email file | sender name from the "From" header | sender address from the "From" header | recipient names from the "To" header | recipient addresses from the "To" header | email subject | email body in plain text | recognized claimed sender identities | recognized sender-recipient potential relation | next-step instruction from the email | If True, we have detected the sender-address inconsistency and an instruction from the email => Phish | The imitated brand if is_inconsistent is True | Time taken for identities, instructions extraction | Time taken for imitated brand matching |

For example, if we have the following entry, it indicates that the email is imitating DHL Express because we have detected the sender identity as DHL, but its sender address is not from the official contacts of DHL. 
In addition, it has a required instruction for the recipients to "confirm the delivery details," which suggests the email is a phishing attempt.

| email_file_path | sender_name |            sender_address             | to_names | to_addresses | subject | email_body_text | sender_identities |        sender_relations                        |           required_actions           | is_inconsistent |               matched_identity                |               identity_recog_runtime               |       identity_matching_runtime        |
|:---------------:|:-----------:|:-------------------------------------:|:--------:|:------------:|:-------:|:---------------:|:-----------------:|:----------------------------------------------:|:------------------------------------:|:---------------:|:---------------------------------------------:|:--------------------------------------------------:|:--------------------------------------:|
|       ...       |     DHL Shp     | sanjiv.bahl@rgnau.ac.in |   ...    |     ...      |   ...   |       ...       |       {'dhl shp'}       | set() | {'kindly find attached to track shp and confirm delivery details.'} |      True       | DHL Express | 0.014865398406982422 |0.00941777229309082|


[//]: # (---)

[//]: # (## Setup the Docker container)

[//]: # (1. Build the docker image &#40;This may take some time&#41;)

[//]: # (```commandline)

[//]: # (sudo docker compose build)

[//]: # (```)

[//]: # (Make sure the docker image has been successfully built by verifying whether the image 'lindsey98/email' is listed in ``sudo docker images``.)

[//]: # ()
[//]: # (2. Run the docker container)

[//]: # (For Interactive mode &#40;suppose we use a proxy&#41;)

[//]: # (```commandline)

[//]: # (sudo docker compose up)

[//]: # (```)

[//]: # ()
[//]: # (For Detached mode)

[//]: # (```commandline)

[//]: # (sudo docker compose up -d)

[//]: # (```)

[//]: # ()
[//]: # (## Other useful commands)

[//]: # (# View contents even if the container has been exited for some reasons)

[//]: # (```commandline)

[//]: # (sudo docker run --rm -it --entrypoint /bin/bash lindsey98/email)

[//]: # (```)

[//]: # ()
[//]: # (# Stop the container)

[//]: # (```commandline)

[//]: # (sudo docker stop lindsey98/email)

[//]: # (```)

[//]: # ()
[//]: # (# Prune unused docker images)

[//]: # (```commandline)

[//]: # (sudo docker system prune)

[//]: # (```)

