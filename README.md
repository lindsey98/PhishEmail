# PhishEmail

## Setup

1. Install Anaconda, create a conda environment with name email:
```commandline
conda create -n email
```

2. Install the requirements:
```commandline
conda activate email
pip install -r requirements.txt
```

3. Install the torch manually on https://pytorch.org/get-started/locally/. 

4. Process the annotated dataset in lib/encoder/wb_log_dataset.py, split them into training and testing.

5. Train the NER model in lib/encoder/wb_finetune_bert.py