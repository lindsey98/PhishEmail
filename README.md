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

## Current problems

1. 我们需要清理标注: 1) 我们需要把organization和internal role（admin，helpdesk这种）统一转换成identity 2）一个token可以拥有多个类别
2. 可能需要对action做data augmentation，目前的action句式很相似
3. 对于错误情况，需要仔细观察预测的概率