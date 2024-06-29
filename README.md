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

1. 我们需要清理标注:
   1) 我们需要把organization和internal role（admin，helpdesk这种）统一转换成identity.
   2) 一个邮件 没有label =〉看是不是漏标 =〉确实没有就放过
   3) 一个邮件 有label =〉看有没有标全，比如 有可能一个邮件里面出现多个identity的不同形式，有可能有多句话是action
   4) 一个token可以拥有多个类别
3. 可能需要对action做data augmentation，目前的action句式很相似
4. 对于错误情况，需要仔细观察预测的概率
