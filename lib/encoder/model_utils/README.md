
### Introduction
This folder contains the training scripts for the sender identity recognition (as well as instruction recognition) model.

### Step-by-step Guide
- Step 1: First use ``data_augmentation.py`` to randomly rephrase the instructions with 50% of probability. This is because the original emails do not have a diverse pattern of instructions.
- Step 2: Then we use ``log_dataset.py`` to prepare the annotated datasets into the training format for the NER task. For the NER task, we have the following labels:
```
    ["O",
    "B-identity", # sender identity
    "I-identity",
    "B-relation", # sender relation to the recipient
    "I-relation",
    "B-action", # instruction
    "I-action"]
```
- Step 3: We finetune the BERT-large model ``finetune.py``. 
- Step 4: Optionally, we can also visualize the inference results ``evaluate.py``.