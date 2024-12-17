# D-Fence

D-Fence: A Flexible, Efficient, and Comprehensive Phishing Email Detection System

### Setting Up Environment

Use anaconda to set up the environment

```
conda create -n dfence python=3.7
conda activate dfence 
pip install -r requirements.txt
```

### Processing Emails

Before we can run inference or train models, we need to process the sample set of emails. This is done with the command

```python inference.py extract --input [EMAIL DIR] --output [FEATURE FILENAME] --label [0 or 1]```

where 0 corresponds to benign and 1 corresponds to phishing. You can also omit the label parameter if your emails are unlabelled.

The features will be stored in the features folder.

[//]: # (### Training Models)

[//]: # ()
[//]: # (The model we use &#40;test_v6&#41; is available in the model directory. However, if you want to train a model, you can do so with the command)

[//]: # ()
[//]: # (```python main.py train --model [MODEL NAME] --mal [MAL FEATURE FILENAME] --benign [BENIGN FEATURE FILENAME]```)

### Running Inference

[//]: # (To run inference on labelled datasets, we can run the command)

[//]: # ()
[//]: # (```python main.py test --model [MODEL NAME] --testmode labelled --mal [MAL FEATURE FILENAME] --benign [BENIGN FEATURE FILENAME]```)

[//]: # ()
[//]: # (The testmode and mal / benign flags can be removed if we want to test on only a specific dataset.)

For unlabelled datasets, we can run the command

```python inference.py test --model [MODEL NAME] --batch [UNLABELLED FEATURE FILENAME]```

The results will be output in the console, as well as saved in the reports folder.