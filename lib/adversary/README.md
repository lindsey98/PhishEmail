# Instructions

This directory contains the adversary experiments for the NER model.

## To setup the environment

---

```commandline
pip install -r requirements.txt
```

## To attack 

---

Select the attacking method from one of the following: 'bae', 'scpn', 'deepwordbug', 'gpt', 'viper', 'bart', 't5', 'concatsent'. E.g.
```commandline
python attack_gen.py --attacker bae 
```

If 'deepwordbug' is selected, you also need to further specify the typo type as 'repeat', 'delete', 'switch' or 'replace'. E.g.
```commandline
python attack_gen.py --attacker deepwordbug --typo_type repeat 
```

The following table explains the details of each attack
The following table illustrates the comparison of the attack models.

|  Attacker   | Attacking Class |  Perturbation   | Main Idea                                                                                                                                                                                                                          |
|:-----------:|:---------------:|:---------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|     **BAE**     |    Identity     |   Token-level   | Mask the entity starting token, then use a pre-trained BERT model to predict the top-k candidates that would be the best token to insert at here, finally select the candidate that can decrease the confidence most signficantly. |
| **DeepWordBug** |    Identity     | Character-level | Replace, delete, switch or repeat one character in the entity                                                                                                                                                                      |
|    **VIPER**    |    Identity     | Character-level | Replace one character in its superscript format e.g. 'a' =>'â'                                                                                                                                                                     |
|     **GPT**     |     Action      | Sentence-level  | Use GPT to rephrase the original entity                                                                                                                                                                                            |
|     **T5**      |      Action      | Sentence-level  | Use T5-based paragraphsing model to rephrase the original entity                                                                                                                                                                   |
|    **BART**     |      Action      |      Sentence-level       | Use BART-based paragraphsing model to rephrase the original entity                                                                                                                                                                 |
| **ConcatSent**  |      Action      |      Sentence-level       | Move the entity to the end of its previous sentence                                                                                                             |


## To evaluate 

---

Select the attacking method from one of the following: 'bae', 'scpn', 'deepwordbug', 'gpt', 'viper', 'bart', 't5', 'concatsent'.
If 'deepwordbug' is selected, you also need to further specify the typo type as 'repeat', 'delete', 'switch' or 'replace'. 
Specify the --cls_to_attack to be 'identity' or 'action' based on the above Table.
Inference results will be saved in ``./datasets/nazario_results_adversarial_{attacker}.csv``
E.g.
```commandline
python attack_gen.py --attacker bae --cls_to_attack 'identity' 
```

Add --defence flag to activate the spelling corrector (A pre-trained T5-based typo correction model).
```commandline
python attack_gen.py --attacker deepwordbug --cls_to_attack 'identity' --typo_type repeat --defence
```