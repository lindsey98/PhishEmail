# NER Adversarial Attack

## To attack the NER model

---

Select the attacking method from one of the following: 'bae', 'scpn', 'deepwordbug', 'gpt', 'viper', 'bart', 't5', 'concatsent'. E.g.
```commandline
pixi run python -m lib.adversary.ner_attack_gen --attacker bae 
```

If 'deepwordbug' is selected, you also need to also specify the typo type as 'repeat', 'delete', 'switch' or 'replace'. E.g.
```commandline
pixi run python -m lib.adversary.ner_attack_gen --attacker deepwordbug --typo_type repeat 
```

The following table explains the details of each attack
The following table illustrates the comparison of the attack models.

|  Attacker   | Attacking Class |  Perturbation   | Main Idea                                                                                                                                                                                                         |
|:-----------:|:---------------:|:---------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|     **BAE**     |    Identity     |   Token-level   | Mask the token before the entity, then use a pre-trained BERT model to predict the top-k candidates that would be best fit here, finally select the candidate that can decrease the confidence most signficantly. |
| **DeepWordBug** |    Identity     | Character-level | Replace, delete, switch or repeat one character in the entity                                                                                                                                                     |
|    **VIPER**    |    Identity     | Character-level | Replace one character in its superscript format e.g. 'a' =>'â'                                                                                                                                                    |
|     **GPT**     |     Action      | Sentence-level  | Use GPT to rephrase the original entity                                                                                                                                                                           |
|     **T5**      |      Action      | Sentence-level  | Use T5-based paragraphsing model to rephrase the original entity                                                                                                                                                  |
|    **BART**     |      Action      |      Sentence-level       | Use BART-based paragraphsing model to rephrase the original entity                                                                                                                                                |
| **ConcatSent**  |      Action      |      Sentence-level       | Move the entity to the end of its previous sentence                                                                                                                                                               |


## To evaluate the NER model

---

Select the attacking method from one of the following: 'bae', 'scpn', 'deepwordbug', 'gpt', 'viper', 'bart', 't5', 'concatsent'.
If 'deepwordbug' is selected, you also need to further specify the typo type as 'repeat', 'delete', 'switch' or 'replace'. 
Specify the --cls_to_attack to be 'identity' or 'action' based on the above Table.
Inference results will be saved in ``./datasets/nazario_results_adversarial_{attacker}.csv``
E.g.
```commandline
pixi run python -m lib.adversary.ner_attack_eval --attacker bae --cls_to_attack 'identity' 
```

# CharacterBERT Adversarial Attack

---

```commandline
pixi run python -m lib.adversary.db_attack_gen
```

