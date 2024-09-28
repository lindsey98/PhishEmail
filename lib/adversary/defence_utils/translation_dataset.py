# coding=utf-8

# Datasets are downloaded from https://github.com/neuspell/neuspell/blob/master/data/traintest/download_datafiles.py

import os
import random

import datasets

_DESCRIPTION = """T&F translation dataset"""
_HOMEPAGE_URL = ""
_CITATION = ""

_VERSION = "1.0.0"
_BASE_NAME = "Typo correction"
_BASE_URL = ""

# Please note that only few pairs are shown here. You can use config to generate data for all language pairs
_LANGUAGE_PAIRS=[("noise", "clean")]

class PoeditConfig(datasets.BuilderConfig):
    def __init__(self, *args, srcl=None, tgtl=None, **kwargs):
        super().__init__(
            *args,
            name=f"{srcl}-{tgtl}",
            **kwargs,
        )
        self.srcl = srcl
        self.tgtl= tgtl


class Poedit(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        PoeditConfig(
            srcl=lang1,
            tgtl=lang2,
            description=f"Translating {lang1} to {lang2} or vice versa",
            version=datasets.Version(_VERSION),
        )
        for lang1, lang2 in _LANGUAGE_PAIRS
    ]
    BUILDER_CONFIG_CLASS = PoeditConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "translation": datasets.Translation(languages=(self.config.srcl, self.config.tgtl)),
                },
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE_URL,
            citation=_CITATION,
        )

    """ 
    download data files
    organize into splits
    """
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        train_source_file = os.path.join(os.getcwd(), "./datasets/typo_adversarial_training/train.noise")  # In total there are 60k sentences
        train_target_file = os.path.join(os.getcwd(), "./datasets/typo_adversarial_training/train")

        # Define the paths to your source and target files
        test_source_file = os.path.join(os.getcwd(), "./datasets/typo_adversarial_training/test.noise") # In total there are 60k sentences
        test_target_file = os.path.join(os.getcwd(), "./datasets/typo_adversarial_training/test")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "source_file": train_source_file,
                    "target_file": train_target_file,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "source_file": test_source_file,
                    "target_file": test_target_file,
                },
            ),
        ]

    """ 
    read data files 
    create examples from dataset
    """

    def _generate_examples(self, source_file, target_file):
        """Yields examples."""
        # Read the source and target files
        with open(source_file, encoding="utf-8") as src_f, open(target_file, encoding="utf-8") as tgt_f:
            for idx, (src_line, tgt_line) in enumerate(zip(src_f, tgt_f)):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                if src_line and tgt_line:
                    yield idx, {
                        "id": str(idx),
                        "translation": {
                            self.config.srcl: src_line,
                            self.config.tgtl: tgt_line,
                        },
                    }


## WHY not we create a dataset with organization names?

if __name__ == "__main__":
    import datasets
    from tqdm import tqdm
    from ..attack_utils import MyDeepWordBugAttacker

    # data = datasets.load_dataset("eriktks/conll2003")
    data = datasets.load_dataset("shrop/multinerd_en_filtered")
    # conll_label2id = {'O': 0,
    #             'B-PER': 1, 'I-PER': 2,
    #             'B-ORG': 3, 'I-ORG': 4,
    #             'B-LOC': 5, 'I-LOC': 6,
    #             'B-MISC': 7, 'I-MISC': 8}
    attacker = MyDeepWordBugAttacker()
    split = "train"
    for entry in tqdm(data[split]):
        tokens, tags = entry['tokens'], entry['ner_tags']
        if entry.get("lang"):
            if entry.get("lang") != "en":
                continue
        orig_text = " ".join(tokens) # join the original text

        entities = []
        start_idx = 0
        while start_idx < len(tags):
            if tags[start_idx] == 3: # B-ORG
                end_idx = start_idx + 1
                while end_idx < len(tags):
                    if tags[end_idx] == 4: # I-ORG
                        end_idx += 1
                    else:
                        break
                if tags[start_idx:end_idx]:
                    entities.append(' '.join(tokens[start_idx:end_idx]))
                start_idx = end_idx + 1
            else:
                start_idx += 1

        if entities:
            typo_injected = False
            for entry in entities:
                random_perturbation_type = random.choice(['repeat', 'delete', 'switch', 'replace'])
                perturbed_entry_choices = attacker.transform(random_perturbation_type, entry)
                if perturbed_entry_choices:
                    random_perturbed_entry = random.choice(perturbed_entry_choices)
                    new_text = orig_text.replace(entry, random_perturbed_entry)
                    typo_injected = True

            if typo_injected:
                with open(f"./datasets/typo_adversarial_training/{split}.noise", 'a+') as f:
                    f.write(new_text+'\n')
                with open(f"./datasets/typo_adversarial_training/{split}", 'a+') as f:
                    f.write(orig_text+'\n')
