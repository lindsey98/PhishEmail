import copy

from OpenAttack.attackers import SCPNAttacker

from textattack.transformations import Transformation
from seqattack.utils.ner_attacked_text import NERAttackedText
import ssl
from typing import List
import random
ssl._create_default_https_context = ssl._create_unverified_context # fix the ssl certificate expiration error
import nltk
# Function to select the template based on the sentence structure


class ParaphraseTransformation(Transformation):
    """
        Generates paraphrases of the input sentence,
        re-mapping the original ground truth and making
        sure named entities are preserved.

        Based on (as implemented in OpenAttack):

        Adversarial Example Generation with Syntactically Controlled
        Paraphrase Networks.

        Mohit Iyyer, John Wieting, Kevin Gimpel, Luke Zettlemoyer.
        NAACL-HLT 2018.

        `[pdf] <https://www.aclweb.org/anthology/N18-1170.pdf>`__
        `[code] <https://github.com/miyyer/scpn>`__
    """
    def __init__(self):
        self.attacker = SCPNAttacker()
        self.templates=[
                        '( ROOT ( S ( VP ) ) ) EOP',
                        '( ROOT ( NP ( NP ) ) ) EOP',
                        '( ROOT ( S ( LST ) ( VP ) ) ) EOP',
                        '( ROOT ( S ( PP ) ( , ) ( NP ) ( VP ) ) ) EOP',
                        '( ROOT ( S ( ADVP ) ( NP ) ( VP ) ) ) EOP',
                        '( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ) ) EOP'
                    ]

    @staticmethod
    def get_entity_from_context(contexts, entities_dict, used_entities):
        """ Helper function to find the matching entity in the given contexts. """
        for context in contexts:
            if context in entities_dict and context not in used_entities:
                used_entities.add(context)
                return entities_dict[context]["token"], entities_dict[context]["truth"]
        return None, None

    def _select_template(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        any_verb = any(filter(lambda x: "VB" in x[1], nltk.pos_tag(tokens)))

        # Logic to select appropriate template based on flags
        if not any_verb: # the original sentence doesnt have verb
            return self.templates[1]  # '( ROOT ( NP ( NP ) ) ) EOP'
        else:
            pick_one = random.choice(self.templates)
            while pick_one == '( ROOT ( NP ( NP ) ) ) EOP':
                pick_one = random.choice(self.templates)
            return pick_one

    def _sentence_segmentation(self, tokens: List[str], ground_truth: List[int], cls_to_modify: List[int]):
        '''
        Segment the paragraphs based on the entities
        :param tokens:
        :param ground_truth:
        :param cls_to_modify:
        :return:
        '''
        # Precompute cls_to_modify + 1 for faster lookup
        cls_to_modify_plus_one = set([x + 1 for x in cls_to_modify]) # I-xxx

        sentences_to_be_rephrased = []
        sentences_to_be_rephrased_entity_id = []
        other_sentences = []
        all_sentences = []

        current_sent = []
        current_sent_id = 0
        current_rest = []

        for token, truth in zip(tokens, ground_truth):
            token = token.lower()
            if truth in cls_to_modify:  # Start of entity
                if current_rest:
                    rest_sentence = ' '.join(current_rest).strip()
                    other_sentences.append(rest_sentence)
                    all_sentences.append(rest_sentence)
                    current_rest = []
                current_sent.append(token)
                current_sent_id = truth

            elif truth in cls_to_modify_plus_one and current_sent:  # Continue collecting the entity
                current_sent.append(token)

            else:  # Collect non-entity tokens
                if current_sent:  # End of entity, append the collected sentence
                    rephrased_sentence = ' '.join(current_sent).strip()
                    sentences_to_be_rephrased.append(rephrased_sentence)
                    all_sentences.append(rephrased_sentence)
                    sentences_to_be_rephrased_entity_id.append(current_sent_id)
                    current_sent = []
                    current_sent_id = 0
                current_rest.append(token)  # Add to non-entity part

        # Append any remaining sentences
        if current_sent:
            rephrased_sentence = ' '.join(current_sent).strip()
            sentences_to_be_rephrased.append(rephrased_sentence)
            all_sentences.append(rephrased_sentence)
            sentences_to_be_rephrased_entity_id.append(current_sent_id)

        if current_rest:
            rest_sentence = ' '.join(current_rest).strip()
            other_sentences.append(rest_sentence)
            all_sentences.append(rest_sentence)

        return sentences_to_be_rephrased, other_sentences, all_sentences, sentences_to_be_rephrased_entity_id

    def _get_transformations(self, current_text: NERAttackedText, cls_to_modify: List[int]):
        """
            Returns paraphrases of the input text, mapping named entities to the new ground truth.
        """
        cls_to_modify = [5]
        tokens = current_text.text.split(" ")
        ground_truth = current_text.attack_attrs["ground_truth"]

        sentences_to_be_rephrased, other_sentences, all_sentences, sentences_to_be_rephrased_entity_id = \
            self._sentence_segmentation(tokens=tokens, ground_truth=ground_truth, cls_to_modify=cls_to_modify)

        # Collect all other entities first with a context window (bigram or trigram) => we need to keep track
        entities = {}
        cls_to_modify_plus_one = set([x + 1 for x in cls_to_modify])

        for idx, (token, truth) in enumerate(zip(tokens, ground_truth)):
            if truth == 0 or truth in cls_to_modify or truth in cls_to_modify_plus_one:
                continue
            # Create context window (trigram with the token and neighboring words)
            context = ' '.join(tokens[max(0, idx - 1):min(len(tokens), idx + 2)]).lower()
            entities[context] = {  # track entities by their context
                "token": token,
                "truth": truth
            }

        # Faster lookup of sentence indices by using a dictionary
        sentence_index_map = {sent: idx for idx, sent in enumerate(all_sentences)}

        all_sentence_candidates = {}
        rephrased_entities = {}
        # Generate k candidates for each sentence
        for i, sent in enumerate(sentences_to_be_rephrased):
            if sent.strip():  # Skip empty sentences
                sent_entity_id = sentences_to_be_rephrased_entity_id[i]
                sent_candidate = self.attacker.gen_paraphrase(
                    sent=sent,
                    templates=[self._select_template(sent)]
                )[0]  # generate only ONE candidate
                sent_index = sentence_index_map[sent]  # Get the sentence index efficiently
                all_sentence_candidates[(sent, sent_index)] = sent_candidate  # record the rephrased versions

                rephrased_tokens = sent_candidate.split(' ')
                for j, token in enumerate(rephrased_tokens):
                    # Use the rephrased sentence's context (trigram) for entity matching
                    rephrased_context = ' '.join(rephrased_tokens[max(0, j - 1):min(len(rephrased_tokens), j + 2)]).lower()
                    rephrased_entities[rephrased_context] = {
                        "token": token,
                        "truth": sent_entity_id if j == 0 else sent_entity_id + 1
                    }

        # Efficient string concatenation
        candidate_paragraphs = []
        last_index = 0

        # Concatenate the aligned sentence candidates to form k different paragraphs
        for (sent, sent_index), cdn_sent in all_sentence_candidates.items():
            if last_index == len(all_sentences):  # reached the end
                candidate_paragraphs.append(cdn_sent)
            else:
                candidate_paragraphs.append(' '.join(all_sentences[last_index:sent_index]) + ' ' + cdn_sent)
            last_index = sent_index + 1

        if last_index < len(all_sentences):
            candidate_paragraphs.append(' '.join(all_sentences[last_index:]))

        candidate_paragraphs = ' '.join(candidate_paragraphs)

        # Remap the ground-truth
        out_texts = []
        cnd_tokens = candidate_paragraphs.split(" ") # the altered tokens
        final_cnd_tokens, cnd_truth = [], []
        used_entities = set()  # Track which entities have been used to avoid duplicates

        for idx, cnd_token in enumerate(cnd_tokens):
            # Use the context window (trigram) around the rephrased token for matching
            cnd_context = ' '.join(cnd_tokens[max(0, idx - 1):min(len(cnd_tokens), idx + 2)]).lower()
            left_cdn_context = ' '.join(cnd_tokens[max(idx-1, 0):idx+1]).lower() # including the token itself
            right_cdn_context = ' '.join(cnd_tokens[idx:min(len(cnd_tokens), idx + 2)]).lower()

            # Check both entities and rephrased entities in the context windows
            contexts_to_check = [cnd_context, left_cdn_context, right_cdn_context]

            token, truth = self.get_entity_from_context(contexts_to_check, entities, used_entities)
            if token is None:
                token, truth = self.get_entity_from_context(contexts_to_check, rephrased_entities, used_entities)

            # Append the matched token and truth, or the default values if no entity is found
            if token is not None:
                final_cnd_tokens.append(token)
                cnd_truth.append(truth)
            else:
                final_cnd_tokens.append(cnd_token)
                cnd_truth.append(0)

        attack_attrs = copy.deepcopy(current_text.attack_attrs)
        attack_attrs["ground_truth"] = cnd_truth

        final_text = " ".join(final_cnd_tokens)

        out_texts.append(
            NERAttackedText(
                final_text,
                attack_attrs=attack_attrs
            )
        )

        return out_texts
