
from lib.reference_db.CharacterBert import CharacterBertModel, CharacterIndexer
from transformers import BertTokenizer
import os
import torch.nn.functional as F
import torch
import faiss
from tqdm import tqdm
import numpy as np
import json
from typing import List, Tuple, Union, Optional, Set, Any
from openai import OpenAI
import re
from tldextract import tldextract
from lib.utilities.gpt_utils import assistant_completion
from lib.utilities.logger import Logger, Timer
import difflib
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'


class CharacterBERT:
    _CallerPrefix = "CharacterBert"

    def __init__(self, model_id: str='./checkpoints/characterbert-typos-st/', return_cls: bool=True, do_l2_norm: bool=True) -> None:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = CharacterBertModel.from_pretrained(model_id)
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.indexer = CharacterIndexer()

        self.return_cls = return_cls
        self.do_l2_norm = do_l2_norm

    def _character_bert_tokenize(self, text):
        x = self.tokenizer.basic_tokenizer.tokenize(text)
        x = ['[CLS]', *x, '[SEP]']
        return x

    @torch.inference_mode()
    def __call__(self, input_texts: List[str]) -> Tuple[torch.Tensor, float]:
        with Timer() as timer:
            tokens = [self._character_bert_tokenize(t) for t in input_texts]
            batch_ids = self.indexer.as_padded_tensor(tokens, maxlen=512)
            embeddings_for_batch, _ = self.model(batch_ids.to(self.device)) # Batch_size x Sequence_length x Embed_dim

        if self.return_cls:
            embeddings_for_batch = embeddings_for_batch[:, 1, :]  # return CLS embedding # Batch_size x Embed_dim
        else:
            embeddings_for_batch = torch.mean(embeddings_for_batch[:, 1:-1, :], dim=1, keepdim=True)  # return Average embedding, remove [CLS] and [SEP]

        if self.do_l2_norm:
            embeddings_for_batch = F.normalize(embeddings_for_batch, p=2, dim=-1) # L2 normalization over Embed_dim
        return embeddings_for_batch, timer.interval


class BaseFaissIPRetriever:
    _CallerPrefix = "BaseFaissIPRetriever"

    def __init__(self, tags: List[str], init_reps: Optional[np.ndarray]=None, embed_model: Optional[CharacterBERT]=None):
        self.tags = tags

        if init_reps is None:
            assert embed_model is not None
            self.embed_model = embed_model
            Logger.spit("No cached representations, build index from scratch ..", caller_prefix=BaseFaissIPRetriever._CallerPrefix, debug=True)
            init_reps = self.build_reps_from_scratch()

        assert len(tags) == init_reps.shape[0], "Number of tags must match the number of representations."
        self.index = faiss.IndexFlatIP(init_reps.shape[1])
        self.index.train(init_reps)
        self.index.add(init_reps)
        Logger.spit("Index base of size {} with dimension {} is trained".format(init_reps.shape[0], init_reps.shape[1]),
                    caller_prefix=BaseFaissIPRetriever._CallerPrefix,
                    debug=True)

    def build_reps_from_scratch(self):
        index_reps = np.empty((0, 768))
        batch_size = 128
        tag_list = self.tags
        for i in range(0, len(tag_list), batch_size):
            batch = tag_list[i:min(i + batch_size, len(tag_list))]  # Get the next batch of brand names
            batch_embeddings, time = self.embed_model(batch).cpu().numpy()  # Predict embeddings for the batch
            index_reps = np.concatenate((index_reps, batch_embeddings), axis=0)  # Append new embeddings
        return index_reps

    def search(self, q_reps: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        with Timer() as timer:
            scores, indices = self.index.search(q_reps, k)
        tag_results = np.array(self.tags)[indices]
        return scores, indices, tag_results, timer.interval

    def add(self, p_reps: np.ndarray, p_tags: List[str]) -> None:
        assert len(p_tags) == p_reps.shape[0], "Number of tags must match the number of representations."
        self.index.add(p_reps)
        self.tags.extend(p_tags) # fixme: save the updated index again?

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        all_tags = []
        total_searching_time = 0

        for start_idx in tqdm(range(0, num_query, batch_size)):
            nn_scores, nn_indices, nn_tags, searching_time = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
            all_tags.append(nn_tags)
            total_searching_time += searching_time

        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)
        all_tags = np.concatenate(all_tags, axis=0)

        return all_scores, all_indices, all_tags, total_searching_time


class BrandMatcher:
    _CallerPrefix = "BrandMatcher"

    def __init__(self,
                brand_index_db: BaseFaissIPRetriever,
                internal_relation_index_db: BaseFaissIPRetriever,
                embed_model: CharacterBERT,
                brand_domain_map_path: str,
                knowledge_base_expansion: bool=False,
                gpt_client: Optional[OpenAI]=None, gpt_assistant: Optional[Any]=None,
                check_action: bool=True,
                threshold: float=0.95):

        self.brand_index_db = brand_index_db
        self.internal_relation_index_db = internal_relation_index_db
        self.brand_domain_map_path = brand_domain_map_path
        with open(self.brand_domain_map_path, 'r') as file:
            self.brand_domain_map = json.load(file)

        self.knowledge_expansion = knowledge_base_expansion
        self.check_action = check_action
        self.embed_model = embed_model
        self.threshold = threshold
        self.gpt_client = gpt_client
        self.gpt_assistant = gpt_assistant

    @staticmethod
    def contains_domain(text: str) -> Tuple[bool, Optional[List[str]]]:

        # Search for the pattern in the given text
        matches = re.findall(r'\b(?:[a-zA-Z0-9-]+\s?\.\s?)+[a-zA-Z]{2,}\b', text)

        # Clean up the matches by removing any spaces and checking if the TLD is valid
        cleaned_matches = []
        for match in matches:
            cleaned_match = match.replace(' ', '')
            # Extract the TLD and check if it's valid
            tld = tldextract.extract(cleaned_match).suffix
            if tld:
                cleaned_matches.append(cleaned_match)

        # Return True if a domain name is found, otherwise False
        return bool(cleaned_matches), cleaned_matches if cleaned_matches else None

    @staticmethod
    def filter_duplicates(strings: Union[List[str], Set[str]], threshold: float=0.8) -> List[str]:
        filtered = []
        for string in strings:
            # Remove "from:" or any case variation with optional spaces
            string = re.sub(r'(?i)from\s*:?', '', string)
            # Remove leading "##"
            string = re.sub(r'^\s*#+', '', string)
            string = string.strip()
            if len(string) == 0:
                continue
            # Check similarity with already filtered strings
            similar_found = False
            for i, f in enumerate(filtered):
                if len(f) == 0:
                    continue
                if difflib.SequenceMatcher(None, string, f).ratio() > threshold:
                    similar_found = True
                    # Keep the longer string
                    if len(string) > len(f):
                        filtered[i] = string
                    break
            if not similar_found:
                filtered.append(string)
        return filtered

    def find_closest_match(self, query: str, value_index_db: BaseFaissIPRetriever) -> Tuple[Optional[float], Optional[str]]:

        query_embed, embedding_time = self.embed_model([query.lower()]).cpu().numpy() # 1 x Embed_dim
        score, index, tag, searching_time = value_index_db.search(query_embed, 1) # 1-NN

        score = score.tolist()[0][0]
        tag = tag.tolist()[0][0]

        thre = self.threshold

        if len(query) <= 3: # require exact match, because I observe a FP case where 'pay' is matched to 'pay now'
            thre = 1

        if score >= thre:
            return score, tag
        else:
            return None, None

    def expand_knowledge_base(self, queried_identity: str) -> Tuple[Optional[List[str]], float]:

        with Timer() as timer:
            search_email = assistant_completion(client=self.gpt_client, query=queried_identity, assistant_id=self.gpt_assistant.id)
        searching_time = timer.interval
        Logger.spit(f'Searching {queried_identity} in GPT, return {search_email}, searching time = {searching_time}', caller_prefix=BrandMatcher._CallerPrefix, debug=True)

        email_regex = re.compile(
            r'^\[?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(,\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})*\]?$')

        if email_regex.fullmatch(search_email):  # if the return are valid email addresses
            emails = search_email.strip('[]').split(',')
            official_emails = list(set([email.split('@')[1].strip() for email in emails]))

            # update index DB
            queried_identity_embed, embedding_time = self.embed_model([queried_identity.lower()]).cpu().numpy()
            self.brand_index_db.add(queried_identity_embed, [queried_identity.lower()])

            # update the reference list
            self.brand_domain_map[queried_identity.lower()] = official_emails
            with open(self.brand_domain_map_path, 'w') as file:
                json.dump(self.brand_domain_map, file, indent=4)

            return official_emails, searching_time

        return None, searching_time

    def handle_external_emails(self, identities: Set[str]) -> Tuple[Union[None, str], Union[None, List[str]]]:

        for potential_organization in identities:
            if potential_organization.startswith("##"): # noisy prediction
                continue

            # Match against known brands or directly extract domains
            matched_score, closest_match = self.find_closest_match(query=potential_organization, value_index_db=self.brand_index_db)
            contains_a_domain, extracted_domains = self.contains_domain(potential_organization)

            if closest_match:
                official_emails = self.brand_domain_map[closest_match]
                official_domains = [tldextract.extract(x).domain + '.' + tldextract.extract(x).suffix for x in official_emails]
                return closest_match, official_domains

            elif contains_a_domain:
                return extracted_domains[0], extracted_domains

            elif self.knowledge_expansion:
                updated_emails, searching_time = self.expand_knowledge_base(potential_organization)
                if updated_emails:
                    return potential_organization, updated_emails

        return None, None

    def handle_internal_emails(self, relations: Set[str]) -> Tuple[bool, Optional[str]]:
        for relation in relations:
            matched_score, closest_match = self.find_closest_match(query=relation, value_index_db=self.internal_relation_index_db)
            if closest_match:  # internal
                return True, closest_match
        return False, None


    def __call__(self, identities: Set[str], actions: Set[str], relations: Set[str],
                 sender_domain: Optional[str], recipient_domains: List[str]) -> Tuple[Optional[bool], str, float]:

        total_time = 0
        # Check sender organization or relation
        if len(identities) == 0:
            Logger.spit('No predicted identity', caller_prefix=BrandMatcher._CallerPrefix, debug=True)
            return None, 'No Prediction', total_time

        identities = set(self.filter_duplicates(identities))
        sender_org_rel_combined = identities.union(relations)

        with Timer() as timer:
            matched_brand, official_email_domains = self.handle_external_emails(identities)
        total_time += timer.interval

        # Report mismatches or missing email specifications
        if official_email_domains and (sender_domain is None or sender_domain not in official_email_domains):
            if self.check_action:
                if len(actions) > 0:
                    Logger.spit(f'[!] Matched brand = {matched_brand}, inconsistent identity-address found, and contains at least 1 instruction', caller_prefix=BrandMatcher._CallerPrefix, debug=True)
                    return True, matched_brand, total_time
                else:
                    Logger.spit(f'Matched brand = {matched_brand}, inconsistent identity-address found, but does not contain any instruction => Benign', caller_prefix=BrandMatcher._CallerPrefix, debug=True)
                    return False, matched_brand, total_time
            else:
                Logger.spit(f'[!] Matched brand = {matched_brand}, inconsistent identity-address found', caller_prefix=BrandMatcher._CallerPrefix, debug=True)
                return True, matched_brand, total_time

        if official_email_domains:  # do not further check the internal relations
            Logger.spit('Consistent identity-address => Benign', caller_prefix=BrandMatcher._CallerPrefix, debug=True)
            return False, 'Consistent', total_time

        # Check internal relations

        with Timer() as timer:
            is_internal_emails, imitated_role = self.handle_internal_emails(sender_org_rel_combined)
        total_time += timer.interval

        if is_internal_emails and (sender_domain is None or sender_domain not in recipient_domains):
            if self.check_action:
                if len(actions) > 0:
                    Logger.spit(f'[!] Imitating an internal role {imitated_role} but from an external domain, and contains at least 1 instruction', caller_prefix=BrandMatcher._CallerPrefix, debug=True)
                    return True, 'Internal', total_time
                else:
                    Logger.spit(f'Imitating an internal role {imitated_role} but from an external domain, but does not contain any instruction => Benign', caller_prefix=BrandMatcher._CallerPrefix, debug=True)
                    return False, 'Internal', total_time
            else:
                Logger.spit(f'[!] Imitating an internal role {imitated_role} but from an external domain', caller_prefix=BrandMatcher._CallerPrefix, debug=True)
                return True, 'Internal', total_time

        Logger.spit('Does not match to any known identity or internal role => Benign', caller_prefix=BrandMatcher._CallerPrefix, debug=True)
        return False, 'No Matched Brand', total_time



if __name__ == '__main__':

    ## Build database
    model = CharacterBERT()

    brand_domain_map_path = './datasets/company_database.json'
    with open(brand_domain_map_path, 'r') as file:
        brand_domain_map = json.load(file)

    index_reps = np.empty((0, 768))
    batch_size = 128
    tags = []
    brand_name_list = list(brand_domain_map.keys())
    for i in tqdm(range(0, len(brand_name_list), batch_size)):
        batch = brand_name_list[i:min(i + batch_size, len(brand_name_list))]  # Get the next batch of brand names
        batch_embeddings = model(batch).cpu().numpy()  # Predict embeddings for the batch
        index_reps = np.concatenate((index_reps, batch_embeddings), axis=0)  # Append new embeddings
        tags.extend(batch)  # Collect tags

    np.save('./datasets/company_database_names.npy', np.asarray(tags))
    np.save('./datasets/company_database_reps.npy', index_reps)

    Internal_Relations = ['admin',
                             'mail admin',
                          'admin portal',
                             'administrator',
                             'mail team',
                             'mail service',
                          'mail server administrator',
                             'mail desk',
                             'webmail service',
                             'mail delivery system',
                             'employee',
                             'staff',
                             'colleague',
                             'e-mail',
                             'email',
                             'mailbox',
                             'server',
                             'faculty',
                             'manager',
                             'student',
                             'human resource',
                             'human resource team',
                             'HR team',
                             'Finance',
                             'it department',
                             'it support',
                             'it service support',
                             'Payroll',
                             'helpdesk',
                             'help desk',
                             'support desk',
                             'technical support',
                             'desk support',
                             'help center',
                             'support team',
                             'administration',
                             'administrative',
                             'supervisor',
                             'tech support',
                             'mail notification',
                             'it maintenance services',
                             'webmail panel',
                             'it report',
                             'itdesk',
                             'system support',
                             'tech team',
                             'mail security',
                             'webmaster',
                             'webmailservice',
                             'technical assistance',
                             'webmail team'
                             ]

    index_reps = np.empty((0, 768))
    tags = []
    for internal_relation in tqdm(Internal_Relations):
        embed = model([internal_relation.lower()]).cpu().numpy()
        index_reps = np.concatenate((index_reps, embed), axis=0)
        tags.append(internal_relation)

    np.save('./datasets/internal_relation_names.npy', np.asarray(tags))
    np.save('./datasets/internal_relation_reps.npy', index_reps)