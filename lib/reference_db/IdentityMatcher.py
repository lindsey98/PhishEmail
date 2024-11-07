
from .CharacterBert import CharacterBertModel, CharacterIndexer
from transformers import BertTokenizer
import os
import torch.nn.functional as F
import torch
import faiss
from tqdm import tqdm
import numpy as np
import json
from typing import List, Tuple, Union, Optional, Set, Any
import re
from tldextract import tldextract
ext = tldextract.TLDExtract(cache_dir='./lib/reference_db')
from ..utilities.gpt_utils import chat_completion
from ..utilities.data_utils import clean_string, DomainUtils
from ..utilities import Logger, Timer
from typing import Union, List, Set
import difflib

# fixme: several things to adjust:
#  1. Whether to check contains domain or not? => FP, do not check
#  2. What's the threshold used to group the reported identities into clusters? => higher would produce more clusters, lower would produce bigger clusters
#  3. What's the threshold used for internal keywords matching?

class CharacterBERT:
    _CallerPrefix = "CharacterBert"

    def __init__(self, model_id: str,
                 return_cls: bool=True,
                 do_l2_norm: bool=True) -> None:

        self.tokenizer = BertTokenizer.from_pretrained(model_id)
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

    def __init__(self, tags: List[str],
                 init_reps: Optional[np.ndarray]=None,
                 embed_model: Optional[CharacterBERT]=None,
                 index_type: str = "Flat"):
        self.tags = tags

        if init_reps is None:
            assert embed_model is not None
            self.embed_model = embed_model
            Logger.spit("No cached representations, build index from scratch ..", caller_prefix=BaseFaissIPRetriever._CallerPrefix, debug=True)
            init_reps = self.build_reps_from_scratch(self.tags)

        assert len(tags) == init_reps.shape[0], "Number of tags must match the number of representations."
        # Create index based on the specified index type
        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(init_reps.shape[1])
        elif index_type == "IVF":
            nlist = 100  # Number of clusters
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatIP(init_reps.shape[1]), init_reps.shape[1], nlist)
            self.index.train(init_reps)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(init_reps.shape[1], 32)  # 32 is the default value for the number of neighbors
            self.index.train(init_reps)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        self.index.add(init_reps)  # Add vectors to the index
        Logger.spit("Index base of size {} with dimension {} is trained".format(init_reps.shape[0], init_reps.shape[1]),
                    caller_prefix=BaseFaissIPRetriever._CallerPrefix,
                    debug=True)

    def build_reps_from_scratch(self, tag_list):
        index_reps = np.empty((0, 768))
        batch_size = 128
        for i in range(0, len(tag_list), batch_size):
            batch = tag_list[i:min(i + batch_size, len(tag_list))]  # Get the next batch of brand names
            batch_preds = self.embed_model(batch)
            batch_embeddings, time = batch_preds[0].cpu().numpy(), batch_preds[1]  # Predict embeddings for the batch
            index_reps = np.concatenate((index_reps, batch_embeddings), axis=0)  # Append new embeddings
        return index_reps

    def search(self, q_reps: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        with Timer() as timer:
            scores, indices = self.index.search(q_reps, k)
        tag_results = np.array(self.tags)[indices]
        return scores, indices, tag_results, timer.interval

    def add(self, p_reps: Optional[np.ndarray], p_tags: List[str]) -> None:
        if p_reps is not None:
            self.index.add(p_reps)
        else:
            p_reps = self.build_reps_from_scratch(p_tags)
            self.index.add(p_reps)
        assert len(p_tags) == p_reps.shape[0], "Number of tags must match the number of representations."
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


class IdentityMatcher:
    _CallerPrefix = "IdentityMatcher"

    def __init__(self,
                brand_index_db: Optional[BaseFaissIPRetriever],
                internal_relation_index_db: Optional[BaseFaissIPRetriever],
                embed_model: CharacterBERT,
                brand_domain_map_path: Optional[str],
                knowledge_base_expansion: bool=False,
                gpt_client: Optional[Any]=None, gpt_assistant: Optional[Any]=None,
                check_action: bool=True,
                threshold: float=0.95,
                relax_match: bool=False):

        self.brand_index_db = brand_index_db
        self.internal_relation_index_db = internal_relation_index_db
        self.brand_domain_map_path = brand_domain_map_path
        if brand_domain_map_path:
            with open(self.brand_domain_map_path, 'r') as file:
                self.brand_domain_map = json.load(file)

        self.knowledge_expansion = knowledge_base_expansion
        self.check_action = check_action
        self.embed_model = embed_model
        self.threshold = threshold
        self.gpt_client = gpt_client
        self.gpt_assistant = gpt_assistant
        self.relax_match = relax_match
        # if relax match is set, the matcher will try to match all reported identities.
        # Otherwise, the matcher will only match the most common, most confident identity.


    @staticmethod
    def filter_duplicates(strings: Union[List[str], Set[str]], threshold: float=0.5) -> List[List[str]]:
        clusters = []
        for string in strings:
            # Remove "from:" or any case variation with optional spaces
            string = re.sub(r'(?i)from\s*:?', '', string)
            # Remove leading "##"
            string = re.sub(r'^\s*#+', '', string)
            string = string.strip()
            if len(string) <= 1:
                continue

            # Check similarity with already filtered strings
            similar_found = False
            for i, f in enumerate(clusters):
                if difflib.SequenceMatcher(None, string, f[0]).ratio() > threshold: # group into a cluster
                    similar_found = True
                    # Add the string to the existing cluster
                    clusters[i].append(string)
                    break
            if not similar_found:
                clusters.append([string])

        return clusters

    def find_closest_match(self, query: str, value_index_db: BaseFaissIPRetriever,
                           thre: float) -> Tuple[Optional[float], Optional[str]]:

        query_embed, embedding_time = self.embed_model([query.lower()])
        query_embed = query_embed.cpu().numpy().astype('float32') # use float32 to speed up?
        score, index, tag, searching_time = value_index_db.search(query_embed, 1) # 1-NN

        score = score.tolist()[0][0]
        tag = tag.tolist()[0][0]

        if len(query) <= 3: # require exact match if query is too short, because I observe a FP case where 'pay' is matched to 'pay now'
            thre = 1

        if score >= thre:
            return score, tag
        else:
            return None, None

    def expand_knowledge_base(self, queried_identity: str) -> Tuple[Optional[List[str]], float]:
        if self.gpt_client is None:
            from openai import OpenAI
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            openai.proxy = os.getenv("http_proxy")  # proxy
            self.gpt_client = OpenAI()

        with Timer() as timer:
            search_email = chat_completion(client=self.gpt_client,
                            context="Given a brand, output its official email address. "
                                    "If the input is not a brand/organization name, output ''. "
                                    "Directly give the email address with no additional explanation. "
                                    "If the you are not sure about the official email, do not respond. "
                                    "If there are multiple possible emails, output them all as a list [email_1, email_2, ...].",
                            query=queried_identity)

        searching_time = timer.interval
        Logger.spit(f'Searching {queried_identity} in GPT, \t'
                    f'Return {search_email}, \t'
                    f'Searching time = {searching_time}',
                    caller_prefix=IdentityMatcher._CallerPrefix, debug=True)

        email_regex = re.compile(
            r'^\[?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(,\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})*\]?$')

        if email_regex.fullmatch(search_email):  # if the return are valid email addresses
            emails = search_email.strip('[]').split(',')
            official_emails = list(set([email.split('@')[1].strip() for email in emails]))

            # update index DB
            queried_identity_embed, embedding_time = self.embed_model([queried_identity.lower()])
            queried_identity_embed = queried_identity_embed.cpu().numpy()
            self.brand_index_db.add(queried_identity_embed, [queried_identity.lower()])
            # fixme: it doesnt save the index db

            # update the reference list
            self.brand_domain_map[queried_identity.lower()] = official_emails
            with open(self.brand_domain_map_path, 'w') as file:
                json.dump(self.brand_domain_map, file, indent=4)

            return official_emails, searching_time

        return None, searching_time

    def handle_external_emails(self, identities: Set[str]) -> Tuple[Union[None, Set[str]], Union[None, Set[str]]]:
        closest_match_set = set()
        official_domains_set = set()
        for potential_organization in identities:
            if potential_organization.startswith("##"): # noisy prediction
                continue

            # Match against known brands or directly extract domains
            matched_score, closest_match = self.find_closest_match(query=potential_organization,
                                                                   value_index_db=self.brand_index_db,
                                                                   thre=self.threshold)
            # contains_a_domain, extracted_domains = DomainUtils.contains_domain(potential_organization)
            # todo: I remove the contains domain checking, because this may introduce FP (pte.ltd or matches to a domain but the official email address doesnt use this domain)

            if closest_match:
                official_emails = self.brand_domain_map[closest_match]
                official_domains = DomainUtils.extract_domain_set(official_emails)
                closest_match_set.add(closest_match)
                official_domains_set = official_domains_set.union(official_domains)

            # elif contains_a_domain:
            #     return list(extracted_domains)[0], extracted_domains

            elif self.knowledge_expansion:
                updated_emails, searching_time = self.expand_knowledge_base(potential_organization)
                if updated_emails:
                    closest_match_set.add(potential_organization)
                    official_domains_set = official_domains_set.union(updated_emails)

        return closest_match_set, official_domains_set

    def handle_internal_emails(self, relations: Set[str]) -> Tuple[bool, Optional[str]]:
        for relation in relations:
            matched_score, closest_match = self.find_closest_match(query=relation,
                                                                   value_index_db=self.internal_relation_index_db,
                                                                   thre=self.threshold - 0.05) # fixme: relax the threshold a bit
            if closest_match:  # internal
                return True, closest_match
        return False, None

    def __call__(self, identities: List[str], actions: Set[str], relations: List[str],
                 sender_domains: Set[str], recipient_domains: Set[str],
                 top_k_identities: int = 1) -> Tuple[Optional[bool], Union[Set[str], str], float]:

        total_time = 0
        # Check sender organization or relation
        if len(identities) == 0 and len(relations) == 0:
            Logger.spit('No predicted identity', caller_prefix=IdentityMatcher._CallerPrefix, debug=True)
            return None, 'No Prediction', total_time

        identities = [clean_string(x) for x in identities]
        relations = [clean_string(x) for x in relations]
        combined_set = identities + relations  # Identities first, then relations

        if self.relax_match == False: # if relax_match is False, only try to match the top-1 identity with the highest confidence. May introduce FNs when the NER didnt rank the true identity as top-1.
            identities = self.filter_duplicates(identities)  # Returns a list in desired order
            first_cluster_identities = set()
            for this_set in identities:
                first_cluster_identities = first_cluster_identities.union(this_set)
                clean_this_set_representative_str = re.sub(r'[@!:;.]', '', this_set[0]).strip()
                if len(clean_this_set_representative_str) > 3:
                    break

            with Timer() as timer:
                matched_brand_set, official_email_domains_set = self.handle_external_emails(first_cluster_identities)
            total_time += timer.interval
        else: # if relax_match is True, match to ANY recognized identity. This has risk of FP, e.g. the true identity is outside knowledge base, but some other identities are inside.
            with Timer() as timer:
                matched_brand_set, official_email_domains_set = self.handle_external_emails(identities)
            total_time += timer.interval

        # Report mismatches or missing email specifications
        if official_email_domains_set and (not DomainUtils.domain_set_overlap(sender_domains, official_email_domains_set)):
            if self.check_action:
                if len(actions) > 0:
                    Logger.spit(f'[!Phish] Matched brand = "{matched_brand_set}", inconsistent identity-address found with sender address as "{sender_domains}", and contains at least 1 instruction',
                                caller_prefix=IdentityMatcher._CallerPrefix, debug=True)
                    return True, matched_brand_set, total_time
                else:
                    Logger.spit(f'Matched brand = "{matched_brand_set}", inconsistent identity-address found, but does not contain any instruction => Benign',
                                caller_prefix=IdentityMatcher._CallerPrefix, debug=True)
                    return False, matched_brand_set, total_time
            else:
                Logger.spit(f'[!Phish] Matched brand = "{matched_brand_set}", inconsistent identity-address found with sender address as "{sender_domains}"',
                            caller_prefix=IdentityMatcher._CallerPrefix, debug=True)
                return True, matched_brand_set, total_time

        if official_email_domains_set:  # do not further check the internal relations
            Logger.spit('Consistent identity-address => Benign', caller_prefix=IdentityMatcher._CallerPrefix, debug=True)
            return False, 'Consistent', total_time

        # Check internal relations

        with Timer() as timer:
            is_internal_emails, imitated_role = self.handle_internal_emails(combined_set)
        total_time += timer.interval

        if is_internal_emails and recipient_domains and (not DomainUtils.domain_set_overlap(sender_domains, recipient_domains)):
            if self.check_action:
                if len(actions) > 0:
                    Logger.spit(f'[!Phish] Imitating an internal role "{imitated_role}" but from an external domain with sender address as "{sender_domains}", and contains at least 1 instruction',
                                caller_prefix=IdentityMatcher._CallerPrefix, debug=True)
                    return True, 'Internal', total_time
                else:
                    Logger.spit(f'Imitating an internal role "{imitated_role}" but from an external domain, but does not contain any instruction => Benign',
                                caller_prefix=IdentityMatcher._CallerPrefix, debug=True)
                    return False, 'Internal', total_time
            else:
                Logger.spit(f'[!Phish] Imitating an internal role "{imitated_role}" but from an external domain with sender address as "{sender_domains}"',
                            caller_prefix=IdentityMatcher._CallerPrefix, debug=True)
                return True, 'Internal', total_time

        if is_internal_emails:  # do not further check the internal relations
            Logger.spit('Consistent sender-recipient-address => Benign', caller_prefix=IdentityMatcher._CallerPrefix, debug=True)
            return False, 'Consistent', total_time

        Logger.spit('Does not match to any known identity or internal role => Benign',
                    caller_prefix=IdentityMatcher._CallerPrefix, debug=True)
        return False, 'No Matched Brand', total_time


if __name__ == '__main__':
    '''Build database'''
    model = CharacterBERT("./checkpoints/characterbert-typos-st")

    brand_domain_map_path = './checkpoints/company_database_knowphish.json'
    with open(brand_domain_map_path, 'r') as file:
        brand_domain_map = json.load(file)

    index_reps = np.empty((0, 768))
    batch_size = 128
    tags = []
    brand_name_list = list(brand_domain_map.keys())
    for i in tqdm(range(0, len(brand_name_list), batch_size)):
        batch = brand_name_list[i:min(i + batch_size, len(brand_name_list))]  # Get the next batch of brand names
        batch_embeddings, _ = model(batch)
        batch_embeddings = batch_embeddings.cpu().numpy()  # Predict embeddings for the batch
        index_reps = np.concatenate((index_reps, batch_embeddings), axis=0)  # Append new embeddings
        tags.extend(batch)  # Collect tags

    np.save('./checkpoints/company_database_names.npy', np.asarray(tags))
    np.save('./checkpoints/company_database_reps.npy', index_reps)

    Internal_Relations = [
                          'admin',
                          'mail admin',
                          'admin portal',
                          'administrator',
                          'administration',
                          'administrative',

                          'mail team',
                          'mail service',
                          'mail server administrator',
                          'mail desk',
                          'webmail service',
                          'mail delivery system',
                          # 'e-mail',
                          # 'email',
                          'mailbox',
                          'mail support',
                          'email storage',
                          'mail notification',
                          'webmail panel',
                          'mail security',
                          'webmaster',
                          'webmailservice',
                          'webmail team',
                          'webmail mail service team'

                          'employee',
                          'staff',
                          'colleague',
                          'faculty',
                          # 'manager',
                          'student',
                          # 'supervisor',

                          # 'human resource',
                          'human resource team',
                          'HR team',
                          'HR',
                          'human resources department',

                          'Finance team'
                          'IT department',
                          'IT support',
                          'IT service support',
                          'Payroll',
                          'IT maintenance services',
                          'IT report',
                          'IT Desk',

                          'internal company',
                          ]

    tags = [x.lower() for x in Internal_Relations]
    embed, _ = model(tags)
    embed = embed.cpu().numpy()

    np.save('./checkpoints/internal_relation_names.npy', np.asarray(tags))
    np.save('./checkpoints/internal_relation_reps.npy', embed)

