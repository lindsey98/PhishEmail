from lib.reference_db import CharacterBERT, BaseFaissIPRetriever
from lib.encoder import IdentityBert, Visualizer
import os
from tqdm import tqdm
import numpy as np
import json
from lib.utilities import Logger

class Config:

    Internal_Relations = [
        'admin',
        'mail admin',
        'webmail mail service team',
        'admin portal',
        'administrator',
        'administration',
        'administrative',
        'mail server administrator',
        'mail delivery system',
        'employee',
        'staff',
        'colleague',
        'colleagues',
        'student',
        'human resource team',
        'HR team',
        'HR',
        'human resources department',
        'Finance team',
        'IT department',
        'IT support',
        'IT service support',
        'Payroll',
        'IT maintenance services',
        'IT report',
        'IT Desk',
        'internal company',
        # the following internal roles are observed in field study
        'billing support',
        'helpdesk',
        'IT helpdesk',
        'Postmaster',
        'IMAP',
        'Internal company',
        'administration department',
        'mail inc.',
        'email security center',
        'financial management department',
        'general affairs department',
        'finance department',
        'human resources department',
        'email system network record',
        'ministry of information',
        'email system management notification',
        'trusted server',
        'oa data migration and upgrade',
        'information management center',
        'the mail system',
        'support IT',
        'mailbox administrator',
        'enterprise mailbox',
        'lending department',
        'system administrator',
        'accounts payable team',
        'management department',
        'service information center',
        'e-mail mailbox system',
        "security administrator",
        "account administrator",
        "maintenance department",
        "IT teams",
        "campus email",
        "network and data center",
        "post office system administrator",
        "mygov",
        "security center",
        "tomHRM",
        "home depot department",
        "mail inc."
    ]

    # IDENTITY_MODEL_CHECKPOINT = os.getenv("IDENTITY_MODEL_CHECKPOINT", "./checkpoints/identity_adversarial_training")
    IDENTITY_MODEL_CHECKPOINT = os.getenv("IDENTITY_MODEL_CHECKPOINT", "./checkpoints/identity-model")

    MATCHING_MODEL_CHECKPOINT = os.getenv("MATCHING_MODEL_CHECKPOINT", "./checkpoints/characterbert-typos-st-adv")

    REF_IDENTITY_REPS  = os.getenv("REF_IDENTITY_REPS", "./checkpoints/company_database_knowphish_v2_reps.npy")
    REF_IDENTITY_NAMES = os.getenv("REF_IDENTITY_NAMES", "./checkpoints/company_database_knowphish_v2_names.npy")
    REF_IDENTITY_MAP   = os.getenv("REF_IDENTITY_MAP", "./checkpoints/company_database_knowphish_v2.json")

    # REF_IDENTITY_REPS  = os.getenv("REF_IDENTITY_REPS", "./checkpoints/company_database_reps_field_study.npy")
    # REF_IDENTITY_NAMES = os.getenv("REF_IDENTITY_NAMES", "./checkpoints/company_database_names_field_study.npy")
    # REF_IDENTITY_MAP   = os.getenv("REF_IDENTITY_MAP", "./checkpoints/company_database_field_study_new.json")

    # REF_IDENTITY_REPS  = os.getenv("REF_IDENTITY_REPS", "./checkpoints/company_database_spearbot_reps.npy")
    # REF_IDENTITY_NAMES = os.getenv("REF_IDENTITY_NAMES", "./checkpoints/company_database_spearbot_names.npy")
    # REF_IDENTITY_MAP   = os.getenv("REF_IDENTITY_MAP", "./checkpoints/company_database_spearbot.json")

    # REF_IDENTITY_REPS  = os.getenv("REF_IDENTITY_REPS", "./checkpoints/company_database_ephishgen_reps.npy")
    # REF_IDENTITY_NAMES = os.getenv("REF_IDENTITY_NAMES", "./checkpoints/company_database_ephishgen_names.npy")
    # REF_IDENTITY_MAP   = os.getenv("REF_IDENTITY_MAP", "./checkpoints/company_database_ephishgen.json")

    # REF_IDENTITY_REPS  = os.getenv("REF_IDENTITY_REPS", "./checkpoints/company_database_prompted_contextual_reps.npy")
    # REF_IDENTITY_NAMES = os.getenv("REF_IDENTITY_NAMES", "./checkpoints/company_database_prompted_contextual_names.npy")
    # REF_IDENTITY_MAP   = os.getenv("REF_IDENTITY_MAP", "./checkpoints/company_database_prompted_contextual.json")

    REF_RELATION_REPS = os.getenv("REF_RELATION_REPS", "./checkpoints/internal_relation_reps.npy")
    REF_RELATION_NAMES = os.getenv("REF_RELATION_NAMES", "./checkpoints/internal_relation_names.npy")

    identity_model = IdentityBert(IDENTITY_MODEL_CHECKPOINT)
    visualizer_model = Visualizer(IDENTITY_MODEL_CHECKPOINT)
    matching_model = CharacterBERT(MATCHING_MODEL_CHECKPOINT)

    Logger.spit('Loaded the identity recognition model and identity matching model into memory', caller_prefix="Main", debug=True)

    try:
        with open(REF_IDENTITY_MAP, 'r') as file:
            brand_domain_map = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        Logger.spit(f'Error loading identity map: {e}', caller_prefix="Main", warning=True)
        raise
    ref_embed_list = np.load(REF_IDENTITY_REPS) if os.path.exists(REF_IDENTITY_REPS) else None

    index_reps = np.empty((0, 768))
    batch_size = 128
    tags = []
    brand_name_list = list(brand_domain_map.keys())
    for i in tqdm(range(0, len(brand_name_list), batch_size), desc="Caching the knowledge base embeddings"):
        batch = brand_name_list[i:min(i + batch_size, len(brand_name_list))]  # Get the next batch of brand names
        batch_embeddings, _ = matching_model(batch)
        batch_embeddings = batch_embeddings.cpu().numpy()  # Predict embeddings for the batch
        index_reps = np.concatenate((index_reps, batch_embeddings), axis=0)  # Append new embeddings
        tags.extend(batch)  # Collect tags
    np.save(REF_IDENTITY_NAMES, np.asarray(tags))
    np.save(REF_IDENTITY_REPS, index_reps)

    tags = [x.lower() for x in Internal_Relations]
    embed, _ = matching_model(tags)
    embed = embed.cpu().numpy()
    np.save(REF_RELATION_NAMES, np.asarray(tags))
    np.save(REF_RELATION_REPS, embed)

    Logger.spit('Loaded the identity knowledge base into memory', caller_prefix="Main", debug=True)
    ref_embed_list = np.load(REF_IDENTITY_REPS) if REF_IDENTITY_REPS else None
    ref_tag_list = np.load(REF_IDENTITY_NAMES).tolist()
    brand_index_db = BaseFaissIPRetriever(init_reps=ref_embed_list, tags=ref_tag_list, embed_model=matching_model)
    brand_domain_map_path = REF_IDENTITY_MAP

    relation_embed_list = np.load(REF_RELATION_REPS) if REF_RELATION_REPS else None
    relation_tag_list = np.load(REF_RELATION_NAMES).tolist()
    internal_relation_index_db = BaseFaissIPRetriever(init_reps=relation_embed_list, tags=relation_tag_list, embed_model=matching_model)

    thre = 0.95

    try:
        # with open("whitelist_senders.txt", 'r') as f:
        #     whitelist_senders = set([x.strip() for x in f.readlines() if x.strip()])
        whitelist_senders = []
    except FileNotFoundError:
        Logger.spit('Whitelist file not found, using empty whitelist', caller_prefix="Main", warning=True)
        whitelist_senders = set()  # ignore the paper submission, online course, and the internal senders
    forbidden_subject_prefix = ["re:", "fwd:", "fw:", "回复:", "转发:", "reply:"]

    knowledge_expansion_on = True