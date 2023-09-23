import os, sys, email,re
import numpy as np
import pandas as pd
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('whitegrid')
import wordcloud

# Network analysis
import networkx as nx

# NLP
from nltk.tokenize.regexp import RegexpTokenizer

from subprocess import check_output

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import LatentDirichletAllocation

import gensim
from gensim import corpora
import nltk
nltk.set_proxy('http://127.0.0.1:7890')
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.colors as mcolors
import networkx as nx
import random
from gensim.models.coherencemodel import CoherenceModel

stop_words = list(STOP_WORDS)
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'cc', 'email', 'bcc', 'subject']) # add some email stopwords
def extract_data(feature, df):
    column= []
    for row in df:
        e = email.message_from_string(row)
        column.append(e.get(feature))
    return column

def get_email_body(data):
    column = []
    col_len = []
    for msg in data:
        e = email.message_from_string(msg)
        column.append(e.get_payload())
        col_len.append(len(e.get_payload().split()))
    return column, col_len

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]

def get_address(from_field, strict):
    regex = re.compile('[\w\.-]+@[\w\.-]+\.\w+')
    match = re.findall(regex, from_field)
    if match:
        return match[0]
    else:
        if strict:
            return None
        else:
            return from_field

def normalize_address(address):
    if not address:
        return None
    elif "</O=ENRON" in address:
        names = re.findall(r'([a-zA-Z]+)', address)
        names_concatenated = ''.join(names)
        normalized_address = f"{names_concatenated}@enron.com"
        return normalized_address.lower()
    elif '@' in address:
        return address.lower()
    else:
        return None


def jaccard_similarity(topic_1, topic_2):
    intersection = set(topic_1).intersection(set(topic_2))
    union = set(topic_1).union(set(topic_2))

    return float(len(intersection)) / float(len(union))

if __name__ == '__main__':
    '''data cleaning'''
    # df = pd.read_csv('./datasets/enron_mail_2015/emails.csv')
    # print(df.shape)
    # eng_stopwords = set(stopwords.words('english'))
    # #
    # df['Date'] = extract_data("Date", df['message'])
    # df['Subject'] = extract_data("Subject", df['message'])
    # df['X-From'] = extract_data("X-From", df['message'])
    # df['X-To'] = extract_data("X-To", df['message'])
    # df['X-Folder'] = extract_data("X-Folder", df['message'])
    # df['body'], df['body_token_len'] = get_email_body(df['message'])
    # df.to_csv('./datasets/enron_mail_2015/emails_processed.csv')
    #
    # # Remove punctuation, dropna
    # df = pd.read_csv('./datasets/enron_mail_2015/emails_processed.csv')
    # df = (df[(~df['body'].str.contains('Forwarded')) &
    #          (~df['body'].str.contains('Original Message')) &
    #          (~df['Subject'].fillna('').str.contains('Re:|Fw:', case=False))])
    # regex = "\n[Tt][Oo]:[^\n]*|\n[CC][cc]:[^\n]*|\n[Bb][Cc][Cc]:[^\n]*|\n[Ss]ubject:[^\n]*"
    # df['body'] = df['body'].apply(lambda x: re.sub(regex, '', x))
    # df['body_text_processed'] = df['body'].apply(lambda x: re.sub('[^\w \n]', '', x.lower()))
    # df.dropna(subset=['body_text_processed', 'X-From', 'X-To'], how='any', inplace=True)
    # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # df.to_csv('./datasets/enron_mail_2015/emails_processed.csv')
    # print(df.describe())
    # exit()

    # df = pd.read_csv('./datasets/enron_mail_2015/emails_processed.csv')
    # df['From'] = df['X-From'].apply(get_address, strict=True).apply(normalize_address)
    # df['To'] = df['X-To'].apply(get_address, strict=False)
    # df.dropna(subset=['From'], how='any', inplace=True)
    # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # df.to_csv('./datasets/enron_mail_2015/emails_processed_clean.csv')
    # print(df)
    # exit()
    #
    '''Topic analysis'''
    # df = pd.read_csv('./datasets/enron_mail_2015/emails_processed_clean.csv')
    # # remove stop words
    # data = df.body_text_processed.values.tolist()
    # data_words = list(sent_to_words(data))
    # data_words = remove_stopwords(data_words)
    # print(data_words[:2][1][:30])
    # # document to word matrix
    # id2word = corpora.Dictionary(data_words)
    # texts = data_words
    # corpus = [id2word.doc2bow(text) for text in texts]
    # print(corpus[2][:30])
    # #
    # # LDA
    # num_topics = list(range(20)[1:])
    # num_keywords = 15
    #
    # LDA_models = {}
    # LDA_topics = {}
    # for i in num_topics:
    #     print(f'Iteration {i}')
    #     LDA_models[i] = gensim.models.LdaModel(corpus=corpus,
    #                                            id2word=id2word,
    #                                            num_topics=i,
    #                                            chunksize=len(corpus),
    #                                            passes=20,
    #                                            random_state=214,
    #                                            iterations=10)
    #
    #     shown_topics = LDA_models[i].show_topics(num_topics=i,
    #                                              num_words=num_keywords,
    #                                              formatted=False)
    #     LDA_topics[i] = [[word[0] for word in topic[1]] for topic in shown_topics]
    #
    # LDA_stability = {}
    # for i in range(0, len(num_topics) - 1):
    #     jaccard_sims = []
    #     for t1, topic1 in enumerate(LDA_topics[num_topics[i]]):  # pylint: disable=unused-variable
    #         sims = []
    #         for t2, topic2 in enumerate(LDA_topics[num_topics[i + 1]]):  # pylint: disable=unused-variable
    #             sims.append(jaccard_similarity(topic1, topic2))
    #         jaccard_sims.append(sims)
    #     LDA_stability[num_topics[i]] = jaccard_sims
    # mean_stabilities = [np.array(LDA_stability[i]).mean() for i in num_topics[:-1]]
    #
    # coherences = [CoherenceModel(model=LDA_models[i], texts=texts, dictionary=id2word, coherence='c_v').get_coherence() \
    #               for i in num_topics[:-1]]
    #
    # coh_sta_diffs = [coherences[i] - mean_stabilities[i] for i in
    #                  range(num_keywords)[:-1]]  # limit topic numbers to the number of keywords
    # coh_sta_max = max(coh_sta_diffs)  # lower value means higher coherence since we used u_mass
    #
    # coh_sta_max_idxs = [i for i, j in enumerate(coh_sta_diffs) if j == coh_sta_max]
    # ideal_topic_num_index = coh_sta_max_idxs[0]  # choose less topics in case there's more than one max
    # ideal_topic_num = num_topics[ideal_topic_num_index]
    # plt.figure(figsize=(20, 10))
    # ax = sns.lineplot(x=num_topics[:-1], y=mean_stabilities, label='Average Topic Overlap')
    # ax = sns.lineplot(x=num_topics[:-1], y=coherences, label='Topic Coherence')
    #
    # ax.axvline(x=ideal_topic_num - 1, label='Ideal Number of Topics', color='black')
    # ax.axvspan(xmin=ideal_topic_num - 2, xmax=ideal_topic_num, alpha=0.5, facecolor='grey')
    #
    # y_max = max(max(mean_stabilities), max(coherences)) + (0.10 * max(max(mean_stabilities), max(coherences)))
    # ax.set_ylim([-y_max, y_max])
    # ax.set_xlim([1, num_topics[-1] - 1])
    #
    # ax.axes.set_title('Model Metrics per Number of Topics', fontsize=25)
    # ax.set_ylabel('Metric Level', fontsize=20)
    # ax.set_xlabel('Number of Topics', fontsize=20)
    # plt.legend(fontsize=20)
    # plt.savefig('./datasets/enron_mail_select_numtopic.png')
    # plt.close()

    # num_topics = 5
    # lda_model = gensim.models.LdaModel(corpus=corpus,
    #                                    id2word=id2word,
    #                                    num_topics=num_topics,
    #                                    random_state=214,
    #                                    chunksize=len(corpus),
    #                                    passes=20,
    #                                    iterations=10)

    # Print the Keyword
    # print(lda_model.print_topics())
    # [(0,
    #   '0.012*"size" + 0.011*"td" + 0.010*"width" + 0.008*"tr" + 0.007*"error" + 0.006*"font" + 0.006*"week" + 0.006*"key" + 0.005*"free" + 0.005*"random"'),
    #  (1,
    #   '0.013*"said" + 0.012*"power" + 0.010*"energy" + 0.007*"state" + 0.007*"california" + 0.005*"enron" + 0.005*"electricity" + 0.004*"gas" + 0.004*"market" + 0.004*"new"'),
    #  (2,
    #   '0.011*"enron" + 0.010*"image" + 0.006*"information" + 0.005*"time" + 0.005*"questions" + 0.005*"new" + 0.005*"know" + 0.004*"attached" + 0.004*"contact" + 0.004*"click"'),
    #  (3,
    #   '0.009*"company" + 0.009*"new" + 0.008*"million" + 0.006*"business" + 0.005*"enron" + 0.005*"services" + 0.005*"said" + 0.005*"management" + 0.004*"gas" + 0.004*"capital"'),
    #  (4,
    #   '0.015*"pm" + 0.010*"final" + 0.010*"data" + 0.009*"scheduled" + 0.008*"outages" + 0.008*"sat" + 0.007*"london" + 0.007*"schedule" + 0.006*"ct" + 0.006*"pt"')]

    # doc_lda = lda_model[corpus]
    # get_document_topics = [lda_model.get_document_topics(item) for item in corpus]
    # df['topics'] = get_document_topics
    # df['top_topic'] = df['topics'].apply(lambda x: max(x, key=lambda y: y[1]))
    # df['top_index'] = df['topics'].apply(lambda x: max(x, key=lambda y: y[1])[0])
    # df['topic'] = df['top_topic'].apply(lambda x: lda_model.print_topic(x[0]))
    # df.to_csv('./datasets/enron_mail_2015/emails_with_topics.csv', index=False, mode='w')
    # exit()

    '''Sender-Receiver network analysis'''

    # Create the graph
    df = pd.read_csv('./datasets/enron_mail_2015/emails_processed_clean.csv')
    count_enron = df['From'].apply(lambda x: 'enron' in x).sum()
    count_not_enron = len(df) - count_enron
    print(len(df))
    print(count_enron/len(df), count_not_enron/len(df))
    subdf = df.sample(500, random_state=214)
    G = nx.from_pandas_edgelist(subdf, 'From', 'To')

    # Drawing code
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=.1)
    node_colors = ['green' if 'enron' in node else 'red' for node in G.nodes()]
    nx.draw_networkx(G, pos, node_size=100, node_color=node_colors, with_labels=False, edge_color='blue')
    # Save the figure
    plt.savefig('./datasets/enron_mail_network.png')
    # Close the figure to free up resources
    plt.close()

    # Count the frequency of each sender and receiver
    sender_count = df['From'].value_counts()
    receiver_count = df['To'].value_counts()

    # Get the top 10 senders and receivers
    top_10_senders = sender_count.nlargest(3)
    top_10_receivers = receiver_count.nlargest(3)

    # Plotting the frequency of top 10 senders and receivers
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Top 10 Sender frequency
    top_10_senders.plot(kind='bar', ax=axes[0], color='blue')
    axes[0].set_title('Top 10 Sender Frequency')
    axes[0].set_xlabel('Email Address')
    axes[0].set_ylabel('Count')

    # Top 10 Receiver frequency
    top_10_receivers.plot(kind='bar', ax=axes[1], color='green')
    axes[1].set_title('Top 10 Receiver Frequency')
    axes[1].set_xlabel('Email Address')
    axes[1].set_ylabel('Count')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
    axes[0].tick_params(axis='x', labelsize=20)
    axes[1].tick_params(axis='x', labelsize=20)
    plt.tight_layout()
    plt.savefig('./datasets/enron_mail_distribution.png')
    plt.close()

    pete_davis = (df[(df['From'].str.contains('pete.davis')) | (df['To'].str.contains('pete.davis'))])
    pete_davis = pete_davis.loc[:, ~pete_davis.columns.str.contains('^Unnamed')]
    print(pete_davis)

    '''Subject analysis'''
    df = pd.read_csv('./datasets/enron_mail_2015/emails_processed_clean.csv')
    # most common subjects
    # 1. Common Subjects: Identify the most common subjects or themes in the emails.
    common_subjects = df['Subject'].value_counts().nlargest(5)
    print("Most Common Subjects:")
    print(common_subjects)

    # 2. Keyword Analysis: Use text mining techniques to find frequently occurring words or phrases in the subject lines.
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Subject'].dropna())
    word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    sum_words = word_freq.sum(axis=0)
    common_keywords = sum_words.sort_values(ascending=False).nlargest(5)
    print("\nMost Common Keywords:")
    print(common_keywords)

    # 3. Contain potential phishy words: List of potential phishy words
    phishy_words = ['urgent', 'verify', 'account', 'suspension', 'login', 'password', 'credentials']

    # Function to flag subjects with phishy words
    def detect_phishy_words(subject):
        if isinstance(subject, float):
            return False
        subject = subject.lower().split()
        return any(word in subject for word in phishy_words)

    # Apply the function to the 'Subject' column
    df['Is_Phishy'] = df['Subject'].apply(detect_phishy_words)
    # Display emails flagged as phishy
    phishy_emails = df[df['Is_Phishy']] # 173 of emails
    phishy_emails = phishy_emails[~phishy_emails['From'].str.contains('enron')]
    print(phishy_emails)
    phishy_emails.to_csv('./datasets/enron_email_potential_phishing.csv')
