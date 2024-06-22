import pandas as pd
from gensim import corpora
from gensim.utils import simple_preprocess
from collections import defaultdict
from tqdm import tqdm
from functionDataFrame import readDataframe


def preprocess_text(text):
    return simple_preprocess(text)


def create_bow(df):
    # Pre-process the descriptions
    texts = [preprocess_text(text) for text in tqdm(df['description'], desc="Preprocessing texts")]

    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(texts)

    # Filter out words that occur in less than 5 documents, or more than 50% of the documents
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    # Create the bag-of-words representation of the documents
    corpus = [dictionary.doc2bow(text) for text in tqdm(texts, desc="Creating BoW representation")]

    # Create a DataFrame with the bag-of-words representation
    bows = []
    for doc_bow in tqdm(corpus, desc="Converting BoW to DataFrame"):
        bow_dict = defaultdict(int, doc_bow)
        bows.append(bow_dict)

    df['bow'] = bows
    return df, dictionary

def modify_df_bow(df_bow):
    df_bow.drop([['description'], ['claim']], axis=1, inplace=True)
    