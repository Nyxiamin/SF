import pandas as pd
from gensim import corpora
from gensim.utils import simple_preprocess
from collections import defaultdict
from tqdm import tqdm
import ast
from pathlib import Path


def preprocess_text(text):
    return simple_preprocess(text)


def create_bow(df):
    # Pre-process the descriptions
    texts = [preprocess_text(text) for text in tqdm(df['description'], desc="Preprocessing texts")]

    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(texts)

    # Filter out words that occur in less than 5 documents, or more than 50% of the documents
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    # Save the dictionary
    dictionary.save('df_bow_dict')

    # Create the bag-of-words representation of the documents
    corpus = [dictionary.doc2bow(text) for text in tqdm(texts, desc="Creating BoW representation")]

    # Create a DataFrame with the bag-of-words representation
    bows = []
    for doc_bow in tqdm(corpus, desc="Converting BoW to DataFrame"):
        bow_dict = defaultdict(int, doc_bow)
        bows.append(bow_dict)

    df['bow'] = bows
    return df, dictionary


def str_to_list(str_list):
    return ast.literal_eval(str_list)


def list_to_first_char_set(cpc_list):
    return set(item[0] for item in cpc_list)


def modify_df_bow():
    file_path = Path("../EFREI_LIPSTIP_50k_elements_EPO_bow.csv")
    if file_path.is_file():
        df_bow = pd.read_csv('../EFREI_LIPSTIP_50k_elements_EPO_bow.csv')

        df_bow.drop([['description'], ['claim']], axis=1, inplace=True)

        df_bow['CPC'] = df_bow['CPC'].apply(str_to_list)
        df_bow['CPC'] = df_bow['CPC'].apply(list_to_first_char_set)

        return df_bow
    return None
