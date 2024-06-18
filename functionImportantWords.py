
#%%
from gensim.models import FastText
from gensim.utils import simple_preprocess
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
#%%

# Calculate document vector by averaging word vectors
def get_document_vector(doc, model):
    vectors = [model.wv[word] for word in doc if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Find the most important words
def get_most_important_words(doc_vector, model,top_n=10):
    all_words = list(model.wv.index_to_key)
    similarities = {}
    for word in all_words:
        word_vector = model.wv[word].reshape(1, -1)
        similarity = cosine_similarity(doc_vector, word_vector)[0][0]
        similarities[word] = similarity
    most_important_words = Counter(similarities).most_common(top_n)
    return most_important_words
