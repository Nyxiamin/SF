# %%
import pandas as pd
from gensim.models import FastText
from gensim.utils import simple_preprocess
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

from functionCleanXML import suppEveryBalise
from functionImportantWords import get_document_vector, get_most_important_words
# %%
df = pd.read_csv("../EFREI - LIPSTIP - 50k elements EPO.csv")

# %%
print(df.info())
print(df.head())

# %%
print(df)

# %%
print((df.isnull().sum() / len(df)) * 100)
# il n'y a pas de valeur null

# %%
columns = ["Numéro d'application", "Date d'application", "Numero de publication", "date de publication", "IPC"]
df_cleaned = df.drop(columns, axis=1)
print(df_cleaned.head())

# %%
# df_cleaned.to_csv('../EFREI_LIPSTIP_50k_elements_EPO_clean.csv', sep=',', index=False, encoding='utf-8')

# %%
text = df_cleaned['description'][0]
print("\n\nDescription de la première ligne clean:\n", suppEveryBalise(text))

#%%

text_data = [suppEveryBalise(text) for text in df_cleaned['description'].iloc[:3]]
# Preprocess the text data
processed_text = [simple_preprocess(doc) for doc in text_data]

# Train FastText model
model = FastText(sentences=processed_text, vector_size=100, window=3, min_count=1, sg=1, epochs=10)

# Combine all documents to find the most important words
combined_doc = [word for doc in processed_text for word in doc]
combined_doc_vector = get_document_vector(combined_doc, model).reshape(1, -1)

# Find the most important words
most_important_words = get_most_important_words(combined_doc_vector,model)
print("Most important words:")
for word, similarity in most_important_words:
    print(f"{word}: {similarity}")

#%%
text_data