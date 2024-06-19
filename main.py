# %%
from gensim.models import FastText
from gensim.utils import simple_preprocess
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

from functionCleanXML import suppEveryBalise
from functionImportantWords import get_document_vector, get_most_important_words
from functionDataFrame import readDataframe, createdCleanCSV
from functionKNN import KNN

# %%
df_cleaned = readDataframe()
createdCleanCSV(df_cleaned)

#%%
#KNN(df_cleaned)


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
most_important_words = get_most_important_words(combined_doc_vector, model)
print("Most important words:")
for word, similarity in most_important_words:
    print(f"{word}: {similarity}")
