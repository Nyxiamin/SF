# %%
from gensim.models import FastText
from gensim.utils import simple_preprocess
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import random

from functionCleanXML import suppEveryBalise
from functionImportantWords import get_document_vector, get_most_important_words
from functionDataFrame import readDataframe, createdCleanCSV
from functionKNN import KNN
from functionTXT import transform_to_txt, lire_fichier_txt
from regression_models import logistic_regression_classification
# %%
df_cleaned = readDataframe()
createdCleanCSV(df_cleaned)

#%%
df_cleaned.head()    

#%%
filenames = []
codes_to_find = []
# Loop pour créer 50 fichiers textes aléatoires contenant des descriptions (nécessaire pour le KNN suivant, mais attention ça crée beaucoup de fichiers)
for i in range(1, 51):
    random_id = random.randint(0, 49999)
    filename = f"element_{i}.txt"
    codes = transform_to_txt(df_cleaned, filename, random_id)
    print(f"Fichier {filename} créé pour l'ID {random_id}")
    filenames.append(filename)
    codes_to_find.append(codes)

print(filenames)
print(codes_to_find)

#%%
KNN(df_cleaned, filenames, codes_to_find)

#%%
logistic_regression_classification(df_cleaned, filenames, codes_to_find)
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
# %%
