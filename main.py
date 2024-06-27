# %%
from gensim.models import FastText
from gensim.utils import simple_preprocess
import numpy as np
import random
from functionCleanXML import suppEveryBalise
from functionImportantWords import get_document_vector, get_most_important_words
from functionDataFrame import readDataframe, createdCleanCSV
from functionKNN import KNN
from functionTXT import transform_to_txt
from functiontest import functiontest
import numpy as np
import spacy
from gensim.models import Word2Vec, Phrases
from sklearn.ensemble import RandomForestClassifier
from joblib import load
import random
import ast
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd

# %%
df_cleaned = readDataframe()
createdCleanCSV(df_cleaned)

#%%

df = pd.read_csv("../EFREI_LIPSTIP_50k_elements_EPO_important.csv")
print(df.head())




#%%
#==== Fonction de création de fichiers ====

filenames = []
codes_to_find = []
# Loop pour créer 100 fichiers textes aléatoires contenant des descriptions (nécessaire pour le KNN suivant, mais attention ça crée beaucoup de fichiers)
for i in range(1, 101):
    random_id = random.randint(0, 49999)
    filename = f"element_{i}.txt"
    codes = transform_to_txt(df_cleaned, filename, random_id)
    print(f"Fichier {filename} créé pour l'ID {random_id}")
    filenames.append(filename)
    codes_to_find.append(codes)

#%%
KNN(df_cleaned, filenames, codes_to_find)

#%%
# Appeler functiontest avec cette copie indépendante
functiontest(df)


#%%

# Load the trained models
word2vec_model = Word2Vec.load('word2vec_model')
bigram = Phrases.load('bigrams_model')
best_clf = load('random_forest_model.joblib')
le = load('label_encoder.pkl')  # Load the LabelEncoder

# Load SpaCy model for English
nlp = spacy.load('en_core_web_sm')

# Function for text preprocessing
def preprocess(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

# Function to preprocess a new element
def preprocess_new_element(combined):
    processed_text = preprocess(combined)
    bigrams_text = bigram[processed_text]
    vector = np.mean([word2vec_model.wv[word] for word in bigrams_text if word in word2vec_model.wv], axis=0)
    return vector

# Function to predict CPC codes for a new element
def predict_cpc_codes(important_words):
    # Preprocess new element
    vector = preprocess_new_element(important_words).reshape(1, -1)
    
    # Predict CPC codes
    prediction_probabilities = best_clf.predict_proba(vector)

    # Sort probabilities and get top 3 labels
    top_label_indices = np.argsort(prediction_probabilities, axis=1)[:, ::-1][:, :3].flatten()  # Flattened top 3 labels

    # Inverse transform to get original labels
    predicted_labels = le.inverse_transform(top_label_indices.reshape(-1, 1))  # Reshape to (3, 1)

    return predicted_labels.flatten().tolist()  # Convert to list for easier handling


# Function to calculate F1 score for multiple predictions
def calculate_f1_score(df_cleaned, predict_function, num_samples=100):
    y_true_global = []
    y_pred_global = []

    for i in range(num_samples):
        print(i)
        random_id = random.randint(0, len(df_cleaned) - 1)
        new_imp_words = df_cleaned["important_words_tfidf_saved"][random_id]
        
        # Predict CPC codes
        predicted_cpc_codes = predict_function(new_imp_words)
        
        # Collect true codes
        cpc_codes_str = df_cleaned['CPC'][random_id]
        if isinstance(cpc_codes_str, str):
            cpc_codes = ast.literal_eval(cpc_codes_str)
        else:
            cpc_codes = cpc_codes_str
        cpc_codes = ast.literal_eval(cpc_codes_str)
        true_codes = set(code[0] for code in cpc_codes)  # Convert to list of strings
        

        y_true = []
        y_pred = []
        
        for letter in true_codes:
            y_true.append(1)
            y_pred.append(1 if letter in predicted_cpc_codes else 0)
        
        for letter in predicted_cpc_codes:
            if letter not in true_codes:
                y_true.append(0)
                y_pred.append(1)

        # Append results to global lists
        y_true_global.extend(y_true)
        y_pred_global.extend(y_pred)
        
        print(f"Predicted CPC Codes: {predicted_cpc_codes}")
        print(f"True codes: {true_codes}")
    
    # Calculate F1 score
    f1 = f1_score(y_true_global, y_pred_global) # Sample-wise F1 score
    
    print(f"F1 score: {f1}")

# Example usage
calculate_f1_score(df, predict_cpc_codes, num_samples=100)

#%%


from gensim import corpora, models, similarities

# Fonction pour extraire les mots importants à partir du modèle TF-IDF chargé
def extract_important_words_tfidf_saved_model(text_data, tfidf_model, similarity_index, dictionary, top_n=5):
    important_words = []
    
    for text in tqdm(text_data, desc="Extracting important words"):
        # Prétraitement du texte
        processed_text = simple_preprocess(text)
        
        # Conversion en vecteur TF-IDF
        vec_bow = dictionary.doc2bow(processed_text)
        vec_tfidf = tfidf_model[vec_bow]
        
        # Tri des mots par importance TF-IDF
        sorted_tfidf = sorted(vec_tfidf, key=lambda x: x[1], reverse=True)
        
        # Sélection des top_n mots les plus importants
        top_words = []
        for word_index, score in sorted_tfidf[:top_n]:
            top_words.append(dictionary[word_index])
        
        important_words.append(' '.join(top_words))
    
    return important_words

tfidf = models.TfidfModel.load('tfidf_model')  # Ensure 'tfidf_model' file exists
index = similarities.SparseMatrixSimilarity.load('similarity_index')  # Ensure 'similarity_index' file exists
dictionary = corpora.Dictionary.load('dictionary')

# Extraction des mots importants avec les modèles chargés
text_data = df_cleaned['combined'].tolist()
important_words_tfidf_saved = extract_important_words_tfidf_saved_model(text_data, tfidf, index, dictionary)

# Ajout des mots importants à votre dataframe
df_cleaned['important_words_tfidf_saved'] = important_words_tfidf_saved
# Supprimer les colonnes 'claim' et 'description'
df_cleaned.drop(['claim', 'description'], axis=1, inplace=True)

# Enregistrement du dataframe mis à jour dans un fichier CSV
df_cleaned.to_csv('../EFREI_LIPSTIP_50k_elements_EPO_important.csv', sep=',', index=False, encoding='utf-8')

#%%

# Fonction pour combiner les colonnes "description" et "claim"
def combine_columns(row):
    return f"{row['description']} {row['claim']}"

# Création de la colonne combinée en appliquant la fonction à chaque ligne
df_cleaned['combined'] = df_cleaned.apply(combine_columns, axis=1)

# Affichage des premières lignes pour vérifier
print(df_cleaned[['description', 'claim', 'combined']].head())


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