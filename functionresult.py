import numpy as np
import spacy
import random
import ast
from gensim.models import Word2Vec, Phrases
from joblib import load
from sklearn.metrics import f1_score
import re
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.utils import simple_preprocess
from functionKNN import KNN
from functionDataFrame import readDataframe
from tqdm import tqdm


# Charger les modèles une seule fois au démarrage
tfidf = models.TfidfModel.load('KNNmodel\\tfidf_model')
dictionary = corpora.Dictionary.load('KNNmodel\\dictionary')
word2vec_model = Word2Vec.load('Randomforest\\word2vec_model')
bigram = Phrases.load('Randomforest\\bigrams_model')
best_clf = load('Randomforest\\random_forest_model.joblib')
le = load('Randomforest\\label_encoder.pkl')
nlp = spacy.load('en_core_web_sm')

# Fonction pour nettoyer et prétraiter le texte
def preprocess_text_important_words(text):
    # Convertir en minuscules
    text = text.lower()
    # Enlever les balises HTML
    text = re.sub(r'<.*?>', ' ', text)
    # Enlever les caractères spéciaux et les chiffres
    text = re.sub(r'\W+|\d+', ' ', text)
    # Tokenisation et suppression des stopwords
    stop_words = set(stopwords.words('english'))
    tokens = simple_preprocess(text)
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Fonction pour extraire les mots importants à partir du modèle TF-IDF chargé pour un seul texte
def extract_important_words_from_text(text, tfidf_model, dictionary, top_n=500):
    # Prétraitement du texte
    processed_text = preprocess_text_important_words(text)
    
    # Conversion en vecteur TF-IDF
    vec_bow = dictionary.doc2bow(processed_text)
    vec_tfidf = tfidf_model[vec_bow]
    
    # Tri des mots par importance TF-IDF
    sorted_tfidf = sorted(vec_tfidf, key=lambda x: x[1], reverse=True)
    
    # Sélection des top_n mots les plus importants
    top_words = []
    for word_index, score in sorted_tfidf[:top_n]:
        top_words.append(dictionary[word_index])
    
    stop_words = set(stopwords.words('english'))
    clean_words = []
    for word in top_words:
        if word not in stop_words and word != " ":
            clean_words.append(word)

    return ' '.join(top_words)


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

    # Sort probabilities and get top 1 label
    top_label_indices = np.argsort(prediction_probabilities, axis=1)[:, ::-1][:, :3].flatten()  # Flattened top 1 label

    # Inverse transform to get original labels
    predicted_labels = le.inverse_transform(top_label_indices.reshape(-1, 1))  # Reshape to (1, 1)

    return predicted_labels.flatten().tolist()  # Convert to list for easier handling


def functionresult(text,df_cleaned):
    new_imp_words = extract_important_words_from_text(text, tfidf, dictionary)

    # Utiliser tqdm pour suivre la progression de la prédiction des codes CPC
    with tqdm(total=1, desc="Predicting CPC codes") as pbar:
        predicted_cpc_codes = predict_cpc_codes(new_imp_words)
        pbar.update(1)

    similar_documents = []
    with tqdm(total=1, desc="Finding similar documents") as pbar:
        similar_document, similar_code, similar_pourcentage = KNN(df_cleaned, text)
        for i in range(len(similar_code)):
            similar_code[i] = ast.literal_eval(similar_code[i])
            similar_code[i] = list(set(code[0] for code in similar_code[i]))
        for i in range(len(similar_document)):
            similar_documents.append([similar_document[i],similar_code[i],similar_pourcentage[i]])
        pbar.update(1)
    for i in range(len(similar_documents)):
        # Enlever les balises HTML
        similar_documents[i][0] = re.sub(r'<.*?>', ' ', similar_documents[i][0])
    
    markdown_words = extract_important_words_from_text(text, tfidf, dictionary,20)

    return predicted_cpc_codes, similar_documents, markdown_words