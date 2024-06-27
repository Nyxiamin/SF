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
from functionresult import functionresult

# %%
df_cleaned = readDataframe()
createdCleanCSV(df_cleaned)

#%%

# Fonction pour combiner les colonnes "description" et "claim"
def combine_columns(row):
    return f"{row['description']} {row['claim']}"

# Création de la colonne combinée en appliquant la fonction à chaque ligne
df_cleaned['combined'] = df_cleaned.apply(combine_columns, axis=1)

#%%
from gensim import corpora, models
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


tfidf = models.TfidfModel.load('KNNmodel\\tfidf_model')
dictionary = corpora.Dictionary.load('KNNmodel\\dictionary')

# Extract text and labels
texts = df_cleaned['combined']
corpus = [dictionary.doc2bow(text.split()) for text in texts]
tfidf_corpus = [tfidf[doc] for doc in corpus]

# Convert the TF-IDF corpus to a dense matrix
from gensim.matutils import corpus2dense
X = corpus2dense(tfidf_corpus, num_terms=len(dictionary)).T
y = df_cleaned['CPC']

# Rank features by importance using Chi-Squared test
chi2_scores, _ = chi2(X, y)
feature_ranking = chi2_scores.argsort()[::-1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize variables to store results
num_features = [5000, 3000, 2000, 1500, 1000, 800, 500, 200]
cv_scores = []

# Iterate over different numbers of features with tqdm progress bar
for n in tqdm(num_features, desc="Testing feature sets"):
    # Reduce the feature set
    selected_features = feature_ranking[:n]
    X_train_reduced = X_train[:, selected_features]
    X_test_reduced = X_test[:, selected_features]
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_reduced, y_train)
    
    # Evaluate using cross-validation
    scores = cross_val_score(model, X_train_reduced, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())


# Plot the results
plt.plot(num_features, cv_scores, marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Optimal Number of Features for Random Forest Model')
plt.show()

























#%%

print(df_cleaned.head(1))

predicted_cpc_codes, similar_documents, new_imp_words = functionresult(df_cleaned["combined"][0], df_cleaned)

print(predicted_cpc_codes)
for text in similar_documents:
    print(text)
print(new_imp_words)















#%%

print(df_cleaned["combined"][0])





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

import numpy as np
import spacy
import random
import ast
from gensim.models import Word2Vec, Phrases
from joblib import load
from sklearn.metrics import f1_score

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

    # Sort probabilities and get top 1 label
    top_label_indices = np.argsort(prediction_probabilities, axis=1)[:, ::-1][:, :1].flatten()  # Flattened top 1 label

    # Inverse transform to get original labels
    predicted_labels = le.inverse_transform(top_label_indices.reshape(-1, 1))  # Reshape to (1, 1)

    return predicted_labels.flatten().tolist()  # Convert to list for easier handling

# Function to calculate F1 score for multiple predictions
def calculate_f1_score(df_cleaned, predict_function, num_samples=100, verbose=False):
    y_true_global = []
    y_pred_global = []

    for i in range(num_samples):
        if verbose:
            print(f"Sample {i+1}/{num_samples}")
        
        random_id = random.randint(0, len(df_cleaned) - 1)
        new_imp_words = df_cleaned["important_words_tfidf_saved"].iloc[random_id]
        
        if not new_imp_words:
            continue
        
        # Predict CPC codes
        predicted_cpc_codes = predict_function(new_imp_words)
        
        # Collect true codes
        cpc_codes_str = df_cleaned['CPC'].iloc[random_id]
        if isinstance(cpc_codes_str, str):
            cpc_codes = ast.literal_eval(cpc_codes_str)
        else:
            cpc_codes = cpc_codes_str

        true_codes = set(code[0] for code in cpc_codes)  # Convert to set of strings

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
        
        if verbose:
            print(f"Predicted CPC Codes: {predicted_cpc_codes}")
            print(f"True codes: {true_codes}")
    
    # Calculate F1 score
    f1 = f1_score(y_true_global, y_pred_global)  # Sample-wise F1 score
    
    if verbose:
        print(f"F1 score: {f1}")
    return f1

# Load test and train indices
def load_indices():
    test_indices = np.load('test_indices.npy')
    train_indices = np.load('train_indices.npy')
    return test_indices, train_indices

# Example usage
test_indices, train_indices = load_indices()

# Filter valid indices for df
valid_test_indices = [idx for idx in test_indices if idx in df.index]
df_cleaned_test = df.iloc[valid_test_indices]

calculate_f1_score(df_cleaned_test, predict_cpc_codes, num_samples=100, verbose=True)


#%%


from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
from tqdm import tqdm
import pandas as pd
import re
from nltk.corpus import stopwords

# Assurez-vous d'avoir téléchargé les stopwords de NLTK avant d'exécuter ce code
# import nltk
# nltk.download('stopwords')

# Fonction pour nettoyer et prétraiter le texte
def preprocess_text(text):
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

# Fonction pour extraire les mots importants à partir du modèle TF-IDF chargé
def extract_important_words_tfidf_saved_model(text_data, tfidf_model, dictionary, top_n=50):
    important_words = []
    
    for text in tqdm(text_data, desc="Extracting important words"):
        # Prétraitement du texte
        processed_text = preprocess_text(text)
        
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

# Charger les modèles et le dictionnaire
tfidf = models.TfidfModel.load('tfidf_model')  # Assurez-vous que le fichier 'tfidf_model' existe
dictionary = corpora.Dictionary.load('dictionary')  # Assurez-vous que le fichier 'dictionary' existe

# Extraction des mots importants avec les modèles chargés
text_data = df_cleaned['combined'].tolist()
important_words_tfidf_saved = extract_important_words_tfidf_saved_model(text_data, tfidf, dictionary)

# Ajout des mots importants à votre dataframe
df_cleaned['important_words_tfidf_saved'] = important_words_tfidf_saved

# Supprimer les colonnes 'claim' et 'description'
df_cleaned.drop(['claim', 'description'], axis=1, inplace=True)

# Enregistrement du dataframe mis à jour dans un fichier CSV
df_cleaned.to_csv('../EFREI_LIPSTIP_50k_elements_EPO_important.csv', sep=',', index=False, encoding='utf-8')












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

