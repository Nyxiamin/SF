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

# %%
df_cleaned = readDataframe()
createdCleanCSV(df_cleaned)
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
functiontest(df_cleaned)


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
def preprocess_new_element(description, claim):
    combined = description + " " + claim
    processed_text = preprocess(combined)
    bigrams_text = bigram[processed_text]
    vector = np.mean([word2vec_model.wv[word] for word in bigrams_text if word in word2vec_model.wv], axis=0)
    return vector

# Function to predict CPC codes for a new element
def predict_cpc_codes(description, claim):
    # Preprocess new element
    vector = preprocess_new_element(description, claim).reshape(1, -1)
    
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
        new_description = df_cleaned["description"][random_id]
        new_claim = df_cleaned["claim"][random_id]
        
        # Predict CPC codes
        predicted_cpc_codes = predict_function(new_description, new_claim)
        
        # Collect true codes
        cpc_codes_str = df_cleaned['CPC'][random_id]
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
calculate_f1_score(df_cleaned, predict_cpc_codes, num_samples=100)

#%%






def get_document_vector(document, model):
    # Fonction pour obtenir le vecteur du document
    doc_vector = [model.wv[word] for word in document if word in model.wv]
    return sum(doc_vector) / len(doc_vector) if doc_vector else np.zeros(model.vector_size)

def get_most_important_words(document_vector, model, top_n=5):
    # Fonction pour obtenir les mots les plus importants
    word_similarities = {}
    for word in model.wv.index_to_key:
        word_vector = model.wv[word].reshape(1, -1)
        similarity = cosine_similarity(document_vector, word_vector)[0][0]
        word_similarities[word] = similarity
    most_important_words = sorted(word_similarities.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return most_important_words

# Combinaison des colonnes 'description' et 'claim' avec tqdm pour suivre l'avancement
tqdm.pandas(desc="Combinaison des colonnes")
df_cleaned['combined'] = df_cleaned.progress_apply(lambda row: f"{suppEveryBalise(row['description'])} {suppEveryBalise(row['claim'])}", axis=1)

# Transformation du texte en tokens avec tqdm
text_data = df_cleaned['combined'].progress_apply(simple_preprocess)

# Entraînement du modèle FastText avec tqdm
model = FastText(vector_size=100, window=3, min_count=1, sg=1)
model.build_vocab(sentences=text_data)
model.train(sentences=tqdm(text_data, desc="Entraînement du modèle"), total_examples=len(text_data), epochs=10)

# Fonction pour extraire les mots importants pour chaque description combinée
def extract_important_words(text, model):
    processed_text = simple_preprocess(text)
    document_vector = get_document_vector(processed_text, model).reshape(1, -1)
    important_words = get_most_important_words(document_vector, model, top_n=2)
    return ' '.join([word for word, similarity in important_words])

# Application de la fonction sur chaque description combinée avec tqdm
tqdm.pandas(desc="Extraction des mots importants")
df_cleaned['important_words'] = df_cleaned['combined'].progress_apply(lambda x: extract_important_words(x, model))

# Affichage des premières lignes du dataframe avec les mots importants
print(df_cleaned[['combined', 'important_words']].head())














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