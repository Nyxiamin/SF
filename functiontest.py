import spacy
from tqdm import tqdm
from gensim.models import Phrases, Word2Vec
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from sklearn.model_selection import train_test_split
import ast

def preprocess(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

def preprocess_long_text(text):
    nlp = spacy.load('en_core_web_sm')
    max_chunk_length = 1000000  # Maximum chunk length
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    processed_chunks = [preprocess(chunk) for chunk in chunks]
    return sum(processed_chunks, [])

def apply_bigrams(doc):
    bigram = Phrases.load('bigrams_model')
    return bigram[doc]

def document_vector(doc, word2vec_model):
    # Calculate the mean vector for the document
    vectors = [word2vec_model.wv[word] for word in doc if word in word2vec_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        # Return a zero vector if no words in the doc are in the model's vocabulary
        return np.zeros(word2vec_model.vector_size)

def functiontest(df_cleaned):
    df_cleaned['CPC'] = df_cleaned['CPC'].apply(ast.literal_eval)

    nlp = spacy.load('en_core_web_sm')

    tqdm.pandas(desc="Preprocessing Text")
    df_cleaned['combined_processed'] = df_cleaned['important_words_tfidf_saved'].progress_apply(preprocess_long_text)

    sentences = df_cleaned['combined_processed'].tolist()
    bigram = Phrases(sentences, min_count=5, threshold=100)
    bigram.save('bigrams_model')

    tqdm.pandas(desc="Applying Bigrams")
    df_cleaned['combined_processed_bigrams'] = df_cleaned['combined_processed'].progress_apply(apply_bigrams)

    sentences = df_cleaned['combined_processed_bigrams'].tolist()
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    word2vec_model.save('word2vec_model')

    tqdm.pandas(desc="Vectorizing Documents")
    df_cleaned['vector'] = df_cleaned['combined_processed_bigrams'].progress_apply(lambda doc: document_vector(doc, word2vec_model))

    # Debugging step: Check dimensions of each vector
    vector_lengths = df_cleaned['vector'].apply(lambda x: len(x))
    print("Vector lengths:", vector_lengths.unique())

    # Ensure all vectors have the correct length
    expected_vector_size = word2vec_model.vector_size
    df_cleaned['vector'] = df_cleaned['vector'].apply(lambda x: x if len(x) == expected_vector_size else np.zeros(expected_vector_size))

    df_cleaned['CPC_first_letter'] = df_cleaned['CPC'].apply(lambda x: x[0][0])

    le = LabelEncoder()
    y = le.fit_transform(df_cleaned['CPC_first_letter'])
    dump(le, 'label_encoder.pkl')

    X = np.vstack(df_cleaned['vector'].values)

    # Retain indices to track training and testing split
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices, test_size=0.2, random_state=42)

    # Save the indices
    np.save('train_indices.npy', train_indices)
    np.save('test_indices.npy', test_indices)

    clf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_macro')
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_

    y_pred = best_clf.predict(X_test)

    if len(np.unique(y_test)) != len(np.unique(y_pred)):
        print("Warning: Number of unique classes in y_test and y_pred do not match.")

    print(classification_report(y_test, y_pred, target_names=le.classes_, labels=np.unique(y_test)))

    dump(best_clf, 'random_forest_model.joblib')
    print("Finished! Model saved!!")