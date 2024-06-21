
def functiontest(df_cleaned):
    from joblib import dump
    import spacy
    from tqdm import tqdm
    from gensim.models import Phrases, Word2Vec
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import classification_report
    from transformers import BertTokenizer, BertModel
    import ast

    # Load the SpaCy model for English
    nlp = spacy.load('en_core_web_sm')

    # Function for text preprocessing
    def preprocess(text):
        doc = nlp(text)
        return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

    # Combine 'description' and 'claim' columns
    df_cleaned['combined'] = df_cleaned['description'] + " " + df_cleaned['claim']
    
    # Apply preprocessing with progress bar
    tqdm.pandas(desc="Preprocessing Text")

        # Fonction pour découper les textes longs en morceaux
    def preprocess_long_text(text):
        max_chunk_length = 1000000  # Limite maximale de longueur par morceau
        chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        processed_chunks = [preprocess(chunk) for chunk in chunks]
        return sum(processed_chunks, [])  # Concaténer les listes de résultats

    
    # Appliquer le prétraitement aux textes longs
    df_cleaned['combined_processed'] = df_cleaned['combined'].progress_apply(preprocess_long_text)
    
    # Build the bigrams model
    sentences = df_cleaned['combined_processed'].tolist()
    bigram = Phrases(sentences, min_count=5, threshold=100)

    # Apply the bigrams model with progress bar
    def apply_bigrams(doc):
        return bigram[doc]

    tqdm.pandas(desc="Applying Bigrams")
    df_cleaned['combined_processed_bigrams'] = df_cleaned['combined_processed'].progress_apply(apply_bigrams)

    # Train Word2Vec on the processed texts
    sentences = df_cleaned['combined_processed_bigrams'].tolist()
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Function to convert document to vector using Word2Vec
    def document_vector(doc):
        return np.mean([word2vec_model.wv[word] for word in doc if word in word2vec_model.wv], axis=0)

    # Apply document vector conversion with progress bar
    tqdm.pandas(desc="Vectorizing Documents")
    # Fonction pour découper les textes longs en morceaux
    def preprocess_long_text(text):
        max_chunk_length = 1000000  # Limite maximale de longueur par morceau
        chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        processed_chunks = [preprocess(chunk) for chunk in chunks]
        return sum(processed_chunks, [])  # Concaténer les listes de résultats

    # Appliquer le prétraitement aux textes longs
    df_cleaned['combined_processed'] = df_cleaned['combined'].progress_apply(preprocess_long_text)

    # Prepare the data
    X = np.vstack(df_cleaned['vector'].values)

    # Function to convert string representation of list to actual list
    def convert_to_list(string):
        return ast.literal_eval(string)

    # Apply the conversion to the 'CPC' column with progress bar
    tqdm.pandas(desc="Converting CPC Strings to Lists")
    df_cleaned['CPC'] = df_cleaned['CPC'].progress_apply(convert_to_list)

    # Now y will contain actual lists
    y = df_cleaned['CPC']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the Random Forest model
    clf = RandomForestClassifier(random_state=42)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_macro')
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_

    # Evaluate the best model
    y_pred = best_clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Function to get BERT embeddings
    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    # Apply BERT embeddings to texts with progress bar
    tqdm.pandas(desc="Getting BERT Embeddings")
    df_cleaned['bert_embedding'] = df_cleaned['combined'].progress_apply(get_bert_embedding)

    # Convert BERT embeddings to the correct shape
    X_bert = np.vstack(df_cleaned['bert_embedding'].values)

    # Ensure the shapes are compatible and concatenate Word2Vec and BERT embeddings
    X_combined = np.hstack((X, X_bert))

    # Split the combined data into training and testing sets
    X_train_combined, X_test_combined, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    # Train and evaluate the Random Forest model on the combined embeddings
    grid_search.fit(X_train_combined, y_train)
    best_clf = grid_search.best_estimator_

    #Save the model
    dump(best_clf, 'random_forest_model.joblib')

    # Evaluate the best model on combined embeddings
    y_pred = best_clf.predict(X_test_combined)
    print(classification_report(y_test, y_pred))








