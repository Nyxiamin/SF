from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from gensim import corpora, models
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
import ast
import numpy as np


def randomForest(df_cleaned, filenames, codes_to_find):
    # Créer une liste de documents et des étiquettes associées
    documents = []
    CPC = []
    for i, filename in enumerate(filenames):
        with open(filename, "r", encoding="utf-8") as f:
            documents.append(f.read())
        CPC.append(codes_to_find[i])

    # Vectorisation des documents avec TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_cleaned['description'])

    # Création des étiquettes binaires pour chaque lettre unique dans les CPC codes
    all_letters = sorted(set(letter for letters in CPC for letter in letters))
    y = []
    for CPC_set in CPC:
        y.append([1 if letter in CPC_set else 0 for letter in all_letters])
    y = np.array(y)

    # Entraînement du modèle de régression logistique pour chaque lettre
    models = []
    for i in range(len(all_letters)):
        y_i = y[:, i]
        model = LogisticRegression()
        model.fit(X, y_i)
        models.append(model)

    y_true_global = []
    y_pred_global = []

    for i, doc in enumerate(documents):
        X_test = vectorizer.transform([doc])
        y_true = y[i]
        y_pred = [model.predict(X_test)[0] for model in models]

        y_true_global.extend(y_true)
        y_pred_global.extend(y_pred)

        # Print results for each query document
        print(f"Query Document: {filenames[i]}")
        print(f"Query First Letters: {codes_to_find[i]}")
        similar_first_letters = {all_letters[idx] for idx, pred in enumerate(y_pred) if pred == 1}
        print(f"Similar First Letters: {similar_first_letters}")
        print()

    # Calculate global precision, recall, and F1 score
    precision = precision_score(y_true_global, y_pred_global)
    recall = recall_score(y_true_global, y_pred_global)
    f1 = f1_score(y_true_global, y_pred_global)

    print(f"Global Precision: {precision:.4f}")
    print(f"Global Recall: {recall:.4f}")
    print(f"Global F1 Score: {f1:.4f}")