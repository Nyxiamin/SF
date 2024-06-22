from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from gensim import corpora
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
import ast
from gensim.utils import simple_preprocess

def preprocess_text(text):
    return simple_preprocess(text)

def text_to_bow(text, dictionary):
    preprocessed_text = preprocess_text(text)
    return dictionary.doc2bow(preprocessed_text)

def randomForest(df, filenames, codes_to_find):
    # Charger le dictionnaire existant
    dictionary = corpora.Dictionary.load('dictionary')

    # Supprimer les lignes où la colonne 'bow' est vide
    df = df[df['bow'].map(bool)]

    # Convertir la colonne 'bow' en DataFrame
    bows = df['bow'].apply(lambda x: defaultdict(int, ast.literal_eval(x)))
    df_bow = pd.DataFrame(list(bows))
    df_bow = df_bow.fillna(0)

    # Ajouter la colonne "CPC" à df_bow
    df_bow['CPC'] = df['CPC'].values

    # Diviser les données en ensembles d'entraînement et de test
    X = df_bow.drop(columns=['CPC'])
    y = df_bow['CPC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Entraîner le modèle Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)

    y_true_global = []
    y_pred_global = []

    for i, filename in enumerate(filenames):
        with open(filename, "r", encoding="utf-8") as f:
            textToAnalyze = f.read()

        # Préparer le texte à analyser
        bow_to_analyze = text_to_bow(textToAnalyze, dictionary)
        bow_to_analyze_dict = defaultdict(int, bow_to_analyze)
        df_bow_to_analyze = pd.DataFrame([bow_to_analyze_dict]).fillna(0)

        # Ajouter des colonnes manquantes avec des zéros
        for col in X.columns:
            if col not in df_bow_to_analyze.columns:
                df_bow_to_analyze[col] = 0
        df_bow_to_analyze = df_bow_to_analyze[X.columns]

        # Prédiction pour le texte à analyser
        y_pred = rf_clf.predict(df_bow_to_analyze)

        # Ajouter les prédictions et les véritables codes à l'évaluation globale
        y_true = [1 if letter in codes_to_find[i] else 0 for letter in y.unique()]
        y_pred_global.extend(y_pred)
        y_true_global.extend(y_true)

        # Print results for each query document
        print(f"Query Document: {filename}")
        print(f"Query CPC: {codes_to_find[i]}")
        print(f"Predicted CPC: {y_pred[0]}")
        print()

    # Calculate global precision, recall, and F1 score
    precision = precision_score(y_true_global, y_pred_global, average='macro')
    recall = recall_score(y_true_global, y_pred_global, average='macro')
    f1 = f1_score(y_true_global, y_pred_global, average='macro')

    print(f"Global Precision: {precision:.4f}")
    print(f"Global Recall: {recall:.4f}")
    print(f"Global F1 Score: {f1:.4f}")

# Exemple d'utilisation de la fonction
df = pd.read_csv('../EFREI_LIPSTIP_50k_elements_EPO_bow.csv')
filenames = ['file1.txt', 'file2.txt', 'file3.txt']  # Remplacer par les noms de fichiers réels
codes_to_find = [['A61K'], ['B65D'], ['C07D']]  # Remplacer par les CPC réels correspondants

randomForest(df, filenames, codes_to_find)
