from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from gensim import corpora
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from gensim.utils import simple_preprocess
from tqdm import tqdm
import ast


# Définition des fonctions de conversion
def preprocess_text(text):
    return simple_preprocess(text)


def text_to_bow(text, dictionary):
    # Tokeniser le texte en mots
    words = text.split()

    # Créer un dictionnaire BoW vide
    bow = dictionary.doc2bow(words)

    # Convertir en vecteur BoW numpy
    bow_vector = np.zeros(len(dictionary))
    for word_id, word_count in bow:
        bow_vector[word_id] = word_count

    return bow_vector


def randomForest(df, filenames, codes_to_find):
    # Charger le dictionnaire existant
    dictionary = corpora.Dictionary.load('df_bow_dict')

    # Supprimer les lignes où la colonne 'bow' est vide
    df = df[df['bow'].map(bool)]

    # Vérifier si 'df' contient des données après suppression des lignes vides
    if df.empty:
        print("Le DataFrame 'df' est vide après suppression des lignes vides.")
        return

    # Diviser les données en ensembles d'entraînement et de test
    X = df['bow']  # Utilisation directe de la colonne 'bow' pour X
    y = df['CPC']

    if X.empty or y.empty:
        print("Les données d'entraînement ou les labels sont vides.")
        return

    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if X_train.empty or y_train.empty:
        print("Les ensembles d'entraînement ou les labels sont vides après la division.")
        return

    # Entraîner le modèle Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Entraîner le modèle
    rf_clf.fit(X_train, y_train)

    y_true_global = []
    y_pred_global = []

    for i, filename in enumerate(tqdm(filenames, desc="Processing files")):
        with open(filename, "r", encoding="utf-8") as f:
            textToAnalyze = f.read()

        # Préparer le texte à analyser
        bow_to_analyze = text_to_bow(textToAnalyze, dictionary)

        # Prédiction pour le texte à analyser
        y_pred = rf_clf.predict([bow_to_analyze])  # Utilisation directe de la prédiction sur le vecteur BoW

        # Ajouter les prédictions et les véritables codes à l'évaluation globale
        y_true = [1 if letter in codes_to_find[i] else 0 for letter in y.unique()]
        y_pred_global.extend(y_pred)
        y_true_global.extend(y_true)

        # Afficher les résultats pour chaque document de requête
        print(f"Document de requête : {filename}")
        print(f"CPC de la requête : {codes_to_find[i]}")
        print(f"CPC prédit : {y_pred[0]}")
        print()

    # Calculer la précision globale, le rappel et le score F1
    precision = precision_score(y_true_global, y_pred_global, average='macro')
    recall = recall_score(y_true_global, y_pred_global, average='macro')
    f1 = f1_score(y_true_global, y_pred_global, average='macro')

    print(f"Précision globale : {precision:.4f}")
    print(f"Rappel global : {recall:.4f}")
    print(f"Score F1 global : {f1:.4f}")


# Exemple d'utilisation de la fonction
# Assurez-vous d'avoir df, filenames et codes_to_find définis avant d'appeler cette fonction
df = pd.read_csv('../EFREI_LIPSTIP_50k_elements_EPO_bow.csv')
filenames = ['element_1.txt', 'element_2.txt', 'element_3.txt', 'element_4.txt', 'element_5.txt', 'element_6.txt',
             'element_7.txt', 'element_8.txt', 'element_9.txt', 'element_10.txt', 'element_11.txt', 'element_12.txt',
             'element_13.txt', 'element_14.txt', 'element_15.txt', 'element_16.txt', 'element_17.txt', 'element_18.txt',
             'element_19.txt', 'element_20.txt', 'element_21.txt', 'element_22.txt', 'element_23.txt', 'element_24.txt',
             'element_25.txt', 'element_26.txt', 'element_27.txt', 'element_28.txt', 'element_29.txt', 'element_30.txt',
             'element_31.txt', 'element_32.txt', 'element_33.txt', 'element_34.txt', 'element_35.txt', 'element_36.txt',
             'element_37.txt', 'element_38.txt', 'element_39.txt', 'element_40.txt', 'element_41.txt', 'element_42.txt',
             'element_43.txt', 'element_44.txt', 'element_45.txt', 'element_46.txt', 'element_47.txt', 'element_48.txt',
             'element_49.txt', 'element_50.txt']
codes_to_find = [{'C'}, {'C', 'A', 'G'}, {'H'}, {'H'}, {'G'}, {'B'}, {'F'}, {'A'}, {'H'}, {'H'}, {'B'}, {'C'}, {'A'},
                 {'F', 'B', 'G'}, {'H'}, {'F', 'B'}, {'C', 'B'}, {'Y', 'C', 'H'}, {'B'}, {'C', 'G'}, {'B'}, {'C', 'B'},
                 {'G'}, {'F'}, {'B', 'G'}, {'G'}, {'B'}, {'C'}, {'Y', 'H'}, {'F'}, {'H'}, {'F', 'B'}, {'G'}, {'G'},
                 {'H', 'G'}, {'B', 'D'}, {'H'}, {'C', 'B'}, {'B'}, {'B', 'H', 'G'}, {'H'}, {'H', 'G'}, {'F'},
                 {'Y', 'H'}, {'G'}, {'Y', 'C'}, {'B'}, {'C'}, {'Y', 'H', 'G'}, {'F'}]


# Charger des fonctions de conversion CPC et bow
def str_to_set(str_set):
    return ast.literal_eval(str_set)


def str_to_defaultdict(str_defaultdict):
    return ast.literal_eval(str_defaultdict)


# Afficher des informations sur le DataFrame initial
print(df.info())

# Appliquer les fonctions aux colonnes CPC et bow avec tqdm pour suivre la progression
tqdm.pandas(desc="Converting 'CPC' column")
df['CPC'] = df['CPC'].progress_apply(str_to_set)

tqdm.pandas(desc="Converting 'bow' column")
df['bow'] = df['bow'].progress_apply(str_to_defaultdict)

# Vérifier le type des colonnes après conversion
print(df.info())
print(df['bow'][0])  # Afficher un exemple pour vérifier la conversion

randomForest(df, filenames, codes_to_find)
