from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from gensim import corpora, models, similarities
import pandas as pd


def randomForest(df, textToAnalyze):
    tfidf = models.TfidfModel.load('tfidf_model')
    dictionary = corpora.Dictionary.load('dictionary')

    textToAnalyze = textToAnalyze.lower()
    listToAnalyze = textToAnalyze.split()

    corpusToAnalyze = dictionary.doc2bow(listToAnalyze)
    vectorToAnalyze = tfidf[corpusToAnalyze]

    tfidf_dict = dict(vectorToAnalyze)
    df_tfidf = pd.DataFrame([tfidf_dict])
    df_tfidf = df_tfidf.fillna(0)

    X = df.drop('CPC', axis=1)
    y = df['CPC']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_clf.fit(X_train, y_train)

    y_pred = rf_clf.predict(df_tfidf)
    return y_pred[0]
