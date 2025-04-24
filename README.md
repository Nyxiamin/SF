# Patent Classification Project (EXPLAIN)

## 🌐 Overview

This project aims to assign **CPC (Cooperative Patent Classification) codes** to patent documents based on their textual content. We apply a document similarity approach using TF-IDF vectorization and cosine similarity, combined with a KNN-style method, and later reinforced with a Random Forest classifier.

This work was conducted as part of the EXPLAIN program in June-July 2024, by a **group of five students**.

---

## ⚡ Objectives

- Clean and process patent data stored in XML format.
- Extract meaningful text and convert it to a structured format.
- Represent documents with TF-IDF vectors.
- Compare documents using cosine similarity.
- Predict CPC code initials using the k most similar documents.
- Evaluate performance using precision, recall, and F1-score.
- Experiment with Random Forest as an alternative model.

---

## 🤖 Technologies & Libraries

- Python
- Gensim (TF-IDF, Dictionary, Cosine Similarity)
- scikit-learn (Random Forest, evaluation metrics)
- pandas
- Jupyter Notebook

---

## 🚀 Pipeline

1. **Cleaning XML files**: removal of unnecessary tags and extraction of relevant content.
2. **Text preprocessing**: tokenization, lowercasing, punctuation removal.
3. **Bag-of-Words & TF-IDF**: transform each document into a weighted vector.
4. **Cosine Similarity**: measure similarity between a query and the corpus.
5. **KNN Prediction**: retrieve k nearest documents and extract their CPC code initials.
6. **Evaluation**: compare predicted vs expected initials using classic ML metrics.

---

## 📤 Authors

Group project – EXPLAIN 2024

---

## 📅 Duration

**4 weeks – June to July 2024**
