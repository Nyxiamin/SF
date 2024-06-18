
def KNN(df_cleaned): 
    from gensim import corpora, models, similarities
    # Load the TF-IDF model and similarity index
    tfidf = models.TfidfModel.load('tfidf_model')  # Ensure 'tfidf_model' file exists
    index = similarities.SparseMatrixSimilarity.load('similarity_index')  # Ensure 'similarity_index' file exists
    dictionary = corpora.Dictionary.load('dictionary')

    # Define your query document
    query_document = 'cracks position perpendicular manual prevented positioning the glass mechanism play'.split()

    # Convert the query document to BoW format using the loaded dictionary
    query_bow = dictionary.doc2bow(query_document)

    # Transform the query BoW vector with TF-IDF
    query_tfidf = tfidf[query_bow]

    # Calculate cosine similarity between the query and all documents in the corpus
    sims = index[query_tfidf]

    # Sort the similarities and retrieve the top ten most similar documents
    top_ten = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:10]

    # Print the top ten most similar documents
    for rank, (document_number, score) in enumerate(top_ten, 1):
        print(f"Rank {rank}: Document Number {document_number}, Similarity Score {score:.4f}")
        print(df_cleaned['CPC'][document_number])  # Print or process the relevant content from df_cleaned
        print()  # Print a blank line for separation
