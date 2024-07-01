def KNN(df_cleaned, text): 
    from gensim import corpora, models, similarities, utils
    
    # Load the TF-IDF model and similarity index
    tfidf = models.TfidfModel.load('KNNmodel\\tfidf_model')  # Ensure 'tfidf_model' file exists
    index = similarities.SparseMatrixSimilarity.load('KNNmodel\\similarity_index')  # Ensure 'similarity_index' file exists
    dictionary = corpora.Dictionary.load('KNNmodel\\dictionary')
    
    # Define your query document
    query_document = text  
    
    # Tokenize the query document
    query_tokens = utils.simple_preprocess(query_document)
    
    # Convert the query document to BoW format using the loaded dictionary
    query_bow = dictionary.doc2bow(query_tokens)
    
    # Transform the query BoW vector with TF-IDF
    query_tfidf = tfidf[query_bow]
    
    # Calculate cosine similarity between the query and all documents in the corpus
    sims = index[query_tfidf]
    
    # Sort the similarities and retrieve the top three most similar documents
    top = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[1:4]
    
    # Retrieve the descriptions of the top three most similar documents
    top_descriptions = [df_cleaned['description'][document_number] for document_number, _ in top]
    top_code = [df_cleaned['CPC'][document_number] for document_number, _ in top]
    top_similarities = [similarity for _, similarity in top]
    
    # Convert similarities to percentage
    top_similarities_percentage = [similarity * 100 for similarity in top_similarities]
    
    return top_descriptions, top_code, top_similarities_percentage
