def KNN(df_cleaned, filenames, codes_to_find): 
    from gensim import corpora, models, similarities, utils
    import ast
    from sklearn.metrics import precision_score, recall_score, f1_score

    # Load the TF-IDF model and similarity index
    tfidf = models.TfidfModel.load('tfidf_model')  # Ensure 'tfidf_model' file exists
    index = similarities.SparseMatrixSimilarity.load('similarity_index')  # Ensure 'similarity_index' file exists
    dictionary = corpora.Dictionary.load('dictionary')

    y_true_global = []
    y_pred_global = []

    for i in range(len(filenames)):
        filename = filenames[i]
        query_first_letters = codes_to_find[i]

        # Define your query document
        with open(filename, "r", encoding="utf-8") as f:
            query_document = f.read()  # Read the content of the file

        # Tokenize the query document
        query_tokens = utils.simple_preprocess(query_document)

        # Convert the query document to BoW format using the loaded dictionary
        query_bow = dictionary.doc2bow(query_tokens)

        # Transform the query BoW vector with TF-IDF
        query_tfidf = tfidf[query_bow]

        # Calculate cosine similarity between the query and all documents in the corpus
        sims = index[query_tfidf]

        # Sort the similarities and retrieve the top three most similar documents
        top_three = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[1:3]
        print(top_three)
        # Extract the first letters of the CPC codes from the top three most similar documents
        similar_first_letters = set()
        for document_number, _ in top_three:
            cpc_codes_str = df_cleaned['CPC'][document_number]
            cpc_codes = ast.literal_eval(cpc_codes_str)
            first_letters = set(code[0] for code in cpc_codes)
            similar_first_letters.update(first_letters)

        # Calculate true positives, false positives, and false negatives for precision and recall
        y_true = []
        y_pred = []
        
        for letter in query_first_letters:
            y_true.append(1)
            y_pred.append(1 if letter in similar_first_letters else 0)
        
        for letter in similar_first_letters:
            if letter not in query_first_letters:
                y_true.append(0)
                y_pred.append(1)

        # Append results to global lists
        y_true_global.extend(y_true)
        y_pred_global.extend(y_pred)

        # Print results for each query document
        print(f"Query Document: {filename}")
        print(f"Query First Letters: {query_first_letters}")
        print(f"Similar First Letters: {similar_first_letters}")
        print()

    # Calculate global precision, recall, and F1 score
    precision = precision_score(y_true_global, y_pred_global)
    recall = recall_score(y_true_global, y_pred_global)
    f1 = f1_score(y_true_global, y_pred_global)

    print(f"Global Precision: {precision:.4f}")
    print(f"Global Recall: {recall:.4f}")
    print(f"Global F1 Score: {f1:.4f}")