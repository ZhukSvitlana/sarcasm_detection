from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_vectorizer(sentences, **tfidf_params):
    vectorizer = TfidfVectorizer(**tfidf_params)
    vectorizer.fit(sentences)

    return vectorizer

get_tfidf_vectorizer.__doc__="""
Generates a TF-IDF vectorizer for the given sentences.

    Parameters:
        sentences (list): List of strings representing sentences.
        **tfidf_params: Additional parameters to pass to TfidfVectorizer.

    Returns:
        TfidfVectorizer: TF-IDF vectorizer fitted on the input sentences.
        """