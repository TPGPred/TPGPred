
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


def extract_features(method, corpus):

    if method == "HashingVectorizer":
        vectorizer = HashingVectorizer(analyzer='char', n_features=1000)
        X = vectorizer.fit_transform(corpus).toarray()

    return X

