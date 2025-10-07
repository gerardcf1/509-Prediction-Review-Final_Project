import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def get_top_features(text, model, n=5):
    """
    Extracts the top n contributing words from the given model for interpretability.
    """
    # Locate the vectorizer and regressor inside the pipeline
    vectorizer = None
    regressor = None

    for name, step in model.named_steps.items():
        if isinstance(step, (TfidfVectorizer, CountVectorizer)):
            vectorizer = step
        else:
            regressor = step

    if vectorizer is None or not hasattr(regressor, "coef_"):
        return []

    # Get feature names and model coefficients
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = regressor.coef_.ravel()

    # Transform input text into vector space
    X = vectorizer.transform([text])
    weights = X.toarray().ravel() * coefs

    # Get indices of top features
    top_idx = np.argsort(weights)[-n:]
    return list(feature_names[top_idx][::-1])
