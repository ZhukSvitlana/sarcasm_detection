from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#ML models that can be choosen in train
def train_gaussian_nb_classifier(X, y):
    clf = GaussianNB()
    clf.fit(X, y)
    return clf

def train_native_bayes_classifier(X, y):
    clf = MultinomialNB(alpha=0.00000000001, force_alpha=True, fit_prior=True)
    clf.fit(X,y)
    return clf

def train_logistic_regression(X, y):
    clf = LogisticRegression(max_iter=100)
    clf.fit(X,y)
    return clf

def train_neighbors_classifier(X, y):
    clf = KNeighborsClassifier(n_neighbors = 3, algorithm = 'auto', weights = 'distance',n_jobs=-1)
    clf.fit(X,y)
    return clf

def train_tree_classifier(X, y):
    clf = DecisionTreeClassifier(min_samples_split=10,max_depth=30)
    clf.fit(X,y)
    return clf


train_gaussian_nb_classifier.__doc__="""
    Trains a Gaussian Naive Bayes classifier.

    Parameters:
        X (array-like): Input features.
        y (array-like): Target labels.

    Returns:
        GaussianNB: Trained Gaussian Naive Bayes classifier.
    """
train_native_bayes_classifier.__doc__="""
    Trains a Multinomial Naive Bayes classifier.

    Parameters:
        X (array-like): Input features.
        y (array-like): Target labels.

    Returns:
        MultinomialNB: Trained Multinomial Naive Bayes classifier.
    """
train_logistic_regression.__doc__="""
    Trains a Logistic Regression classifier.

    Parameters:
        X (array-like): Input features.
        y (array-like): Target labels.

    Returns:
        LogisticRegression: Trained Logistic Regression classifier.
    """
train_neighbors_classifier.__doc__="""
    Trains a K-Nearest Neighbors classifier.

    Parameters:
        X (array-like): Input features.
        y (array-like): Target labels.

    Returns:
        KNeighborsClassifier: Trained K-Nearest Neighbors classifier.
    """
train_tree_classifier.__doc__="""
    Trains a Decision Tree classifier.

    Parameters:
        X (array-like): Input features.
        y (array-like): Target labels.

    Returns:
        DecisionTreeClassifier: Trained Decision Tree classifier.
    """