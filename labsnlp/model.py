from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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