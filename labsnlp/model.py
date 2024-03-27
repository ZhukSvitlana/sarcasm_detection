from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

def train_gaussian_nb_classifier(X, y):
    clf = GaussianNB()
    clf.fit(X, y)
    return clf

def train_native_bayes_classifier(X, y):
    clf = MultinomialNB()
    clf.fit(X,y)
    return clf

def train_logistic_regression(X, y):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X,y)
    return clf

def train_SGD_classifier(X, y):
    clf = SGDClassifier()
    clf.fit(X,y)
    return clf

def train_neighbors_classifier(X, y):
    clf = KNeighborsClassifier(algorithm = 'brute', n_jobs=-1)
    clf.fit(X,y)
    return clf

def train_svm_classifier(X, y):
    clf = LinearSVC(C=0.0001)
    clf.fit(X,y)
    return clf

def train_tree_classifier(X, y):
    clf = DecisionTreeClassifier(min_samples_split=10,max_depth=3)
    clf.fit(X,y)
    return clf