import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
class CreditModelDecisionTree:
    def __init__(self):
        self.model = DecisionTreeClassifier(n_estimators=100, max_depth=50)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class CreditModelRandomForestClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=50)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class CreditModelMLPClassifier:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(100, ),)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class CreditModelSGDClassifier:
    def __init__(self):
        self.model = SGDClassifier()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class CreditModelSVM:
    def __init__(self):
        self.model = svm.SVR()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)