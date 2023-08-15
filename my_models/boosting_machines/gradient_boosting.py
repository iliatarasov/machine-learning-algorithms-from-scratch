import numpy as np
from sklearn.tree import DecisionTreeClassifier

class MyGBClassifier:
    def __init__(self, **cfr_args):
        self.cfr_args = cfr_args
        
    def get_loss(self, y, y_pred):
        return 1/2 * np.mean(np.square(y - y_pred))
    
    def get_gradient(self, y, y_pred):
        return - (y - y_pred)
    
    def get_base_classifier(self):
        return DecisionTreeClassifier(**self.cfr_args)
    
    def fit(self, X, y, n_estimators=100, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.classifiers = []
        self.losses = []
        # get a vector of mean(y) values as 0th prediction
        y_pred = np.zeros_like(y, dtype='float64')
        for _ in range(n_estimators):
            loss = self.get_loss(y, y_pred)
            self.losses.append(loss)
            # calculate gradient
            grad = self.get_gradient(y, y_pred)
            # create a classifier and train it with gradient as y
            self.classifiers.append(self.get_base_classifier())
            self.classifiers[-1].fit(X, grad)
            # calculate residuals
            residuals = self.classifiers[-1].predict(X).astype('float')
            # update predictions
            y_pred -= learning_rate * residuals
            y_pred = np.sign(y_pred)
        
    def predict(self, X):
        y_pred = np.zeros(len(X))
        for classifier in self.classifiers:
            y_pred -= self.learning_rate * classifier.predict(X)
        return np.sign(y_pred)