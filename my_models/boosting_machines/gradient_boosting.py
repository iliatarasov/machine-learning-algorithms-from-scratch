import numpy as np
from sklearn.tree import DecisionTreeClassifier

class MyGBClassifier:
    '''Gradient boosting classifier class'''
    def __init__(self, 
                 base_clf=DecisionTreeClassifier,
                 **cfr_args):
        '''
        Arguments:
            base_clf: base classifier class
            clf_args (optional): collection of keyword arugments for base 
            classifiers
        '''
        self.base_clf = base_clf
        self.cfr_args = cfr_args
        
    def get_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        '''Calculates loss for prediction vector'''
        return 1/2 * np.mean(np.square(y - y_pred))
    
    def get_gradient(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        '''Returns gradient of the loss function at the prediction vector'''
        return - (y - y_pred)
    
    def get_base_classifier(self):
        '''Returns a new base classifier'''
        return self.base_clf(**self.cfr_args)
    
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray, 
            n_estimators: int=100, 
            learning_rate: float=0.1) -> None:
        '''Training procedure'''
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
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Returns an array of predictions for given datapoints X'''
        y_pred = np.zeros(len(X))
        for classifier in self.classifiers:
            y_pred -= self.learning_rate * classifier.predict(X)
        return np.sign(y_pred)