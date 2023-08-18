import math
import numpy as np

from sklearn.tree import DecisionTreeClassifier 

class MyABClassifier:
    '''Adaptive boosting classifier class'''
    def __init__(self, 
                 n_estimators: int=50, 
                 base_clf = DecisionTreeClassifier,
                 **clf_args) -> None:
        '''
        Arguments:
            n_estimators (int): number of weak estimators
            base_clf: base classifier class
            clf_args (optional): collection of keyword arugments for base 
            classifiers
        '''
        self.n_estimators = n_estimators
        self.base_clf = base_clf
        self.clf_args = clf_args
        
    def get_base_classifier(self):
        '''Returns a new base classifier'''
        return self.base_clf(**self.clf_args)
        
    def fit(self, X: np.ndarray, y_true: np.ndarray) -> None:
        '''Training procedure'''
        n_samples = len(X)
        # initialize a uniform distribution
        distribution = np.ones(n_samples, dtype=float) / n_samples
        self.classifiers = []
        self.alphas = []
        for _ in range(self.n_estimators):
            # create a new classifier
            self.classifiers.append(self.get_base_classifier())     
            self.classifiers[-1].fit(X, y_true, sample_weight=distribution)
            # make a prediction
            y_pred = self.classifiers[-1].predict(X)
            # update alphas, append new alpha to self.alphas
            alpha = self.get_alpha(y_true, y_pred, distribution)
            self.alphas.append(alpha)
            # update the distribution
            distribution = self.update_distribution(y_true, y_pred, 
                                                    distribution, alpha)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Returns an array of predictions for given datapoints X'''
        final_predictions = np.zeros(X.shape[0])
        # get weighted votes from each classifier
        for alpha, clf in zip(self.alphas, self.classifiers):
            final_predictions += alpha * clf.predict(X)
        # summarize the votes by sign
        out = np.sign(final_predictions)
        return out
    
    def get_alpha(self, 
                  y: np.ndarray, 
                  y_pred_t: np.ndarray, 
                  distribution: np.ndarray) -> float:
        '''Returns error for a data point'''
        e_i = [int(not y[i] == y_pred_t[i]) for i in range(len(y))]
        e_t = sum([e * w for e, w in zip(e_i, distribution)]) / sum(distribution)
        # return alpha
        return 1/2 * math.log((1-e_t)/e_t)

    def update_distribution(self, 
                            y: np.ndarray, 
                            y_pred_t: np.ndarray, 
                            distribution: np.ndarray, 
                            alpha_t: float) -> np.ndarray:
        '''Updates and normalizes the distribution'''
        new_distribution = np.array([distribution[i] * math.exp(alpha_t * \
            (-1) ** (y[i] == y_pred_t[i])) for i in range(len(y))])
        return new_distribution / new_distribution.sum()