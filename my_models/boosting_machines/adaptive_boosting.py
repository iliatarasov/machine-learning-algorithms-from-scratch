import math
import numpy as np

class MyABClassifier():
    def __init__(self, n_estimators=50, base_classifier=None):
        self.n_estimators = n_estimators
        self.base_classifier = base_classifier
        
    def fit(self, X, y_true):
        n_samples = len(X)
        # initialize a uniform distribution
        distribution = np.ones(n_samples, dtype=float) / n_samples
        self.classifiers = []
        self.alphas = []
        for _ in range(self.n_estimators):
            # create a new classifier
            self.classifiers.append(self.base_classifier())     
            self.classifiers[-1].fit(X, y_true, sample_weight=distribution)
            # make a prediction
            y_pred = self.classifiers[-1].predict(X)
            # update alphas, append new alpha to self.alphas
            alpha = self.get_alpha(y_true, y_pred, distribution)
            self.alphas.append(alpha)
            # update the distribution
            distribution = self.update_distribution(y_true, y_pred, 
                                                    distribution, alpha)
            
    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])
        # get weighted votes from each classifier
        for alpha, clf in zip(self.alphas, self.classifiers):
            final_predictions += alpha * clf.predict(X)
        # summarize the votes by sign
        out = np.sign(final_predictions)
        return out
    
    def get_alpha(self, y, y_pred_t, distribution):
        # calculate error for each data point
        e_i = [int(not y[i] == y_pred_t[i]) for i in range(len(y))]
        e_t = sum([e * w for e, w in zip(e_i, distribution)]) / sum(distribution)
        # return alpha
        return 1/2 * math.log((1-e_t)/e_t)

    def update_distribution(self, y, y_pred_t, distribution, alpha_t):
        # calculate new distribution and normalize
        new_distribution = np.array([distribution[i] * math.exp(alpha_t * \
            (-1) ** (y[i] == y_pred_t[i])) for i in range(len(y))])
        return new_distribution / new_distribution.sum()