import collections
import random
import numpy as np


class MyKNN:
    '''KNN classifier class'''
    def __init__(self, 
                 n_neighbors: int=5, 
                 method: str='maj') -> None:
        '''Arguments:
            n_neighbors (int): number of neighbors to consider
            multiclass_method (str): name for the classification method to use:
                'maj' for 'majority': label is selected by most common among
                neighbors;
                'prob' for 'probabilistic': where label is drawn randomly
                among neighbors (with uniform distribution)
        '''
        MULTICLASS_METHODS = ['maj', 'prob']
        if method not in MULTICLASS_METHODS:
            raise ValueError(f'Method {method} not supported.\
                Please select from {MULTICLASS_METHODS}')
        self.k = n_neighbors
        self.trained = False
        self.method = method
    
    def get_distances(self, point: np.ndarray) -> np.ndarray:
        '''
        Function to calculate euclidean distances between a given point
        and the training data
        '''
        return np.sqrt(np.sum((self.X - point) ** 2, axis=1))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Training consists of just remembering training points and labels
        for later predictions
        Arguments:
            X: training points
            y: training labels
        '''
        self.X = X
        self.y = y
        self.n_classes = len(set(y))
        self.trained = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Makes predictions on a given set of testing points
        '''
        assert self.trained, 'Train the model first'
        predictions = []
        for x in X:
            dists = self.get_distances(x)
            neighbors = sorted(zip(dists, self.y), key=lambda x: x[0])
            k_nearest = neighbors[:self.k]
            match self.method:
                case 'maj':
                    labels = collections.Counter([neighbor[1] 
                                                for neighbor in k_nearest])
                    label = max(labels.items(), key=lambda x: x[1])[0]
                case 'prob':
                    label = random.choice(k_nearest)[1]
            predictions.append(label)
        predictions = np.sign(predictions)
        return predictions
        
        