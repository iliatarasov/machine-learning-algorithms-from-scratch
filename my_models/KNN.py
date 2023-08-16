import numpy as np

class MyKNN:
    '''KNN classifier class'''
    def __init__(self, n_neighbors: int=5) -> None:
        '''Arguments:
            n_neighbors (int): number of neighbors to consider'''
        self.k = n_neighbors
        self.trained = False
    
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
            k_nearest = neighbors[1 : 1 + self.k]
            label = sum([neighbor[1]/self.k for neighbor in k_nearest])
            predictions.append(label)
        predictions = np.sign(predictions)
        return predictions
        
        