import warnings
import collections
import numpy as np

class MyKMeans:
    '''K-means clusterization class'''
    def __init__(self, n_clusters: int=2, max_iter: int=1000):
        '''
        Arguments:
            n_clusters (int): how many clusters the data has to be split in
            max_iter (int): max number of iterations before forced termination
        '''
        self.k = n_clusters
        self.max_iter = max_iter
        
    def distances_to_centers(self, point: np.ndarray) -> np.ndarray:
        '''
        Function to calculate euclidean distances between a given point
        and the current cluster centers
        '''
        return np.sqrt(np.sum((self.centers - point) ** 2, axis=1))
    
    def assign_to_cluster(self, point: np.ndarray) -> int:
        '''
        Assigns a point to a cluster based on the distance to one of the
        current cluster centers
        '''
        if point in self.centers:
            cluster = np.where(point==self.centers)[0][0]
        else:
            distances = self.distances_to_centers(point)
            cluster = np.argmin(distances)
        return cluster
    
    def new_centers(self, X: np.ndarray, clusters: list) -> np.ndarray:
        '''Calculates new centers for given clusters'''
        clustered = collections.defaultdict(list)
        for c, x in zip(clusters, X):
            clustered[c].append(x)
        centers = []
        for c in clustered:
            cluster = np.array(clustered[c])
            centers.append(cluster.mean(axis=0))
        return np.array(centers)
        
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        '''Train the model on X and return clusters'''
        # chose centers randomly at first
        self.centers = X[np.random.choice(X.shape[0], self.k, replace=False)]
        #self.centers = np.random.choice(X, self.k, replace=False)   
        self.iteration = 0    
        # iterate until the centers no longer change or max_iter is reached
        while True:
            # find a cluster for each point by distance 
            clusters = [self.assign_to_cluster(point) for point in X]
            # find new centers
            new_centers = self.new_centers(X, clusters)
            # check exit conditions
            if (np.sort(new_centers) == np.sort(self.centers)).all():
                break
            else:
                self.centers = new_centers
            self.iteration += 1
            if self.iteration >= self.max_iter:
                warnings.warn(f'Algorithm {self.__class__.__name__} terminated\
                     early')
                break
        return np.array([self.assign_to_cluster(point) for point in X])
    
    def fit(self, X: np.ndarray) -> None:
        '''Train the model'''
        self.fit_predict(X)
            
    def predict(self, X: np.ndarray):
        '''Assign a new set of points to the existing clusters'''
        return np.array([self.assign_to_cluster(point) for point in X])
                