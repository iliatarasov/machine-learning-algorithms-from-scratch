import warnings
from typing import Any

import numpy as np


class MySVC:
    '''Support vector machine classifier class'''
    def __init__(self, max_iter: int=10000,
                 tol: float=1e-4,
                 C: float=1.0, 
                 expand: bool=True,
                 expand_method: str='poly3') -> None:
        '''
        Arguments:
            max_iter (int): maximum number of iterations
            tol (float): tolerance of convergence
            C (float): hinge loss grad weight coefficient
            expand (bool): will expand dimensions if True
            expand_method (str): method for dimension expansion, currently 
            only degree-3 polynomial is supported
        '''
        SUPPORTED_METHODS = ['poly3']
        self.tol = tol
        self.max_iter = max_iter
        self.expand = expand
        self.C = C
        if expand_method not in SUPPORTED_METHODS:
            raise ValueError(f'Method {expand_method} not supported')
        self.expand_method = expand_method
        
    def hinge_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        '''Hinge loss calculation'''
        loss = 0
        for i in range(len(X)):
            t = y[i] * (np.matmul(self.w.T, X[i].reshape(-1, 1)) + self.b).item()
            loss += np.max([0, 1 - t])
        return loss / len(X)
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            learning_rate: float=1e-3) -> None:
        '''Training process'''
        if X.ndim == 2 and self.expand:
            X = self.expand_dims(X, method=self.expand_method)
            
        n_samples, n_features = X.shape
        self.w = np.zeros((n_features, 1))
        self.b = 0
        
        self.loss = [self.hinge_loss(X, y)]
        
        theta = float('inf')
        epoch = 0
        
        while theta > self.tol and epoch < self.max_iter:
            w_grad = - self.w
            b_grad = 0
            for i in range(n_samples):
                x_i = X[i].reshape(-1, 1)
                y_i = y[i]
                t_i = y_i * (np.matmul(self.w.T, x_i) + self.b).item()
                if t_i <= 1:
                    w_grad -= self.C * y_i * x_i
                    b_grad -= y_i
            self.w -= learning_rate * w_grad
            self.b -= learning_rate * b_grad
            
            loss = self.hinge_loss(X, y)
            self.loss.append(loss)
            theta = abs(self.loss[-1] - self.loss[-2])
            epoch += 1
            
        if epoch >= self.max_iter:
            warnings.warn(f'Solver {self.__class__.__name__} terminated early')
        self.convergence = epoch
            
    def predict(self, X: np.ndarray) -> Any:
        '''Predicts class from test data'''
        if X.ndim == 2 and self.expand:
            X = self.expand_dims(X, method=self.expand_method)
        out = np.sign(np.matmul(self.w.T, X.T) + self.b)
        out = np.array([np.sign(np.matmul(self.w.T, x.reshape(-1, 1)) + \
            self.b)for x in X]).ravel()
        if out.ndim == 2:
            return out[0]
        return out
    
    def expand_dims(self, X: np.ndarray, method='poly3') -> np.ndarray:
        '''Method that expands linear data to a degree n polynomial'''
        if method == 'poly3':
            new_dims = [col.reshape(-1, 1) for col in   [
                            X[:, 0] ** 3,
                            X[:, 1] ** 3,
                            X[:, 0] ** 2,
                            X[:, 1] ** 2,
                            X[:, 0] ** 2 * X[:, 1],
                            X[:, 1] ** 2 * X[:, 0],
                                                        ]
                        ]
            return np.hstack([X, *new_dims])
            
        
        
            
            
    
    
        