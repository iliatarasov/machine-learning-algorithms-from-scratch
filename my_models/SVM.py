import numpy as np


class MySVM:
    def __init__(self, C: float=1.0, expand: bool=True) -> None:
        self.expand = expand
        self.C = C
        
    def hinge_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        loss = 0
        for i in range(len(X)):
            t = y[i] * (np.matmul(self.w.T, X[i].reshape(-1, 1)) + self.b).item()
            loss += np.max([0, 1 - t])
        return loss / len(X)
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            n_iter: int=10, 
            learning_rate: float=1e-3,
            ) -> None:
        if X.ndim == 2 and self.expand:
            X = self.expand_dims(X)
        n_samples, n_features = X.shape
        self.w = np.zeros((n_features, 1))
        self.b = 0
        self.loss = []
        for _ in range(n_iter):
            self.loss.append(self.hinge_loss(X, y))
            w_grad = self.w
            b_grad = 0
            for i in range(n_samples):
                x_i = X[i].reshape(-1, 1)
                y_i = y[i]
                t_i = y_i * (np.matmul(self.w.T, x_i) + self.b).item()
                if t_i <= 1:
                    w_grad += self.C * y_i * x_i
                    b_grad += y_i
            self.w -= learning_rate * w_grad
            self.b += learning_rate * b_grad
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2 and self.expand:
            X = self.expand_dims(X)
        out = np.sign(np.matmul(self.w.T, X.T) + self.b)
        out = np.array([np.sign(np.matmul(self.w.T, x.reshape(-1, 1)) + self.b)for x in X]).ravel()
        if out.ndim == 2:
            return out[0]
        return out
    
    def expand_dims(self, X: np.ndarray, method='poly3') -> np.ndarray:
        if method == 'poly3':
            new_dims = [col.reshape(-1, 1) for col in   [
                            
                            np.sum(X**3, axis=1),
                            np.sum(X ** 2, axis=1),
                            #np.sum(X, axis=1),
                                                        ]
                        ]
            return np.hstack([X, *new_dims])
        else:
            raise ValueError(f'Method {method} not supported')
        
        
            
            
    
    
        