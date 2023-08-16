from typing import Any
from collections import Counter
import numpy as np

from .desicion_tree import MyDTClassifier


class MyRFClassifier:
    def __init__(self, n_estimators: int=50, base_classifier=MyDTClassifier):
        self.n_estimators = n_estimators
        self.base_classifier=base_classifier
    
    def test_accuracy(self, tree: MyDTClassifier, test_data: np.ndarray) -> float:
        predictions = tree.predict(test_data[:, :-1])
        return sum(predictions == test_data[:, -1]) / len(predictions)
        
    def bootstrap(self, data: np.ndarray) -> tuple[np.ndarray]:
        choice = np.random.choice(range(len(data)), len(data), 
                                        replace = True)
        bootstrap_idx = np.zeros(data.shape[0], dtype=bool)
        bootstrap_idx[choice] = True
        test_idx = ~bootstrap_idx
        return data[bootstrap_idx], data[test_idx]
    
    def fit(self, data: np.ndarray) -> None:
        self.trees = []
        self.test_scores = []
        for _ in range(self.n_estimators):
            bootstrap, test = self.bootstrap(data)
            tree = self.base_classifier()
            tree.fit(bootstrap)
            self.trees.append(tree)
            test_score = self.test_accuracy(tree, test)
            self.test_scores.append(test_score)
        
    def predict(self, data: np.ndarray) -> np.ndarray[Any]:
        predictions = []
        for row in data:
            ensemble_predictions = [tree.predict(row) for tree in self.trees]
            counter = Counter(ensemble_predictions)
            predictions.append(counter.most_common(1)[0][0])
        return np.array(predictions)