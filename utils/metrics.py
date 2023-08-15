import numpy as np

def accuracy_score(y_pred, y_true) -> float:
    correct = [y_pred[i] == y_true[i] for i in range(len(y_pred))]
    return sum(correct) / len(correct)