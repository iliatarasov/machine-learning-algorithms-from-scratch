import math
import random
import collections
from typing import Self, Any
import numpy as np
import pandas

class Question:
    def __init__(self, column: int, value: int|float) -> None:
        self.column = column
        self.value = value
        self.numeric = True if isinstance(value, float) or \
            isinstance(value, int) else False
        
    def match(self, example: np.ndarray) -> bool:
        '''
        Compare the feature value in an example to the feature value in this
        question
        '''
        value = example[self.column]
        if self.numeric:
            return value >= self.value
        else:
            return value == self.value

class Leaf:
    def __init__(self, data: np.ndarray) -> None:
        self.predictions = collections.Counter(data[:, -1])
        
    def predict(self) -> Any:
        s = sum(self.predictions.values())
        p = [value / s for value in self.predictions.values()]
        return random.choices(list(self.predictions.keys()), p)[0]
        
class DesicionNode:
    def __init__(self, question: Question, true_branch: Self|Leaf,
                 false_branch: Self|Leaf) -> None:
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        
class MyDT:
    def __init__(self):
        pass
    
    def partition(self, data: np.ndarray, question: Question) -> tuple[list]:
        '''Partitions a dataset by answer to the question'''
        true_rows, false_rows = [], []
        for row in data:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return np.array(true_rows), np.array(false_rows)
    
    def gini(self, data: np.ndarray) -> float:
        '''Calculates gini impurity for data'''
        counts = collections.Counter(data[:, -1])
        impurity = 1
        for label in counts:
            impurity -= (counts[label] / len(data))**2
        return impurity
    
    def info_gain(self, left: np.ndarray, 
                  right: np.ndarray, cur_uncertainty: float) -> float:
        '''Calculates information gain for a given split'''
        p = len(left) / (len(left) + len(right))
        ig = cur_uncertainty - p * self.gini(left) - (1 - p) * self.gini(right)
        return ig
    
    def find_best_split(self, data: np.ndarray) -> tuple[float, Question]:
        best_gain = 0
        best_question = None
        cur_uncertainty = self.gini(data)
        for column in range(len(data[0]) - 1):
            values = set([row[column] for row in data])
            for value in values:
                question = Question(column, value)
                true_rows, false_rows = self.partition(data, question)
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                info_gain = self.info_gain(true_rows, false_rows, 
                                           cur_uncertainty)
                if info_gain >= best_gain:
                    best_gain, best_question = info_gain, question
        return best_gain, best_question
    
    def build_tree(self, data: np.ndarray) -> DesicionNode|Leaf:
        gain, question = self.find_best_split(data)
        if gain == 0:
            return Leaf(data)
        true_rows, false_rows = self.partition(data, question)
        true_branch = self.build_tree(true_rows)
        false_branch = self.build_tree(false_rows)
        return DesicionNode(question, true_branch, false_branch)
    
    def classify(self, row: np.ndarray, node: DesicionNode|Leaf) -> None:
        if isinstance(node, Leaf):
            return node.predict()
        if node.question.match(row):
            return self.classify(row, node.true_branch)
        else:
            return self.classify(row, node.false_branch)
            
    def fit(self, data: np.ndarray) -> None:
        if isinstance(data, pandas.core.frame.DataFrame):
            data = data.values
        self.head = self.build_tree(data)
        
    def predict(self, rows: np.ndarray) -> Any:
        if isinstance(rows, pandas.core.frame.DataFrame):
            rows = rows.values
        if rows.ndim == 1:
            return self.classify(rows, self.head)
        else:
            predictions = []
            for row in rows:
                predictions.append(self.classify(row, self.head))
            return np.array(predictions)
    