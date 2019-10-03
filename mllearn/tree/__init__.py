"""
The :mod:`mllearn.tree` module includes decision tree-based models for
classification and regression.
"""

from .tree import DecisionTree
from .tree import DecisionTreeClassifier
from .tree import DecisionTreeRegressor

__all__ = [ "DecisionTree",
            "DecisionTreeClassifier",
            "DecisionTreeRegressor"]

