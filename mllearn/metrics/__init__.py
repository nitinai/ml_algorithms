"""
The :mod:`mllearn.metrics` module includes score functions, performance metrics
and pairwise metrics and distance computations.

"""

from .classification import accuracy_score
from .regression import mean_squared_error

__all__ = [ "accuracy_score",
            "mean_squared_error"]
