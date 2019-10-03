import pytest

import numpy as np
from mllearn.metrics import accuracy_score, mean_squared_error

from sklearn.metrics import accuracy_score as skaccuracy_score
from sklearn.metrics import mean_squared_error as skmean_squared_error

def test_accuracy_score():
    """Test accuracy score"""
    # binary
    for i in range(10):
        rng = np.random.RandomState(i)
        y_true = rng.randint(2, size=10)
        y_pred = rng.randint(2, size=10)
        score1 = accuracy_score(y_true, y_pred)
        score2 = skaccuracy_score(y_true, y_pred)
        assert (np.isclose(score1, score2))

def test_mean_squared_error():
    """Test mean_squared_error"""
    # binary
    for i in range(10):
        rng = np.random.RandomState(i)
        y_true = rng.rand(10)
        y_pred = rng.rand(10)
        score1 = mean_squared_error(y_true, y_pred)
        score2 = skmean_squared_error(y_true, y_pred)
        assert np.isclose(score1, score2)
