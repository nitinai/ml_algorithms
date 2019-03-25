import pytest

import numpy as np
from mllearn.ensemble import RandomForest

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

RANDOM_STATE = 17

# =============================================================================
# Classification tests
# =============================================================================
@pytest.fixture
def classification_dataset():
    ''' Prepare a synthetic data for classification '''
    X, y = make_classification(n_features=2, n_redundant=0, n_samples=400,
                            random_state=RANDOM_STATE)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=RANDOM_STATE)

    return X_train, X_test, y_train, y_test


def test_random_forest_classifer_entropy(classification_dataset):
    """Test decision tree classifier """
    X_train, X_test, y_train, y_test = classification_dataset
    clf = RandomForest(n_estimators=10, max_depth=7, max_features=2, random_state=RANDOM_STATE) 
    clf.fit(X_train, y_train)
    #y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    predict_accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {predict_accuracy}")
    assert( predict_accuracy > 0.86 )
    assert( np.sum(np.argmax(y_pred_prob, axis=1) - y_pred) == 0 )
