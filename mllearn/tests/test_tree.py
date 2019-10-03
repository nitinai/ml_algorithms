import pytest

import numpy as np
from mllearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from mllearn.metrics import accuracy_score, mean_squared_error

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

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


def test_decision_tree_classifer_entropy(classification_dataset):
    """Test decision tree classifier """
    X_train, X_test, y_train, y_test = classification_dataset
    clf = DecisionTreeClassifier(max_depth=4, min_samples_split=2, criterion='entropy')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)

    predict_accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {predict_accuracy}")
    assert( predict_accuracy > 0.86 )
    assert( np.sum(np.argmax(y_pred_prob, axis=1) - y_pred) == 0 )

def test_decision_tree_classifer_gini(classification_dataset):
    """Test decision tree classifier """
    X_train, X_test, y_train, y_test = classification_dataset
    clf = DecisionTreeClassifier(max_depth=4, min_samples_split=2, criterion='gini')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)

    predict_accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {predict_accuracy}")
    assert( predict_accuracy > 0.85 )
    assert( np.sum(np.argmax(y_pred_prob, axis=1) - y_pred) == 0 )


# =============================================================================
# Regression tests
# =============================================================================
@pytest.fixture
def regression_dataset():
    ''' Prepare a synthetic data for regression '''
    X, y = make_regression(n_features=1, n_samples=200, bias=0, noise=5,
                      random_state=RANDOM_STATE)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                        random_state=RANDOM_STATE)

    return X_train, X_test, y_train, y_test

def test_decision_tree_regressor_mse(regression_dataset):
    """Test decision tree classifier """
    X_train, X_test, y_train, y_test = regression_dataset
    reg = DecisionTreeRegressor(max_depth=6, min_samples_split=2, criterion='mse')
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error : {mse}")
    assert( mse < 160 )

def test_decision_tree_regressor_mad_median(regression_dataset):
    """Test decision tree classifier """
    X_train, X_test, y_train, y_test = regression_dataset
    reg = DecisionTreeRegressor(max_depth=6, min_samples_split=2, criterion='mad_median')
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error : {mse}")
    assert( mse < 169 )
