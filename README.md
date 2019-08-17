[![Build Status](https://travis-ci.org/nitinai/ml_algorithms.svg?branch=master)](https://travis-ci.org/nitinai/ml_algorithms)
[![codecov](https://codecov.io/gh/nitinai/ml_algorithms/branch/master/graph/badge.svg)](https://codecov.io/gh/nitinai/ml_algorithms)

# Machine Learning Algorithms

This repository contains bare minimum implementation of different machine learning algorithms. The main purpose of this project is to understand mechanics of these algorthms.

## Implemented algorithms 
* [Decision trees](mllearn/ensemble/forest.py)
* [Random forests](mllearn/tree/tree.py)

Test cases are written for all algorithms. You can be find them [here](mllearn/tests)

## How to use
	
	git clone https://github.com/nitinai/ml_algorithms.git
	cd ml_algorithms
	
	from sklearn.datasets import load_iris
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score
	iris = load_iris()
	X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3,
                                                    random_state=123)
	
	from mllearn.tree import DecisionTreeClassifier
	clf = DecisionTreeClassifier()
	y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)
	predict_accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {predict_accuracy}")
	
	
Please feel free to [open an issue](https://github.com/nitinai/ml_algorithms/issues/new) if you find a bug in implementation.
