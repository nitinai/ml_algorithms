import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error

__all__ = ["DecisionTree"]

# =============================================================================
# Supportive functions
# =============================================================================

# Classification criterion functions
def entropy(y):    
    P = [len(y[y==k]) / len(y) for k in np.unique(y)]
    return -1 * np.dot(P, np.log2(P))

def gini(y):
    P = [len(y[y==k]) / len(y) for k in np.unique(y)]
    return 1 - np.dot(P, P)

# Regression criterion functions
def variance(y):
    return np.var(y)

def mad_median(y):
    return np.mean(np.abs(y-np.median(y)))

# Dictionary for easy mapping with input string
criterion_dict = {'entropy': entropy,
                   'gini': gini,
                   'mse': variance,
                   'mad_median' : mad_median}

# Target prediction functions
def classification_leaf(y):
    """the most popular class in leaf"""
    return np.bincount(y).argmax()

def regression_leaf(y):
    """the mean of all values in a leaf"""
    return np.mean(y)


# =============================================================================
# Decision Tree
# =============================================================================
class Node():
    '''Node class represents a node in a decision tree'''
    
    def __init__(self, feature_idx=0, threshold=0, labels=None, left=None, right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.labels = labels
        self.left = left
        self.right = right


class DecisionTree(BaseEstimator):
    
    def __init__(self, max_depth=np.inf, min_samples_split=2, 
                 criterion='gini', debug=False):
        
        params = {'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'debug' : debug,
                  'criterion' : criterion}
                
        super(DecisionTree, self).set_params(**params)
        
        self._criterion_fun = criterion_dict[self.criterion]
        
        if self.criterion in ['mse', 'mad_median']:
            self._leaf_value = regression_leaf
        else:
            self._leaf_value = classification_leaf
            
        if debug:
            print("DecisionTree")
            print(f"max_depth: {self.max_depth}, min_samples_split: {self.min_samples_split}, \
                  criterion: {self.criterion}, debug: {self.debug}")
        
    
    def _functional(self, X, y, feature_idx, threshold):
        '''A functional returns the gain achieved if we split the data at given feature and threshold value'''

        if threshold is np.nan:
            return 0
        
        mask = X[:,feature_idx] <= threshold
        X_l = X[ mask ]
        y_l = y[ mask ]

        X_r = X[ ~mask ]
        y_r = y[ ~mask ]
        
        # if all the data goes to one of the child
        if len(X_l) == 0 or len(X_r) == 0:
            return 0

        return self._criterion_fun(y) - \
                (X_l.shape[0]/X.shape[0])* self._criterion_fun(y_l) - \
                (X_r.shape[0]/X.shape[0])* self._criterion_fun(y_r)

    
    
    def _build_tree(self, X, y, depth = 1):
        ''' Recursive function to split the data and form nodes'''
    
        # We already reached to the max_depth, so time to leave recursion 
        # by creating a leaf Node
        if depth > self.max_depth:
            return Node(labels=y)
        
        n_samples, n_features = X.shape
        
        # We do not have sufficient samples to split
        if n_samples < self.min_samples_split:
            return Node(labels=y)
        
        # If all objects in a current vertex have the same values in answers
        # then the value of the functional will be 0 for all partitions.
        # So in this case the vertex is a leaf. In order not to make unnecessary calculations, 
        # perform this check before the main cycle.
        if len(np.unique(y)) == 1:
            return Node(labels=y)
        
        # Here we are trying to split the data such that we will have maximun
        # gain out of split.
        # We will simulate the split for each unique value of each feature and
        # calculate the functional gain. On evey account of finding the maximum gain 
        # from the previous we will keep storing the feature index and threshold value
        # which gave this gain.
        # At the end of this search we will have the best feature index and threshold
        # value we should use to split the data into left and right nodes.
        max_gain = 0
        best_feat_idx = 0
        best_threshold = 0
        
        for feature_idx in range(n_features):
            all_thresholds = np.unique(X[:,feature_idx])
            
            all_gain = [self._functional(X, y, feature_idx, threshold) for threshold in all_thresholds]
            
            threshold_idx = np.nanargmax(all_gain)
            
            if all_gain[threshold_idx] > max_gain:
                max_gain = all_gain[threshold_idx]
                best_feat_idx = feature_idx
                best_threshold = all_thresholds[threshold_idx]

        # Split data at this best feature and threshold
        mask = X[:,best_feat_idx] < best_threshold
        
        return Node(best_feat_idx, best_threshold, labels=None, # We need to cache labels only at leaf node
                             left = self._build_tree(X[mask], y[mask], depth+1), # continue to build on left side
                             right = self._build_tree(X[~mask], y[~mask], depth+1)) # continue to build on right side

    
    def fit(self, X, y):
        '''the method takes the matrix of instances X and a target vector y (numpy.ndarray objects) 
        and returns an instance of the class DecisionTree representing the decision tree trained on the 
        dataset (X, y) according to parameters set in the constructor
        '''
        
        # remember the number classes for classification task
        if self.criterion in ['gini', 'entropy']:
            self._n_classes = len(np.unique(y))
            
        self.root = self._build_tree(X, y)
        return self
        

    # predict only for one object 
    def _predict_object(self, x):
        # Traverse from root to leaf node
        node = self.root
        
        while node.labels is None:
            node = node.left if x[node.feature_idx] < node.threshold else node.right
        
        # calculate the prediction
        return self._leaf_value(node.labels)
            
        
    def predict(self, X):
        '''the method takes the matrix of instances X and returns a prediction vector;
        in case of classification, prediction for an instance  xi  falling into leaf L will be the class,
        mostly represented among instances in  L . 
        In case of regression, it will be the mean value of targets for all instances in leaf  L
        '''
        return np.array([self._predict_object(x) for x in X])
        
    
    def _predict_prob_object(self, x):
        node = self.root
        
        while node.labels is None:
            node = node.left if x[node.feature_idx] < node.threshold else node.right
            
        # calculate the probability of each class
        # i.e number of labels of class k / total number of labels
        return [len( node.labels[node.labels == k] ) / len(node.labels)
            for k in range(self._n_classes)]
        
    
    def predict_proba(self, X):
        '''the method takes the matrix of instances X and returns the matrix P of a size [X.shape[0] x K],
        where K is the number of classes and  Pij  is the probability of an instance in  i -th row of X 
        to belong to class  j∈{1,…,K}
        '''
        return np.array([self._predict_prob_object(x) for x in X])
