import numpy as np

# =============================================================================
# Supportive functions
# =============================================================================

# Classification criterion functions to measure the quality of a split.
def entropy(y):
    """entropy computes the information gain due to split
    F(X_m) = -\sum_{i = 1}^K p_i \log_2(p_i)
    """
    # compute probability of being a particular class
    P = [len(y[y==k]) / len(y) for k in np.unique(y)]
    return -1 * np.dot(P, np.log2(P))

def gini(y):
    """gini impurity to measure the quality of a split
    F(X_m) = 1 -\sum_{i = 1}^K p_i^2
    """
    # compute probability of being a particular class
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
    
    def __init__(self, feature_idx:int=0, threshold:float=0.0, labels:list=None, left=None, right=None, debug:bool=False):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.labels = labels
        self.left = left
        self.right = right
        if debug:
            print(f"threshold: {self.threshold}, labels: {self.labels}")
      

class DecisionTree():
    
    def __init__(self, 
                 max_depth:int=np.inf, 
                 min_samples_split:int=2, 
                 criterion='gini', 
                 debug:bool=False):
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.debug = debug

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
        mask = X[:,best_feat_idx] <= best_threshold
        
        return Node(best_feat_idx, best_threshold, debug=self.debug, labels=None, # We need to cache labels only at leaf node
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


class DecisionTreeClassifier(DecisionTree):
    """A decision tree classifier.

    Parameters
    ----------
    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        
    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for percentages.
           
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from mllearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier()
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    """

    def __init__(self,
                 max_depth:int=np.inf, 
                 min_samples_split:int=2, 
                 criterion='gini', 
                 debug:bool=False):
        super(DecisionTreeClassifier, self).__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split, 
            criterion=criterion,
            debug=debug)


    def fit(self, X, y):
        '''the method takes the matrix of instances X and a target vector y (numpy.ndarray objects) 
        and returns an instance of the class DecisionTree representing the decision tree trained on the 
        dataset (X, y) according to parameters set in the constructor
        '''
        return super(DecisionTreeClassifier, self).fit(X,y)


class DecisionTreeRegressor(DecisionTree):
    """A decision tree regressor.

    Parameters
    ----------
    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        
    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for percentages.
           
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria are
        "mse" for the mean squared error impurity and "mad_median" for the mean median
        error impurity.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from mllearn.tree import DecisionTreeRegressor
    >>> clf = DecisionTreeRegressor()
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    """

    def __init__(self,
                 max_depth:int=np.inf, 
                 min_samples_split:int=2, 
                 criterion='mse', 
                 debug:bool=False):
        super(DecisionTreeRegressor, self).__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split, 
            criterion=criterion,
            debug=debug)


    def fit(self, X, y):
        '''the method takes the matrix of instances X and a target vector y (numpy.ndarray objects) 
        and returns an instance of the class DecisionTree representing the decision tree trained on the 
        dataset (X, y) according to parameters set in the constructor
        '''
        return super(DecisionTreeRegressor, self).fit(X,y)
