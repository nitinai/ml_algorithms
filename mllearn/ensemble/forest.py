import numpy as np
import pandas as pd

from ..tree import DecisionTree

__all__ = ["RandomForestClassifier"]

RANDOM_STATE = 17

class RandomForest():
    def __init__(self, n_estimators=10, max_depth=10, max_features=10, 
                 random_state=RANDOM_STATE, criterion='gini', debug=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.criterion = criterion
        self.debug = debug
        
        self.trees = []
        self.feat_ids_by_tree = []
        

    def fit(self, X, y):
        
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
            
        if isinstance(y, pd.Series):
            y = y.values
        
        # Validate features count
        assert(X.shape[1] >=  self.max_features)
        
        for i in range(self.n_estimators):
            np.random.seed(self.random_state+i)
            
            # select max_features features without replacement (means without repeated occurrence of feature)
            feat_indices = np.random.choice(range(X.shape[1]), size=self.max_features, replace=False)
            self.feat_ids_by_tree.append(feat_indices)
            #if self.debug: print(f'Tree {i+1} feature indices : {feat_indices}')
            
            # make a bootstrap sample (i.e. sampling with replacement, means repeated occurrence of instances)
            # of training instances. But use 'set' to just keep one out of duplicate instances. Not sure but I guess
            # this is done as those duplicates are not bringing any additional value.
            indices = list(set(np.random.choice(X.shape[0], size=X.shape[0], replace=True)))
            sample = X[indices,:][:,feat_indices] # [indices,:] - get specific rows with all columns
                                                  # [:,feat_indices] - then get specific columns keeping all rows
            #if self.debug: print(f'Tree {i+1} X shape : {sample.shape}')
            
            tree = DecisionTree(max_depth=self.max_depth, criterion=self.criterion, debug=self.debug)
            tree.fit(sample, y[indices])
            self.trees.append(tree)
            
        return self
    
    def predict_proba(self, X):
        # You code here
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
            
        prob = []
        for i, tree in enumerate(self.trees):
            p = tree.predict_proba(X[:, self.feat_ids_by_tree[i]])
            #if self.debug: print(f'Tree {i+1} prob shape : {p.shape}')
            prob.append(p)
            
        return np.mean(prob, axis=0)


class RandomForestClassifier(RandomForest):

    def __init__(self, n_estimators=10, max_depth=10, max_features=10, 
                 random_state=RANDOM_STATE, criterion='gini', debug=False):
        super(RandomForestClassifier, self).__init__(
            n_estimators=n_estimators,
            max_depth=max_depth, 
            max_features=max_features,
            random_state=random_state,
            criterion=criterion,
            debug=debug)


    def fit(self, X, y):
        return super(RandomForestClassifier, self).fit(X,y)
        