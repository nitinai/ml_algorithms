import numpy as np

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)
