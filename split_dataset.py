import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset ( X, y, test_split):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=21, shuffle=True)
    return X_train, X_test, y_train, y_test
