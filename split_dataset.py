import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset ( X, y, test_split):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=21, shuffle=True)
    print('X_TRAIN:', X_train)
    print('X_TEST:', X_test)
    print('Y_TRAIN:', y_train)
    print('Y_TEST:', y_test)
    return X_train, X_test, y_train, y_test
