import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def kernel_linear(SV, Alphas, Bias, X_test):

    print('Alpha.K(x,x) - Linear')
    y_pred_looped = []
    for i in range(0, len(X_test)):
        result = np.dot(SV, X_test[i])
        result = np.dot(Alphas, result)
        result = result + Bias
        if result > 0:
            y_pred_looped.append(1)
        else:
            y_pred_looped.append(0)
    return np.array(y_pred_looped)

def kernel_poly(SV, Alphas, Bias, Gamma, Degree, Coeff, X_test):
    
    print('Alpha.K(x,x) - Poly')
    y_pred_looped = []
    for i in range(0, len(X_test)):
        result = np.dot(SV, X_test[i])
        result = result / (Gamma**2)
        result = (Coeff + result)**Degree
        result = np.dot(Alphas, result)
        result = result + Bias
        if result > 0:
            y_pred_looped.append(1)
        else:
            y_pred_looped.append(0) 
    return np.array(y_pred_looped)


def kernel_rbf(SV, Alphas, Bias, Gamma, X_test):
    
    print('Alpha.K(x,x) - RBF')
    y_pred_looped = []

    for i in range(0, len(X_test)):
        result = euclidean_distances(SV, X_test[i].reshape(1, -1),  squared=True)
        result = result / (Gamma**2)
        result = np.exp(-result)
        result = np.dot(Alphas, result)
        result = result + Bias
        if result > 0:
            y_pred_looped.append(1)
        else:
            y_pred_looped.append(0)
    return np.array(y_pred_looped)
