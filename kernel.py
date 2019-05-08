import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def kernel_linear(SV, Alphas, Bias, X_test):

    SCALING_FACTOR = 100
    print('Alpha.K(x,x) - Linear')
    y_pred_looped = []
    for i in range(0, len(X_test)):
        result = np.int16(np.float16(np.dot(SV, X_test[i])) * SCALING_FACTOR)  # This is done in SW
        result = np.int16(np.dot(Alphas, result)) / SCALING_FACTOR  # This is done in HW
        result = result + Bias
        if result > 0:
            y_pred_looped.append(1)
        else:
            y_pred_looped.append(0)
    return np.array(y_pred_looped)

# On this next kernels: Gamma = 1 / Sigma^2
def kernel_poly(SV, Alphas, Bias, Gamma, Degree, Coeff, X_test):
    
    SCALING_FACTOR = 10
    print('Alpha.K(x,x) - Poly')
    y_pred_looped = []
    for i in range(0, len(X_test)):
        result = np.float32(np.dot(SV, X_test[i]) * Gamma)
        result = np.int16(np.float32((Coeff + result)**Degree) * SCALING_FACTOR)
        result = np.int16(np.dot(Alphas, result)) / SCALING_FACTOR
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
        result = np.exp(-Gamma * result)
        result = np.dot(Alphas, result)
        result = result + Bias
        if result > 0:
            y_pred_looped.append(1)
        else:
            y_pred_looped.append(0)
    return np.array(y_pred_looped)
