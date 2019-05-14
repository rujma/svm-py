# Global imports
import sys
import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from dataclasses import dataclass
# User defined files
from split_dataset import split_dataset
from model_accuracy import model_accuracy
from scale_data import scale_data
from kernel import kernel_linear
from kernel import kernel_poly
from kernel import kernel_rbf
from save_model import *
# Ignore warnings
#np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
import warnings
warnings.filterwarnings('ignore')
# Start dataset analysis
abalone = pd.read_csv('abalone.csv', skipinitialspace=True)
abalone.columns = abalone.columns.str.replace(' ','')
# Clean data that has 0.0 height and the outliers that may cause skewness
abalone = abalone[abalone.Height > 0]
abalone = abalone[abalone.Height < 0.4]
abalone = abalone[abalone.Height > 0.019]
abalone = abalone[abalone.Visceraweight < 0.6]
abalone = abalone[abalone.Shellweight < 1]
abalone = abalone[abalone.Wholeweight < 2.75]
abalone = abalone[abalone.Shuckedweight < 1.3]
# Plot all data into subsets
#sns.set(style="ticks", color_codes=True)
#sns.pairplot(abalone, vars=abalone.columns[1:-1],hue='Rings')
#plt.show()
# Categorize Rings feature into young (0) and adult (1)
abalone['Rings'] = pd.cut(abalone['Rings'], [0,9,abalone['Rings'].max()], labels = [0,1])
# Split dataset for classification
y = abalone['Rings']
X = abalone.drop(['Rings', 'Sex', 'Diameter', 'Wholeweight', 'Visceraweight', 'Shuckedweight'], axis = 1)
X_train, X_test, y_train, y_test = split_dataset(X, y, 0.20)
# Scale data - Training isolated from test
X_train = scale_data(X_train, type_of_scale='Standard')
X_test = scale_data(X_test, type_of_scale='Standard')
# Save test set to csv file
#X_test_frame = pd.DataFrame(data=X_test, columns=X.columns)
#y_test_frame = pd.DataFrame(data=np.array(y_test), columns=['Rings'])
#test_data = pd.concat([X_test_frame, y_test_frame], axis=1)
#test_data.to_csv('test_data_svm.csv', sep=',')
# Prepare for SVM
# Get model parameter via user input
print('Model training\nKernel: auto, linear, poly or rbf\nC, Gamma, Degree and Coef parameters')
Kernel = input("Kernel:")
if Kernel == 'auto':
    svmParams = {'kernel':['linear','poly','rbf'], 'C':[0.1,1,10],'gamma':[1,2,3], 'degree':[2,3]}
    svmModel = GridSearchCV(svm.SVC(), svmParams, cv=5, n_jobs = -1)
    # Train the model
    svmModel.fit(X_train, y_train)
    print('Parameters after fit: ', svmModel.best_params_) 
else:
    C = float(input("C:"))
    Gamma = 1
    Degree = 1
    Coef = 1
    if Kernel == 'poly':
        Gamma = float(input("Gamma:"))
        Degree = int(input("Degree:"))
        Coef = int(input("Coef:"))
    if Kernel == 'rbf':
        Gamma = float(input("Gamma:"))
    svmModel = svm.SVC(kernel=Kernel, C=C, gamma=Gamma, degree=Degree, coef0=Coef)
    # Train the model
    svmModel.fit(X_train, y_train)
    print('Dual coef matrix: ', svmModel.dual_coef_.shape)
    print('Support vector matrix: ', svmModel.support_vectors_.shape)
# Evaluate model accuracy
cross_score = cross_val_score(svmModel, X_train, y_train, cv=5)
print('Cross score: ', cross_score)
# Infer about the test data using scikit-learn built in predict function
y_pred = np.array(svmModel.predict(X_test))
y_test = np.array(y_test)
accuracy, recall, precision = model_accuracy(y_test, y_pred)
F1 = 2 * (precision * recall) / (precision + recall)
print('Model accuracy and F1 score: ', accuracy, F1)
if Kernel != 'auto':
    # Size reduction to ease deployment on FPGA
    SV = np.float16(np.around(svmModel.support_vectors_, decimals=3))
    Alphas = svmModel.dual_coef_
    Bias = np.float16(np.around(svmModel.intercept_, decimals=3))
    X_test = np.float16(np.around(X_test, decimals=3))
# Evaluate models performance with different manually implemented kernels
# Linear Kernel
#y_pred_looped = kernel_linear(SV, Alphas, Bias, X_test)
#accuracy, recall, precision = model_accuracy(y_test, y_pred_looped)
#F1 = 2 * (precision * recall) / (precision + recall)
#print('Model accuracy and F1: ', accuracy, F1)

# Poly Kernel
#y_pred_looped = kernel_poly(SV, Alphas, Bias, svmParams['gamma'], svmParams['degree'], svmParams['coeff0'], X_test)
#accuracy, recall, precision = model_accuracy(y_test, y_pred_looped)
#F1 = 2 * (precision * recall) / (precision + recall)
#print('Model accuracy and F1: ', accuracy, F1)

# RBF Kernel
#y_pred_looped = kernel_rbf(SV, Alphas, Bias, svmParams['gamma'], X_test)
#accuracy, recall, precision = model_accuracy(y_test, y_pred_looped)
#F1 = 2 * (precision * recall) / (precision + recall)
#print('Model accuracy and F1: ', accuracy, F1)

# Save model parameters to file
if Kernel == 'linear':
    save_model_linear(SV, Alphas, Bias)
if Kernel == 'poly':
    save_model_poly(SV, Alphas, Bias, Gamma, Degree)
if Kernel == 'rbf':
    save_model_rbf(SV, Alphas, Bias, Gamma)


