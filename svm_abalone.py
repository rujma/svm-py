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
from decimal import *
# User defined files
from split_dataset import split_dataset
from model_accuracy import model_accuracy
from scale_data import scale_data
from kernel import kernel_linear
from kernel import kernel_poly
from kernel import kernel_rbf
# Ignore warnings
#np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
import warnings
warnings.filterwarnings('ignore')
# Start dataset analysis
abalone = pd.read_csv('abalone.csv', skipinitialspace=True)
# Clean data that has 0.0 height and the outliers that may cause skewness
abalone = abalone[abalone.Height > 0]
abalone = abalone[abalone.Height < 0.4]
# The data we have at disposal is great for predicting the Rings between 3 to 15 years
abalone = abalone[abalone.Rings < 16]
abalone = abalone[abalone.Rings > 2]
# Change Sex categorical feature to numerical
abalone['Sex'] =  abalone['Sex'].replace('I', 0)
abalone['Sex'] =  abalone['Sex'].replace('M', 1)
abalone['Sex'] =  abalone['Sex'].replace('F', 2)
# Plot all data into subsets
#sns.set(style="ticks", color_codes=True)
#sns.pairplot(abalone, vars=abalone.columns[:-1],hue='Rings')
#plt.show()
# Categorize Rings feature into young (0) and adult (1)
abalone['Rings'] = pd.cut(abalone['Rings'], [0,11,abalone['Rings'].max()], labels = [0,1])
# Plot all data into subsets after applying class transformation
#sns.set(style="ticks", color_codes=True)
#sns.pairplot(abalone, vars=abalone.columns[:-1], hue='Rings')
#plt.show()
# Data has too many decimal places
abalone['Length'] *= 1000
abalone['Diameter'] *= 1000
abalone['Height'] *= 1000
abalone['Whole weight'] *= 1000
abalone['Shucked weight'] *= 1000
abalone['Viscera weight'] *= 1000
abalone['Shell weight'] *= 1000
# Split dataset for classification
y = abalone['Rings']
X = abalone.drop(['Rings', 'Sex', 'Diameter'], axis = 1)
X = scale_data(X, type_of_scale = 'Standard')
X = np.float16(X)
#X = np.array(X) - When no scaling is applied
X_train, X_test, y_train, y_test = split_dataset(X, y, 0.20)
# Prepare for SVM
#svmParams = {'kernel':['linear', 'poly', 'rbf'], 'C':[0.1,1,10],'gamma':[0.01,0.1,1]}
#svmModel = GridSearchCV(svm.SVC(), svmParams, cv=5, n_jobs = -1)
svmParams = {'kernel': 'linear', 'C': 10, 'gamma': 0.1, 'degree': 3, 'coeff0': 1}
svmModel = svm.SVC(kernel=svmParams['kernel'], C=svmParams['C'], gamma=svmParams['gamma'], degree=svmParams['degree'], coef0=svmParams['coeff0'])
svmModel.fit(X_train, y_train)
#print('Parameters after fit: ', svmModel.best_params_)    
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
print('Model accuracy and F1 score: ', np.around(accuracy, decimals=2), np.around(F1, decimals=2))
# Size reduction to ease deployment on FPGA
MULTIPLIER = 1
SV = np.around(svmModel.support_vectors_ * MULTIPLIER, decimals=2)
Alphas = np.around(svmModel.dual_coef_, decimals=2)
Bias = np.around(svmModel.intercept_ * MULTIPLIER, decimals=2)
X_test = np.around(X_test, decimals=2)

#SV = np.int16(svmModel.support_vectors_)
#Alphas = np.int16(svmModel.dual_coef_)
#Bias = np.int16(svmModel.intercept_)
#X_test = np.int16(X_test)
#print(X_test)

# Evaluate models performance with different manually implemented kernels

# Linear Kernel
y_pred_looped = kernel_linear(SV, Alphas, Bias, X_test)
#y_pred_looped = kernel_linear(svmModel.support_vectors_, svmModel.dual_coef_, svmModel.intercept_, X_test)
accuracy, recall, precision = model_accuracy(y_test, y_pred_looped)
F1 = 2 * (precision * recall) / (precision + recall)
print('Model accuracy and F1: ', np.around(accuracy, decimals=2), np.around(F1, decimals=2))
#error = np.mean( y_pred != y_pred_looped)
#print('Difference between methods: ', error)

# Poly Kernel
#y_pred_looped = kernel_poly(svmModel.support_vectors_, svmModel.dual_coef_, svmModel.intercept_, svmParams['gamma'], svmParams['degree'], svmParams['coeff0'], X_test)
#print('Model accuracy and recall: ', model_accuracy(y_test, y_pred_looped))
#error = np.mean( y_pred != y_pred_looped)
#print('Difference between methods: ', error)

# RBF Kernel
#y_pred_looped = kernel_rbf(svmModel.support_vectors_, svmModel.dual_coef_, svmModel.intercept_, svmParams['gamma'], X_test)
#print('Model accuracy and recall: ', model_accuracy(y_test, y_pred_looped))
#error = np.mean( y_pred != np.array(y_pred_looped) )
#print('Difference between methods: ', error)

# Save model parameters to file
np.savetxt('test.out', SV, '%5.2f' ,delimiter=',')

