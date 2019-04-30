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
from sklearn.model_selection import GridSearchCV
from dataclasses import dataclass
# User defined files
from split_dataset import split_dataset
from model_accuracy import model_accuracy
from scale_data import scale_data
from kernel import kernel_linear
from kernel import kernel_poly
from kernel import kernel_rbf
# Ignore warnings
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
import warnings
warnings.filterwarnings('ignore')
# Start dataset analysis
abalone = pd.read_csv('abalone.csv')
# Clean data that has 0.0 height and the outliers that may cause skewness
abalone = abalone[abalone.Height > 0]
abalone = abalone[abalone.Height < 0.4]
#abalone = abalone[abalone.Sex != 'I']
abalone = pd.get_dummies(abalone)
print('Data shape: ', abalone.shape)
# Categorize Rings feature into young (0) and adult (1)
abalone['Rings'] = pd.cut(abalone['Rings'], [0,8,abalone['Rings'].max()], labels = [0,1])
#sns.set(style="ticks", color_codes=True)
#sns.pairplot(abalone, hue='Rings')
#plt.show()
# Split dataset for classification
y = abalone['Rings']
X = abalone.drop(['Rings'], axis = 1)
X = scale_data(X, type_of_scale = 'Standard')
X_train, X_test, y_train, y_test = split_dataset(X, y, 0.20)
# Prepare for SVM
#svmParams = {'kernel':['linear', 'poly'], 'C':[0.1,1,10],'gamma':[0.01,0.1,1]}
#svmModel = GridSearchCV(svm.SVC(), svmParams, cv=5, n_jobs = -1)
svmParams = {'kernel': 'rbf', 'C': 1, 'gamma': 1, 'degree': 3, 'coeff0': 1}
svmModel = svm.SVC(kernel=svmParams['kernel'], C=svmParams['C'], gamma=svmParams['gamma'], degree=svmParams['degree'], coef0=svmParams['coeff0'])
svmModel.fit(X_train, y_train)
#print('Parameters after fit: ')
#print(svmModel.best_params_)
#print('Dual coef matrix: ', svmModel.dual_coef_.shape)
#print('Support vector matrix: ', svmModel.support_vectors_.shape)
#np.savetxt('test.out', np.array(svmModel.support_vectors_), '%5.5f' ,delimiter=',')
#print('Input matrix: ', X_test.shape)
# Evaluate model accuracy
cross_score = cross_val_score(svmModel, X_train, y_train, cv=5)
print('Cross score: ', cross_score)
y_pred = np.array(svmModel.predict(X_test))
y_test = np.array(y_test)
print('Model accuracy and recall: ', model_accuracy(y_test, y_pred))


# Evaluate model accuracy with size reduction
#y_pred_looped = kernel_linear(svmModel.support_vectors_, svmModel.dual_coef_, svmModel.intercept_, X_test)
#print('Model accuracy and recall: ', model_accuracy(y_test, y_pred_looped))
#error = np.mean( y_pred != y_pred_looped)
#print('Difference between methods: ', error)

# Evaluate model accuracy with size reduction
#y_pred_looped = kernel_poly(svmModel.support_vectors_, svmModel.dual_coef_, svmModel.intercept_, svmParams['gamma'], svmParams['degree], svmParams['coeff0'], X_test)
#print('Model accuracy and recall: ', model_accuracy(y_test, y_pred_looped))
#error = np.mean( y_pred != y_pred_looped)
#print('Difference between methods: ', error)

# Evaluate model accuracy with size reduction
y_pred_looped = kernel_rbf(svmModel.support_vectors_, svmModel.dual_coef_, svmModel.intercept_, svmParams['gamma'], X_test)
print('Model accuracy and recall: ', model_accuracy(y_test, y_pred_looped))
error = np.mean( y_pred != np.array(y_pred_looped) )
print('Difference between methods: ', error)

