# Global imports
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
# User defined files
from split_dataset import split_dataset
from model_accuracy import model_accuracy
from scale_data import scale_data
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
# Start dataset analysis
abalone = pd.read_csv('abalone.csv')
# Clean data that has 0.0 height and the outliers that may cause skewness
abalone = abalone[abalone.Height > 0]
abalone = abalone[abalone.Height < 0.4]
abalone = pd.get_dummies(abalone)
# Categorize Rings feature into young (0) and adult (1)
abalone['Rings'] = pd.cut(abalone['Rings'], [0,8,abalone['Rings'].max()], labels = [0,1])
#print(abalone.head(10))
# Split dataset for classification
y = abalone['Rings']
X = abalone.drop(['Rings', 'Sex_F', 'Sex_I', 'Sex_M'], axis = 1)
X = scale_data(X, type_of_scale = 'Standard')
X_train, X_test, y_train, y_test = split_dataset(X, y, 0.20)
# Prepare for SVM
#svmParams = {'kernel':['linear', 'poly', 'rbf'], 'C':[0.1,1,10,100],'gamma':[0.01,0.1,0.5,1,2]}
#svmModel = GridSearchCV(svm.SVC(), svmParams, cv=5)
#svmModel = svm.SVC(kernel='rbf', gamma=1, C=100)
svmModel = svm.SVC(kernel='linear', C=100)
print('Parameters after fit: ')
print(svmModel.fit(X_train, y_train))
print('Support vector matrix: ', svmModel.support_vectors_.shape)
print('Support vectors: ', svmModel.support_vectors_)
print('Intercept: ', svmModel.intercept_)
print('Dual coef matrix: ', svmModel.dual_coef_.shape)
print('Dual Coef: ', svmModel.dual_coef_)
# Evaluate model accuracy
cross_score = cross_val_score(svmModel, X_train, y_train, cv=5)
print('Cross score: ', cross_score)
y_pred = np.array(svmModel.predict(X_test))
y_test = np.array(y_test)
print('Model accuracy and recall: ', model_accuracy(y_test, y_pred))
