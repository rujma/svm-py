print(__doc__)

# Global imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# User defined files
from split_dataset import split_dataset
from model_accuracy import model_accuracy

# Program
DATA = np.array(pd.read_excel('data.xlsx'))
X = DATA[:,[0,1]]
y = DATA[:,2]

X_train, X_test, y_train, y_test = split_dataset(X, y, 0.20)

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=100)
clf.fit(X_train, y_train)
print('Support vectors: ', clf.support_vectors_)
print('Intercept: ', clf.intercept_)
print('Dual Coef: ', clf.dual_coef_)
print('Coef: ', clf.coef_)
#print('Func: ', clf.decision_function)

eval_score = clf.score(X_test, y_test)
print('Evaluation score: ', eval_score)
cross_score = cross_val_score(clf, X_train, y_train, cv=5)
print('Cross score: ', cross_score)

print('Somatorio alphas: ', np.sum(clf.dual_coef_))

print('W.X + b')
for i in range(0, len(X_test)):
    result = clf.coef_[0,0]*X_test[i,0]+clf.coef_[0,1]*X_test[i,1] + clf.intercept_
    print(result, y_test[i])

print('Alpha.K(x,x)')
y_pred = []
for i in range(0, len(X_test)):
    result = np.dot(clf.support_vectors_, X_test[i])
    result = np.dot(clf.dual_coef_, result)
    result = result + clf.intercept_
    if result > 0:
        y_pred.append(1)
    else:
        y_pred.append(0)

y_pred = np.array(y_pred)
print(y_pred)
print(y_test)

print(model_accuracy(y_test, y_pred))

# scatter training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30)
# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()