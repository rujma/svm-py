# Global imports
import numpy as np
import pandas as pd
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
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

abalone = pd.read_csv('abalone.csv')
# Clean data that has 0.0 height
abalone = abalone[abalone.Height > 0]
abalone.columns=['Sex','Length','Diameter','Height','Whole weight', 'Shucked weight','Viscera weight','Shell weight','Rings']
print(abalone.sample(5))
abalone.info()
print(abalone.describe())

