# -*- coding: utf-8 -*-

# model num 1
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest


#载入log系统随时监控中间执行过程
import sys
sys.path.append('E:\\Spyder\\competition')
from common import log_write, score_func


# This dataset is way too high-dimensional. Better do PCA:
pca = PCA(n_components=1)

# Maybe some original features where good, too?
selection = SelectKBest(k=9)

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

#Read into the dataset.
dt = pd.read_csv('E:\\Spyder\\competition\\data\\r_output\\train_2.csv', sep=',')
dt_test = pd.read_csv('E:\\Spyder\\competition\\data\\r_output\\test_2.csv', sep=',')

X_test = dt_test.as_matrix()
X_train= dt.ix[:,0:9].as_matrix()

X_all=np.concatenate((X_train, X_test),axis=0)


#do scale together. 这里，PCA和feathure selection不需要单独做同步
from sklearn import preprocessing
X_all = preprocessing.scale(X_all)

X = X_all[0:670,]
X_test = X_all[670:,]

y=dt.ix[:,9].as_matrix()

# Use combined features to transform dataset:
X_features = combined_features.fit(X, y).transform(X)


#最近邻分类器
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=3)



# Do grid search :

pipeline = Pipeline([("features", combined_features), ("KNN", KNN)])

param_grid = dict( features__pca__n_components=[0, 1, 2, 3,4],
                   features__univ_select__k=[0,1,2], 
                   KNN__n_neighbors=[3,5,7,9,10, 11]
                   )

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10, cv=10, scoring=score_func, error_score=0)
grid_search.fit(X, y)


grid_search.best_estimator_
grid_search.best_score_ 
grid_search.best_params_  
grid_search.cv_results_ 


from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(y, grid_search.predict(X))

TP = c_matrix[1,1]
TN = c_matrix[0,0]
FP = c_matrix[0,1]
FN = c_matrix[1,0]

Precision = float(TP)/(TP+FP)
Recall = float(TP)/(TP+FN)
F1 = 2*Precision*Recall/(Precision+Recall)

