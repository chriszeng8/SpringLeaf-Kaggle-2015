# /User/bin/python
# coding: utf-8
import os
import csv
import numpy as np
import pandas as pd
import random
import time
import pdb

# Define parameters for this script

#145,231 total examples; 33,773 positive; 23.25% response rate
print 'Reading training data...'
train_data = pd.read_csv('train_clean.csv', delimiter=',')
print 'Finished reading training data...'

print 'Reading test data...'
oot1_data = pd.read_csv('test_clean.csv', delimiter=',')
print 'Finished reading test data...'

use_selected=False
if use_selected:
	sel_features=['VAR_0089',
	'VAR_0238',
	'VAR_0088',
	'VAR_0082',
	'VAR_0234',
	'VAR_0081',
	'VAR_0087',
	'VAR_0886',
	'VAR_0072',
	'VAR_0065',
	'VAR_0006',
	'VAR_0013',
	'VAR_0080',
	'VAR_0071',
	'VAR_0064',
	'VAR_0235',
	'VAR_1127',
	'VAR_1128',
	'VAR_0014',
	'VAR_0007',
	'VAR_0063',
	'VAR_1380',
	'VAR_0233',
	'VAR_1114',
	'VAR_0015',
	'VAR_1824',
	'VAR_1823',
	'VAR_0070',
	'VAR_1129',
	'VAR_1329',
	'VAR_1136',
	'VAR_0687',
	'VAR_1123',
	'VAR_0086',
	'VAR_1119',
	'VAR_1030',
	'VAR_1358',
	'VAR_1124',
	'VAR_0079',
	'VAR_1330',
	'VAR_0895',
	'VAR_0232_false',
	'VAR_0232_true',
	'VAR_1125',
	'VAR_1029',
	'VAR_0795',
	'VAR_1331',
	'VAR_1327',
	'VAR_0062',
	'VAR_0085',
	'VAR_0688',
	'VAR_1031',
	'VAR_0017',
	'VAR_0078',
	'VAR_0137',
	'VAR_1034',
	'VAR_1126',
	'VAR_1035',
	'VAR_0907',
	'VAR_0105',
	'VAR_0145',
	'VAR_0061',
	'VAR_0121',
	'VAR_0069',
	'VAR_1039',
	'VAR_0034',
	'VAR_1038',
	'VAR_0721',
	'VAR_0035',
	'VAR_1791',
	'VAR_0885']

#########################################################################################################
### Obtain all dummy variable columns ####
dummy_regression=True
if dummy_regression:
	dummy_cols=list()
	for col in train_data:
		if len(np.unique(np.array(train_data[col]))) == 2:
			dummy_cols.append(col)
	sel_features = dummy_cols
	idx = sel_features.index('VAR_0217_year')
	sel_features=sel_features[(idx+1):]

y_train = train_data['target'].values.astype(int)
#train_data.drop([set(train_data.columns.values)-set(sel_features)], axis=1,inplace=True)
x_train = train_data[sel_features].values
#oot1_data.drop([set(train_data.columns.values)-set(sel_features)], axis=1,inplace=True)
x_test = oot1_data[sel_features].values
features = list(train_data.columns.values)

#### Computing p=1 predictor gini scores using auc function###
from sklearn.metrics import roc_auc_score

def computeNan(df):
	results=list()
	for col in df:
		results.append(float(df[col].isnull().sum())/len(df))
	return results

def computeGinis(x, y):
	gini_list = list()

	for i in range(x.shape[1]):
		## Implementing METHOD ONE of Handling missing values: removal of rows ### (BEGIN)
		y_pred = x[:,i]
		gini_list.append(2*roc_auc_score(y, y_pred) - 1)

	return gini_list

compute_gini=False
if compute_gini:
	gini_list = computeGinis(x_train, y_train)

	print 'Writing univariate results file....'
	with open('Univariate_Results.csv','wb') as testfile:
		w=csv.writer(testfile)
		w.writerow(('Feature Name','Gini Score'))
		for i in range(len(gini_list)):
			w.writerow((features[i],gini_list[i]))
	testfile.close()
	print 'File written to disk...'

pdb.set_trace()

#########################################################################################################
from sklearn.preprocessing import StandardScaler

### Check if all values are now numeric ###
print np.isnan(x_train).sum() #should be zero
print np.isnan(x_test).sum() #should be zero

### Remove columns that have near zero variance ###
def nearZeroVar(x_array, var_threshold):
	x_train = x_array
	for i in range(x_train.shape[1]):
		temp_col = x_train[:,i]
		temp_var = np.var(temp_col)
		if temp_var <= (var_threshold*(1-var_threshold)):
			x_train=np.delete(x_train, i, axis=1)
	return x_train

#x_train=nearZeroVar(x_train, 0.8)
#x_test=nearZeroVar(x_test, 0.8)

### Mean Standardize the columns ###
### All columns should now have values, and are all numeric ###
print 'Mean standardizing training and test set...'
#std_scaler = StandardScaler()
#x_train=std_scaler.fit_transform(x_train)
#x_test=std_scaler.fit_transform(x_test)
print 'Finished mean standardizing training and test set...'

#########################################################################################################
### Perform 5-fold CV to choose parameters for the LR ###
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

ridge_dummy_regression=True
if ridge_dummy_regression:
	Cs=np.logspace(-1.5, -0.5, 10)
	lr = LogisticRegression(penalty='l2')
	cv_list=list()

	# Fit ridge to various choices of regularization parameter C to select best C
	for c in Cs:
		lr.C = c
		cv_score = cross_val_score(lr, x_train, y_train, scoring='roc_auc', cv=5)
		cv_list.append(np.mean(cv_score))

	print 'Best lambda based on Ridge Cross-Validation...'
	max_score=np.max(cv_list)
	max_lambda_l2=Cs[cv_list.index(max_score)]
	print 1.0/max_lambda_l2, max_score

	"""
	Best lambda based on Ridge Cross-Validation...
	11.3646366639 0.668190884376
	"""

	# reverse the score because C here is actually the inverse of the regularization parameter
	# the smaller the C, the bigger the regularization
	cv_list.reverse()

	import matplotlib.pyplot as plt
	plt.figure(figsize=(8, 6))
	plt.semilogx(Cs, cv_list)
	plt.ylabel('Area Under Curve')
	plt.xlabel('lambda')
	plt.legend(['5-Fold Cross-Validation'], loc='best', shadow=True)
	plt.axvline(1.0/max_lambda_l2, linestyle='--', color='.5')
	plt.show()

	# Train LR with the optimized regularization parameter ###
	lr.C = max_lambda_l2
	lr.fit(x_train,y_train)
	proba_lst = lr.predict_proba(x_train)[:,1]
	proba_lst_test=lr.predict_proba(x_test)[:,1]

	print 'Writing Ridge Prediction results file....'
	with open('Ridge_Dummy_Predictions.csv','wb') as testfile:
		w=csv.writer(testfile)
		w.writerow(('Pred_Proba_train', 'Pred_Proba_test'))
		for i in range(len(proba_lst)):
			w.writerow((proba_lst[i], proba_lst_test[i]))
	testfile.close()
	print 'File written to disk...'

lasso_select=False
if lasso_select:
	Cs=np.logspace(-2, 2, 10)
	lr = LogisticRegression(penalty='l1')
	cv_list=list()

	X = x_train
	y = y_train
	# Fit ridge to various choices of regularization parameter C to select best C
	for c in Cs:
		lr.C = c
		cv_score = cross_val_score(lr, X, y, scoring='roc_auc', cv=5)
		cv_list.append(np.mean(cv_score))

	print 'Best lambda based on Ridge Cross-Validation...'
	max_score=np.max(cv_list)
	max_lambda_l2=Cs[cv_list.index(max_score)]
	print 1.0/max_lambda_l2, max_score

	# reverse the score because C here is actually the inverse of the regularization parameter
	# the smaller the C, the bigger the regularization
	cv_list.reverse()

	import matplotlib.pyplot as plt
	plt.figure(figsize=(8, 6))
	plt.semilogx(Cs, cv_list)
	plt.ylabel('Area Under Curve')
	plt.xlabel('lambda')
	plt.legend(['5-Fold Cross-Validation'], loc='best', shadow=True)
	plt.axvline(1.0/max_lambda_l1, linestyle='--', color='.5')
	plt.show()

	# Train LR with the optimized regularization parameter ###
	lr.C = max_lambda_l2
	lr.fit(x_train,y_train)
	coef_lst=lr.coef_

	print 'Writing LASSO results file....'
	with open('LASSO_Results.csv','wb') as testfile:
		w=csv.writer(testfile)
		w.writerow(('Feature Name','LASSO Coef'))
		for i in range(len(gini_list)):
			w.writerow((features[i],coef_lst[i]))
	testfile.close()
	print 'File written to disk...'

pdb.set_trace()
##########################################################################################################
# ### Tree Based Importance Features ####
# >>> from sklearn.ensemble import ExtraTreesClassifier
# >>> from sklearn.datasets import load_iris
# >>> iris = load_iris()
# >>> X, y = iris.data, iris.target
# >>> X.shape
# (150, 4)
# >>> clf = ExtraTreesClassifier()
# >>> X_new = clf.fit(X, y).transform(X)
# >>> clf.feature_importances_  
# array([ 0.04...,  0.05...,  0.4...,  0.4...])
# >>> X_new.shape               
# (150, 2)

##########################################################################################################
### GridSearchCV for GradientBoostingClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.grid_search import GridSearchCV

# param_grid = {'learning_rate': [0.1,0.05,0.02,0.01],
# 				'max_depth':[6,8],
# 				'min_samples_split':[30,40,50],
# 				'max_features':[1.0,0.3,0.1]}

# est = GradientBoostingClassifier(n_rounds = 1000)
# gs_cv = GridSearchCV(est, param_grid).fit(X,y)
# #best_parameters
# gs_cv.best_params_
