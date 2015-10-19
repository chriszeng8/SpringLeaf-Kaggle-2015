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

print 'Reading stacked data...'
stacked_col = pd.read_csv('stacked_model_pred.csv', delimiter=',')
print 'Finished reading test data...'

print 'Original Training and Test set size:'
print train_data.shape, oot1_data.shape

discardNearZeroGini=True
if discardNearZeroGini:
	zero_gini_cols=['VAR_0444',
	'VAR_0460',
	'VAR_0342_UU',
	'VAR_0274_PR',
	'VAR_0274_UT',
	'VAR_0491',
	'VAR_0495',
	'VAR_0415',
	'VAR_0342_FE',
	'VAR_0159_day',
	'VAR_0274_NY',
	'VAR_0388',
	'VAR_0200_SAN ANTONIO',
	'VAR_0274_KY',
	'VAR_0305_R',
	'VAR_0342_UD',
	'VAR_0877',
	'VAR_0237_IL',
	'VAR_0276',
	'VAR_0237_MS',
	'VAR_0452',
	'VAR_0371',
	'VAR_0380',
	'VAR_0429',
	'VAR_0335',
	'VAR_0274_NV',
	'VAR_0342_BC',
	'VAR_0346',
	'VAR_0274_HI',
	'VAR_0283_R',
	'VAR_1765',
	'VAR_0237_ID',
	'VAR_0342_EB',
	'VAR_0283_F',
	'VAR_0274_OR',
	'VAR_0247',
	'VAR_0243',
	'VAR_0158_month',
	'VAR_0414',
	'VAR_0413',
	'VAR_0342_BD',
	'VAR_0274_other',
	'VAR_0499',
	'VAR_0412',
	'VAR_0342_FC',
	'VAR_0237_KS',
	'VAR_0379',
	'VAR_0107',
	'VAR_0274_MN',
	'VAR_0236_true',
	'VAR_0195',
	'VAR_0236_false',
	'VAR_0457',
	'VAR_0443',
	'VAR_1201',
	'VAR_0248',
	'VAR_0178_day',
	'VAR_0237_NY',
	'VAR_0342_DU',
	'VAR_0274_WY',
	'VAR_0274_AZ',
	'VAR_0347',
	'VAR_0237_NV',
	'VAR_0194',
	'VAR_0274_ID',
	'VAR_0274_NM',
	'VAR_0348',
	'VAR_0274_EE',
	'VAR_0502',
	'VAR_0090',
	'VAR_0524',
	'VAR_0342_CE',
	'VAR_0277',
	'VAR_0274_MA',
	'VAR_0275',
	'VAR_0237_WY',
	'VAR_0342_CF',
	'VAR_0428',
	'VAR_0498',
	'VAR_0274_IA',
	'VAR_0237_NJ',
	'VAR_0377',
	'VAR_0342_BF',
	'VAR_0349',
	'VAR_0500',
	'VAR_0378',
	'VAR_0237_DE',
	'VAR_0496',
	'VAR_0274_RI',
	'VAR_0342_DB',
	'VAR_0168_month',
	'VAR_0274_NJ',
	'VAR_0342_EA',
	'VAR_0180',
	'VAR_0181',
	'VAR_0182',
	'VAR_0269',
	'VAR_0344',
	'VAR_0157_month',
	'VAR_0392',
	'VAR_0393',
	'VAR_0342_AD',
	'VAR_0274_NE',
	'VAR_0342_AC',
	'VAR_0497',
	'VAR_0156_day',
	'VAR_0342_AB',
	'VAR_0342_CB',
	'VAR_0387',
	'VAR_0402',
	'VAR_0342_UE',
	'VAR_0200_CHICAGO',
	'VAR_0274_DE',
	'VAR_0193',
	'VAR_0176_day',
	'VAR_0350',
	'VAR_0168_day',
	'VAR_0114',
	'VAR_0342_BU',
	'VAR_0271',
	'VAR_0325_M',
	'VAR_0305_U',
	'VAR_0192',
	'VAR_0237_MT',
	'VAR_0274_SD',
	'VAR_0274_VT',
	'VAR_0130',
	'VAR_0237_NE',
	'VAR_1427',
	'VAR_0342_FA',
	'VAR_0342_BB',
	'VAR_0283_P',
	'VAR_0091',
	'VAR_0244',
	'VAR_0237_SD',
	'VAR_0274_DC',
	'VAR_0342_AF',
	'VAR_0098',
	'VAR_0138',
	'VAR_0345',
	'VAR_0274_AK',
	'VAR_0274_GS',
	'VAR_0278',
	'VAR_0523',
	'VAR_0342_EU',
	'VAR_0274_MT',
	'VAR_0342_UC',
	'VAR_0386',
	'VAR_0411',
	'VAR_0274_ND',
	'VAR_0214_other',
	'VAR_0399',
	'VAR_0342_AA',
	'VAR_0237_AR',
	'VAR_0342_UB',
	'VAR_0264',
	'VAR_0342_UF',
	'VAR_0342_BA',
	'VAR_0237_MN',
	'VAR_0342_DA',
	'VAR_0342_FB',
	'VAR_0342_CU',
	'VAR_0274_NH',
	'VAR_0325_G',
	'VAR_0459',
	'VAR_0342_UA',
	'VAR_0342_BE',
	'VAR_0437',
	'VAR_0106',
	'VAR_0237_IA',
	'VAR_0398',
	'VAR_0445',
	'VAR_0214_HRE-Home Phone-0621',
	'VAR_0214_HRE-Social Security Number-1289',
	'VAR_0214_HRE-Social Security Number-1397',
	'VAR_0214_HRE-Home Phone-0779',
	'VAR_0214_HRE-Social Security Number-2857',
	'VAR_0237_CT',
	'VAR_0397',
	'VAR_0449',
	'VAR_0463',
	'VAR_0237_AZ',
	'VAR_0237_DC',
	'VAR_0342_CA',
	'VAR_0274_CT',
	'VAR_0270',
	'VAR_0305_M',
	'VAR_0395',
	'VAR_0274_ME',
	'VAR_0342_AE',
	'VAR_0230_true',
	'VAR_0526',
	'VAR_0529',
	'VAR_0214_HRE-Social Security Number-1373',
	'VAR_0214_HRE-Social Security Number-1747',
	'VAR_0214_HRE-Social Security Number-15335',
	'VAR_0214_HRE-Social Security Number-1855',
	'VAR_0214_HRE-Social Security Number-10143',
	'VAR_0214_FSI-0005-1',
	'VAR_0230_false',
	'VAR_0396',
	'VAR_0521',
	'VAR_0191',
	'VAR_0274_RN',
	'VAR_0237_other',
	'VAR_0167_day',
	'VAR_0018',
	'VAR_0019',
	'VAR_0020',
	'VAR_0021',
	'VAR_0022',
	'VAR_0023',
	'VAR_0024',
	'VAR_0025',
	'VAR_0026',
	'VAR_0027',
	'VAR_0028',
	'VAR_0029',
	'VAR_0030',
	'VAR_0031',
	'VAR_0032',
	'VAR_0038',
	'VAR_0039',
	'VAR_0040',
	'VAR_0041',
	'VAR_0042',
	'VAR_0188',
	'VAR_0189',
	'VAR_0190',
	'VAR_0197',
	'VAR_0199',
	'VAR_0203',
	'VAR_0215',
	'VAR_0221',
	'VAR_0223',
	'VAR_0246',
	'VAR_0394',
	'VAR_0438',
	'VAR_0446',
	'VAR_0527',
	'VAR_0528',
	'VAR_0530',
	'VAR_0847',
	'VAR_1428',
	'VAR_0204_year']

use_lasso_sel_cols=True
if use_lasso_sel_cols:
	final_lasso_cols=[]
#########################################################################################################
"""
1. First, we take out all categorical variables (dummy cols), and perform stacking
2. Stacking works by training Out-of-Sample data (link: data-robot )
3. Second, we will append onto the orignal data-set
4. Take out columns with near-zero Gini
5. Perform PCA analysis to reduce dimensionality while maintaining feature space (output: Train + Test)
6. Third, run CV xgboost (done in R studio in AWS)
7. Lastly, see results
"""

### Obtain all dummy variable columns ####
dummy_regression=True
if dummy_regression:
	dummy_cols=list()
	for col in train_data:
		if len(np.unique(np.array(train_data[col]))) == 2:
			dummy_cols.append(col)
	idx = dummy_cols.index('VAR_0217_year')
	dummy_cols=dummy_cols[(idx+1):]

y_train = train_data['target'].values.astype(int)
x_train_dummy = train_data[dummy_cols].values
x_test_dummy = oot1_data[dummy_cols].values

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

#########################################################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

def ridge_dummy_regression(X,y,x_test,lambda_val=None):
	Cs=np.logspace(-1.5, 1.5, 5)
	lr = LogisticRegression(penalty='l2')
	cv_list=list()

	if not lambda_val:
		# Fit ridge to various choices of regularization parameter C to select best C
		for c in Cs:
			lr.C = c
			cv_score = cross_val_score(lr, X, y, scoring='roc_auc', cv=5)
			cv_list.append(np.mean(cv_score))

		print 'Best lambda based on Ridge Cross-Validation...'
		max_score=np.max(cv_list)
		lambda_val=Cs[cv_list.index(max_score)]
		print 1.0/lambda_val, max_score

	# Train LR with the optimized regularization parameter ###
	lr.C = lambda_val
	lr.fit(X,y)
	proba_lst = lr.predict_proba(x_test)[:,1]

	return proba_lst

### Randomly split the categorical data into halves ###
### for stacking regression ###
data_size=len(x_train_dummy)
x_train_A=x_train_dummy[0:data_size/2,:]
x_train_B=x_train_dummy[data_size/2:,:]
y_train_A=y_train[0:data_size/2]
y_train_B=y_train[data_size/2:]

print 'Size of A and B:'
print x_train_A.shape, x_train_B.shape

# print 'running part A of stacking'
# pred_train_A=ridge_dummy_regression(x_train_B,y_train_B,x_train_A,1.0/5.62)
# print 'running part B of stacking'
# pred_train_B=ridge_dummy_regression(x_train_A,y_train_A,x_train_B,1.0/31.62)
# print 'running test set of stacking'
#pred_test=ridge_dummy_regression(x_train_dummy,y_train,x_test_dummy,1.0/5.62)

##########################################################################################################
### Now we have the dummy variables predicted, append it onto the train + test dataset ###
"""
3. Second, we will append onto the orignal data-set
4. Take out columns with near-zero Gini
"""
train_data.drop(dummy_cols+['target','Unnamed: 0'], axis=1,inplace=True)
oot1_data.drop(dummy_cols+['Unnamed: 0'], axis=1,inplace=True)

# stacked_col=pd.concat([pd.DataFrame(pred_train_A),pd.DataFrame(pred_train_B)])
# stacked_col.reset_index(drop=True, inplace=True) #Very important!! concat joins based on indices
# print 'writing stacked columns data to file...'
# stacked_col.to_csv("stacked_model_pred.csv")

train_data=pd.concat([train_data,stacked_col],axis=1)
oot1_data=pd.concat([oot1_data,pd.DataFrame(pred_test)],axis=1)

train_data.drop(list(set(zero_gini_cols)-set(dummy_cols)),axis=1,inplace=True)
oot1_data.drop(list(set(zero_gini_cols)-set(dummy_cols)),axis=1,inplace=True)


feature_names = list(train_data.columns.values)

print 'Training and Test size after stacking dummy features:'
print train_data.shape, oot1_data.shape

x_train_final=train_data.values.astype(float)
x_test_final=oot1_data.values.astype(float)
#pdb.set_trace()
#########################################################################################################
"""
5. Perform PCA analysis to reduce dimensionality while maintaining feature space (output: Train + Test)
"""
from sklearn.preprocessing import StandardScaler

### Check if all values are now numeric ###
print np.isnan(x_train_final).sum() #should be zero
print np.isnan(x_test_final).sum() #should be zero

### Remove columns that have near zero variance ###
### There shouldn't be any, as we already did this in R ###
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

### Mean Standardize the columns in order to perform PCA###
### All columns should now have values, and are all numeric ###
print 'Mean standardizing training and test set...'
std_scaler = StandardScaler()
x_train_final=std_scaler.fit_transform(x_train_final)
x_test_final=std_scaler.fit_transform(x_test_final)
print 'Finished mean standardizing training and test set...'

PCA_run=False
if PCA_run:
	from sklearn.decomposition import PCA
	print 'Performing PCA to keep 97.5% variation...'
	pca=PCA(n_components=0.95)
	pca.fit(x_train_final)
	x_train_final=pca.transform(x_train_final)
	print pca.explained_variance_ratio_
	print pca.components_
	x_test_final=pca.transform(x_test_final)
	print pca.explained_variance_ratio_
	print pca.components_
	print 'PCA Completed....'

	#pdb.set_trace()

	### Add back target to train ###
	### Convert everything back to dataFrame###
	x_train_final=pd.DataFrame(x_train_final)
	x_train_final=pd.concat([pd.DataFrame(y_train),x_train_final],axis=1)
	x_test_final=pd.DataFrame(x_test_final)

LASSO_run=True
if LASSO_run:
	from sklearn.linear_model import LogisticRegression
	from sklearn.cross_validation import cross_val_score

	Cs=np.logspace(-1.5, 1.5, 10)
	lr_lasso = LogisticRegression(penalty='l1')
	cv_lasso_scores=list()

	# Fit lasso to various choices of regularization parameter C to select best C
	for c in Cs:
		lr_lasso.C = c
		cv_lasso_score = cross_val_score(lr_lasso, x_train_final, y_train, scoring='roc_auc', cv=5)
		cv_lasso_scores.append(np.mean(cv_lasso_score))

	print 'Best lambda based on Lasso Cross-Validation...'
	max_score=np.max(cv_lasso_scores)
	max_lambda_l1=Cs[cv_lasso_scores.index(max_score)]
	print 1.0/max_lambda_l1, max_score

	lr_lasso.C = max_lambda_l1
	lr_lasso.fit(x_train_final,y_train)
	print lr_lasso.coef_
	tmp_names=np.array(feature_names)
	selected_features=tmp_names[lr_lasso.coef_[0] != 0]
	print 'writing final selected features to file...'
	selected_features=pd.DataFrame(selected_features)
	selected_features.to_csv("final_selected_cols.csv")

pdb.set_trace()
test_final=pd.DataFrame(x_test_final)

### Final Check ###
print 'Training and Test size after PCA:'
print x_train_final.shape, x_test_final.shape

### create column headers ###
# feature_names=list()
# for i in range(x_test_final.shape[1]):
# 	feature_names.append('VAR_'+str(i))

print 'writing data to file...'
test_final.to_csv("test_final.csv", header=feature_names)

feature_names.insert(0,'target')
x_train_final.to_csv("train_final.csv", header=feature_names)

##########################################################################################################
### GridSearchCV for xgboost
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
