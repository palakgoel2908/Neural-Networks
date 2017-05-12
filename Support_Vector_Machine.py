# Support Vector Machine

# CS 545 Machine Learning Assignment - 3

# Submitted By: Palak Goel

# Dated : May 10, 2017
################################################################################################################################################
import os
import random

import numpy as np

import sklearn
from sklearn.utils import shuffle
from sklearn.svm import *
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn import metrics

import matplotlib.pyplot as plt

#################################################################################################################################################
# Loading the file and processinf the data :                                                                                                    #
#################################################################################################################################################
file_load = shuffle(np.loadtxt('spambase.data.csv', delimiter=','))
X_train, X_test, y_train, y_test = train_test_split(file_load[:, np.arange(57)], file_load[:, 57], test_size=0.50)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
model = SVC(kernel='linear').fit(X_train, y_train)

##################################################################################################################################################
# Experiment 1 
# SVM Learner on train and test data. 
##################################################################################################################################################

predicted_y = model.predict(X_test)

print('\nExperiment 1 :\nAccuracy= '+str(accuracy_score(y_test, predicted_y))+'\nPrecision= '+str(precision_score(y_test, predicted_y)) +'\nRecall= '+str(recall_score(y_test, predicted_y)))

y_score = model.decision_function(X_test)

false_rate, true_rate, thresholds = roc_curve(y_test, y_score)

auc = roc_auc_score(y_test, y_score)

# Plotting the AUC graph:
plt.figure(figsize=(8, 7.5), dpi=100)
plt.plot(false_rate, true_rate, color='blue', label='ROC Curve\n(area under curve = %f)' %auc,  lw=2)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('\nFalse Positive Rate\n', size=18)
plt.ylabel('\nTrue Positive Rate\n', size=18)
plt.title('Spambase Database Classified With Linear SVM\n', size=20)
plt.legend(loc='center right')
plt.show()

################################################################################################################################################
# Experiment 2 
# Feature selection with linear SVM 
################################################################################################################################################

accuracy = []

print('\nExperiment 2 :')

w = np.copy(model.coef_)

i = np.argmax(w)

X_train_feature = X_train[:, i].reshape(X_train.shape[0], 1)

X_test_feature = X_test[:, i].reshape(X_test.shape[0], 1)

w[0][i] = float('-Infinity')

for m in range(2, 57):
	i = np.argmax(w)
	w[0][i] = float('-Infinity')
	X_train_feature = np.insert(X_train_feature, 0, X_train[:, i], axis=1)
	X_test_feature = np.insert(X_test_feature, 0, X_test[:, i], axis=1)
	y_pred_new = model.fit(X_train_feature, y_train).predict(X_test_feature)
	accuracy.append(metrics.accuracy_score(y_test,y_pred_new))
	print('m = ' + str(m) + '\naccuracy = ' + str(accuracy_score(y_test, model.fit(X_train_feature, y_train).predict(X_test_feature))))

# Plotting Graph for the accuracy against the selected features	
plt.plot(accuracy)
plt.ylabel('accuracy')
plt.xlabel('m(number of features selected)')
plt.title('Feature Selection with Linear SVM\n', size=20)
plt.show()

############################################################################################################################################
# Experiment 3 
# Random feature selection
############################################################################################################################################

accuracy1 = []

print('\nExperiment 3 :')

for m in range(2,57):
	random_indices = np.random.choice(np.arange(57), m, replace=0)
	X_train_feature = X_train[:, random_indices]
	X_test_feature = X_test[:, random_indices]
	y_pre_new = model.fit(X_train_feature, y_train).predict(X_test_feature)
	accuracy1.append(metrics.accuracy_score(y_test,y_pre_new))
	print('m = ' + str(m) + '\naccuracy = ' + str(accuracy_score(y_test, model.fit(X_train_feature, y_train).predict(X_test_feature))))

# Plotting Graph for the accuracy against the selected features	
plt.plot(accuracy1)
plt.ylabel('accuracy')
plt.xlabel('m(number of features selected)')
plt.title('Random Feature Selection\n', size=20)
plt.show()