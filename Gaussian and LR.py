################################################################################################################################################################
# Gaussian Naive Bayes and Logistic Regression

# CS 545 Machine Learning Assignment - 4

# Submitted By: Palak Goel

# Dated : May 18, 2017
################################################################################################################################################################
import os
import math
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

#################################################################################################################################################################
# Processing the data:                                                                                                                                          #
# Load data and shuffle                                                                                                                                         #
# Split data in train & test sets                                                                                                                               #
# Group the indices of samples by class                                                                                                                         #
# Compute mean and standard deviation for each feature w.r.t. class                                                                                             #
#################################################################################################################################################################

filter_data = shuffle(np.loadtxt('spambase.data.csv', delimiter=','))														
X_train, X_test, y_train, y_test = train_test_split(filter_data[:, np.arange(57)], filter_data[:, 57], test_size=0.50)	
indices = [np.nonzero(y_train==0)[0], np.nonzero(y_train)[0]]								
mean = np.transpose([np.mean(X_train[indices[0], :], axis=0), np.mean(X_train[indices[1], :], axis=0)])			
std = np.transpose([np.std(X_train[indices[0], :], axis=0), np.std(X_train[indices[1], :], axis=0)])			

###################################################################################################################################################################
# Experiment-1 : Classification with Gaussian Naive Bayes                                                                                                         #
# Compute prior probabilities for both classes and take their log.Find the feature indices which have zero standard deviation.                                    #
# If any such indices exist, Replace the zero standard deviation value with a small number                                                                        #
###################################################################################################################################################################

logP_class = np.log([1-np.mean(y_train), np.mean(y_train)])	
zero_std = np.nonzero(std==0)[0]			
if (np.any(zero_std)):						
	np.place(std, std==0, 0.0001)			

predict = []
for i in range(0, X_test.shape[0]):
	# Compute independent probabilities of features w.r.t. class
	P_X = np.divide(np.exp(-1*(np.divide(np.power(np.subtract(X_test[i, :].reshape(X_test.shape[1], 1), mean), 2), 2*np.power(std, 2)))), math.sqrt(2*np.pi)*std)
	# Compute class prediction for the test sample
	Class_X = np.argmax(logP_class+np.sum(np.nan_to_num(np.log(P_X)), axis=0))	
	predict.append(Class_X)
print ("Experiment-1: Gaussian Naive Bayes")
print ("Accuracy:  ",metrics.accuracy_score(predict,y_test))
print ("Precision: ",metrics.precision_score(y_test, predict))
print ("Recall:    ",metrics.recall_score(y_test, predict))
print ("Confusion Matrix:\n",metrics.confusion_matrix(y_test, predict))

###################################################################################################################################################################
# Experiment-2 : Classification with Logistic Regression                                                                                                          #
###################################################################################################################################################################

#Logistic Regression training and testing
model = LogisticRegression()
model = model.fit(X_train, y_train)
predicted = model.predict(X_test)
probs = model.predict_proba(X_test)
print(model)
print ("")
print ("Experiment-2: Logistic Regression")
print ("Accuracy: ",metrics.accuracy_score(predicted,y_test))
print ("Precision:", metrics.precision_score(y_test, predicted))
print ("Recall:   ", metrics.recall_score(y_test, predicted))
print ("Confusion Matrix:\n ",metrics.confusion_matrix(y_test, predicted))
