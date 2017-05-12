#######################################################################################################################################

# Neural Network 

# CS 545 Machine Learning Assignment - 2

# Submitted By: Palak Goel

# Dated : April 28, 2017

########################################################################################################################################
# Libraries used such as numpy for many easy calculations, sklearn for calculating accuracy, random for generating random weights and  #
# matplotlib for plotting graphs.                                                                                                      #
########################################################################################################################################

import numpy 

import sklearn 
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import os

import random

##########################################################################################################################################
# Global variables which are total inputs, total outputs, size of training dataset, size of testing dataset, total epoches and a list    # 
# of learning rates.                                                                                                                     #
##########################################################################################################################################

inputs = 785

outputs = 10

train_size = 60000

test_size = 10000

total_epoch = 50

plot=0

learning_rates = 0.1

total_hidden_units = [20, 50, 100]

alphas = [0, 0.25, 0.5]

##############################################################################################################################################
# Loading the dataset from CSV file.                                                                                                         #
# First column from the file is taken as labels which are the target values.                                                                 #
# Rest of the data is taken as dataset for neural network. This data contains different values till 255. Therefore, scaled by dividing by 255#
##############################################################################################################################################
def file_load(filename):

	data = numpy.loadtxt(filename, delimiter=',')

	dataset = numpy.insert(data[:, numpy.arange(1, inputs)]/255, 0, 1, axis=1)

	labels = data[:, 0]

	return dataset, labels

########################################################################################################################################
# Training and Testing Perceptron by reading data from CSV files                                                                       #
########################################################################################################################################
print("Training Perceptron.....\n")
train_data, train_labels = file_load('mnist_train.csv')

print("Testing Perceptron......\n")
test_data, test_labels = file_load('mnist_test.csv')

########################################################################################################################################
# Sigmoid Activation Function:                                                                                                         #
# σ(z) = 1/1+exp(-z)                                                                                                                   #
########################################################################################################################################
def sigmoid(z):
	return 1/(1 + numpy.exp(-z))

########################################################################################################################################
# Forward Propogation to compute the hidden layer units and output units after feeding the input image.                                #                                                                                                                                                                                                                      
########################################################################################################################################
def forward_propogation(data, weight_inputtohidden, weight_hiddentooutput):
	input_array = numpy.reshape(data, (1, inputs))
	
	hidden_array = sigmoid(numpy.dot(input_array, weight_inputtohidden))
	hidden_array[0][0] = 1
	
	output_array = sigmoid(numpy.dot(hidden_array, weight_hiddentooutput))
	
	return input_array, hidden_array, output_array

########################################################################################################################################
# Back Propogation to update the weights from input to hidden layer and hidden to output layer.                                        #            
# Step1: Compute error terms                                                                                                           #
#        For each output unit k, calculate error term Δk:                                                                              #
#           Δk = output_unit_k * ( 1 - output_unit_k ) * (target_unit_k - output_unit_k)                                               #
#        For each hidden unit j, calculate error term Δj:                                                                              #
#           Δj = hidden_unit_j * ( 1 - hidden_unit_j ) * summation of (weight to output units * Δk )                                   # 
# Step2: Update Weights                                                                                                                #
########################################################################################################################################
def sub_sigmoid(z):
	return z*(1-z)

def back_propogation(error, input_array, hidden_array, output_array, weight_hiddentooutput, weight_inputtohidden, delta_previous_weight_hiddentooutput, delta_previous_weight_inputtohidden, momentum):
	
	delta_k = sub_sigmoid(output_array)*error			#Step1					
	delta_j = sub_sigmoid(hidden_array)*numpy.dot(delta_k, numpy.transpose(weight_hiddentooutput))		
	
	delta_weight_hiddentooutput = (learning_rates*numpy.dot(numpy.transpose(hidden_array), delta_k)) + (momentum*delta_previous_weight_hiddentooutput)    #Step2
	delta_weight_inputtohidden = (learning_rates*numpy.dot(numpy.transpose(input_array), delta_j)) + (momentum*delta_previous_weight_inputtohidden)
	weight_hiddentooutput += delta_weight_hiddentooutput
	weight_inputtohidden += delta_weight_inputtohidden
	
	return weight_hiddentooutput, weight_inputtohidden, delta_weight_hiddentooutput, delta_weight_inputtohidden

############################################################################################################################################
# Perceptron is trained by feeding the input, calculating the output , checking the error  and then updating the weights for each input of #
# training dataset.                                                                                                                        #
############################################################################################################################################

def Train_network(weight_hiddentooutput, weight_inputtohidden, delta_previous_weight_hiddentooutput, delta_previous_weight_inputtohidden, momentum):

	for i in range(0, train_size):
		
		input_array, hidden_array, output_array = forward_propogation(train_data[i, :], weight_inputtohidden, weight_hiddentooutput)			
		
		target_array = numpy.insert((numpy.zeros((1, outputs-1)) + 0.0001), int(train_labels[i]), 0.9999)								 
		
		weight_hiddentooutput, weight_inputtohidden, delta_previous_weight_hiddentooutput, delta_previous_weight_inputtohidden = back_propogation(target_array-output_array, input_array, hidden_array, output_array, weight_hiddentooutput, weight_inputtohidden, delta_previous_weight_hiddentooutput, delta_previous_weight_inputtohidden, momentum) 	
	
	return weight_inputtohidden, weight_hiddentooutput

#############################################################################################################################################
# Checking the testing accuracy by computing the actual outputs with the predicted outputs                                                  #
############################################################################################################################################# 
def Test_network(dataset, data_labels, set_size, weight_inputtohidden, weight_hiddentooutput):
	
	correct_output = []
	
	for i in range(0, set_size):
		
		input_array, hidden_array, output_array = forward_propogation(dataset[i, :], weight_inputtohidden, weight_hiddentooutput)															
		
		correct_output.append(numpy.argmax(output_array))								# Append the predicted output to correct_output list 
	
	return accuracy_score(data_labels, correct_output), correct_output

##############################################################################################################################################	
# Neural Network                                                                                                                             #
##############################################################################################################################################
def Neural_Network(hidden_units, momentum):

	plot = 0
	# Randomize Weights are generated in the range of (-0.05,0.05)
	weight_inputtohidden = (numpy.random.rand(inputs, hidden_units) - 0.5)*0.1
	weight_hiddentooutput = (numpy.random.rand(hidden_units, outputs) - 0.5)*0.1

	# Initialize previous delta weight arrays to 0 arrays :
	delta_previous_weight_hiddentooutput = numpy.zeros(weight_hiddentooutput.shape)
	delta_previous_weight_inputtohidden = numpy.zeros(weight_inputtohidden.shape)
	
	# Arrays for graph plotting
	testacc_array = [ ]
	trainacc_array = [ ]
	plot+=1
	
	# For each epoch	
	for epoch in range(0, total_epoch):
		# Test the network on training data and compute training accuracy
		train_accu, correct_output = Test_network(train_data, train_labels, train_size, weight_inputtohidden, weight_hiddentooutput)
		
		# Test the network on testing data and compute testing accuracy
		test_accu, correct_output = Test_network(test_data, test_labels, test_size, weight_inputtohidden, weight_hiddentooutput)										
		print("Epoch " + str(epoch) + " :\nTraining Accuracy = " + str(train_accu) + "\nTest Accuracy = " + str(test_accu) + "\n")
		
		# Train the network
		weight_inputtohidden, weight_hiddentooutput = Train_network(weight_hiddentooutput, weight_inputtohidden, delta_previous_weight_hiddentooutput, delta_previous_weight_inputtohidden, momentum)
		
		# Accuracies are appended to plot on graph
		trainacc_array.append(train_accu)
		testacc_array.append(test_accu)
	
	epoch += 1
	i=0
	
	# Test the network on training data and compute training accuracy and correct output
	train_accu, correct_output = Test_network(train_data, train_labels, train_size, weight_inputtohidden, weight_hiddentooutput)	
	
	# Test the network on testing data and compute testing accuracy and correct output
	test_accu, correct_output = Test_network(test_data, test_labels, test_size, weight_inputtohidden, weight_hiddentooutput)											
	print("Epoch " + str(epoch) + " :\nTraining Accuracy = " + str(train_accu) + "\nTest Accuracy = " + str(test_accu) + "\n\nHidden Units = " + str(hidden_units) + "\nMomentum = " + str(momentum) + "\nTraining Data Size = " + str(train_size)+ "\nConfusion Matrix :")
	
	# Computing confusion matrix
	print(confusion_matrix(test_labels, correct_output))
	print("\n")
	
	# Plotting Graph to show training and testing accuracies
	plt.figure(plot)
	plt.title("Learning Rate: %s" %learning_rates)
	plt.plot(testacc_array)
	plt.plot(trainacc_array)
	plt.ylabel("Accuracy %")
	plt.xlabel("Epoch")
	plt.show()
	return

# Experiment 1 : Vary number of hidden units
#                Keeping momentum value constant 0.9
for hidden_units in total_hidden_units:
	Neural_Network(hidden_units, 0.9)																		

# Experiment 2 : Vary the momentum values
#                Keeping hidden units constant 100
for momentum in alphas:
	Neural_Network(100, momentum)																			

# Experiment 3 : Vary the number of training examples
#                 Keeping hidden units and momentum value constant 100 and 0.9
for i in range(0, 2):
	train_data, X, train_labels, Y = train_test_split(train_data, train_labels, test_size=0.50)		
	train_size = int(train_size/2)
	Neural_Network(100, 0.9)