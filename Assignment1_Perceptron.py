
# Perceptron Algorithm Code
########################################################################################################################################
import numpy as np
import sklearn 
import matplotlib.pyplot as plt
##########################################################################################################################################
inputs = 785
outputs = 10
train_size = 60000
test_size = 10000
total_epoch = 50
learning_rates = [0.1,0.01,0.001]
	plot+=1
	# Randomize Weights are generated in the range of (-0.05,0.05)
	weights = (np.random.rand(inputs, outputs) - 0.5)*(0.1)				
	# for each epoch:
	previous_accuracy = 1
	epoch = 0
	testacc_array = [ ]
	trainacc_array = [ ]
	while (1):
		current_accuracy, correct_output = testing_perceptron(train_data, train_labels, train_size)			
		print("Epoch " + str(epoch) + " :\nTraining Accuracy = " + str(current_accuracy))
		trainacc_array.append(current_accuracy)  # Array is maintained to plot the graph
		if epoch==total_epoch:
			break									
		test_accu, correct_output = testing_perceptron(test_data, test_labels, test_size)			
		previous_accuracy = current_accuracy
		epoch+=1
		i=0
		testacc_array.append(test_accu)  # Array is maintained to plot the graph
		weights = training_perceptron(weights)		# Train the network
	test_accu, correct_output = testing_perceptron(test_data, test_labels, test_size)				
	print("Test Accuracy = " + str(test_accu) + "\n\nLearning Rate = " + str(lr) + "\n\nConfusion Matrix :\n")
	print(confusion_matrix(test_labels, correct_output)) #calculating confusion matrix depicting how many correct output is classified for the testing data                   
	print("plot: ",plot)
plt.show()