#Stokes, Jeff
#CS5730 - Artificial Neural Networks
#Final - Train/Test - 2 layers

import numpy as np
import matplotlib.pyplot as plt

#Activation function (sigmoid):
def nonlin(x, deriv=False):
	if(deriv==True):
		return(x*(1-x))
	return(1/(1+np.exp(-x)))

#Retrieve datasets from file
data_train = np.loadtxt('breast_cancer_training.csv',delimiter=',')
data_test = np.loadtxt('breast_cancer_testing.csv',delimiter=',')

#Separate data into features and labels
x_train = data_train[:,:-1]
y_train = np.array([data_train[:,-1]]).T

x_test = data_test[:,:-1]
y_test = np.array([data_test[:,-1]]).T


#set arrays for prediction values
predict_train = []
predict_test = []

#Seed pseudo-random number generator
np.random.seed(1)

#Synapses/weights
syn0 = 2*np.random.random((30,40))-1
syn1 = 2*np.random.random((40,20))-1
syn2 = 2*np.random.random((20,1))-1

#forward propagation, training
for j in range(1000):
	#layers
	if j <= 800:
		l0 = x_train #inputs
	else:
		l0 = x_test
	l1 = nonlin(np.dot(l0,syn0)) #hidden layer 0
	l2 = nonlin(np.dot(l1,syn1)) #hidden layer 1
	l3 = nonlin(np.dot(l2,syn2)) #output


	#back propagation & add to prediction arrays
	if j <= 800:
		l3_error = y_train-l3
		predict_train = l3
	else:
		l3_error = y_test-l3
		predict_test = l3
	
	if(j % 100) == 0:
		print('Error: '+str(np.mean(np.abs(l3_error))))

	l3_delta = l3_error*nonlin(l3,deriv=True)

	l2_error = l3_delta.dot(syn2.T)    
	l2_delta = l2_error*nonlin(l2,deriv=True)

	l1_error = l2_delta.dot(syn1.T)
	l1_delta = l1_error*nonlin(l1,deriv=True)

	#update weights
	syn2 += 0.0001*l2.T.dot(l3_delta)
	syn1 += 0.0001*l1.T.dot(l2_delta)
	syn0 += 0.0001*l0.T.dot(l1_delta)

MSE = np.mean(np.power(l3_error[:,0],2))
print('MSE: ', MSE)


plt.plot(predict_train,'r.', label='Predicted Value')
plt.plot(y_train, 'b.', label='Output Value: Train')
plt.xlabel('Instance')
plt.ylabel('Value 0 to 1')
plt.legend(loc=1)
plt.show()

plt.plot(predict_test,'r.', label='Predicted Value')
plt.plot(y_test, 'b.', label='Output Value: Test')
plt.xlabel('Instance')
plt.ylabel('Value 0 to 1')
plt.legend(loc=1)
plt.show()