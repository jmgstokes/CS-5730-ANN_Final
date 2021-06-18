#Stokes, Jeff
#CS5730 - Artificial Neural Networks
#Final - Full Dataset

import numpy as np
import matplotlib.pyplot as plt

#Activation function (sigmoid):
def nonlin(x, deriv=False):
	if(deriv==True):
		return(x*(1-x))
	return(1/(1+np.exp(-x)))

#Retrieve dataset from file
data = np.loadtxt('breast_cancer_data.csv',delimiter=',')

#Separate data into features and labels
x = data[:,:-1]
y = np.array([data[:,-1]]).T


#normalize features and labels
x_max = np.amax(x,axis=0)
x = x/x_max

y_max = np.max(y)
y = y/y_max


#Seed pseudo-random number generator
np.random.seed(1)

#Synapses/weights
syn0 = 2*np.random.random((30,40))-1
syn1 = 2*np.random.random((40,1))-1

#forward propagation, training
for j in range(1000):
	#layers
	l0 = x #inputs
	l1 = nonlin(np.dot(l0,syn0)) #hidden layer 0
	l2 = nonlin(np.dot(l1,syn1)) #output


	#back propagation
	l2_error = y-l2
	
	if(j % 100) == 0:
		print('Error: '+str(np.mean(np.abs(l2_error))))

	l2_delta = l2_error*nonlin(l2,deriv=True)

	l1_error = l2_delta.dot(syn1.T)
	l1_delta = l1_error*nonlin(l1,deriv=True)

	#update weights
	syn1 += 0.0245*l1.T.dot(l2_delta)
	syn0 += 0.0245*l0.T.dot(l1_delta)


MSE = np.mean(np.power(l2_error[:,0],2))
print('MSE: ', MSE)


plt.plot(l2,'r.', label='Predicted Value')
plt.plot(y, 'b.', label='Output Value')
plt.xlabel('Instance')
plt.ylabel('Value 0 to 1')
plt.legend(loc=1)
plt.show()