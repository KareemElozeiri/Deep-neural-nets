import numpy as np


def step(x)->int:
	if x<=0:
		return 0
	return 1

def sigmoid(x)->float:
	return 1/(1+np.exp(-x))

def relu(x)->float:
	return max(0,x)

def tanh(x)->float:
	return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


