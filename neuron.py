import numpy as np 



class Neuron(object):

	def __init__(self, num_inputs, activation_func)->None:
		self.W  = np.random.rand(num_inputs)
		self.b = np.random.rand(1)
		self.activation_func = activation_func
	
	def forward(self, x)->float:
		z = np.dot(x,self.w) + self.b
		return self.activation_func(z)

 	
