import numpy as np


class DenseLayer(object):

	def __init__(self, num_inputs, layer_size, activation_func)->None:
		self.W = np.random.standard_normal((num_inputs, layer_size))
		self.b = np.random.standard_normal(layer_size)
		self.size = layer_size
		self.activation_func = activation_func
	
	def forward(self, x)->float:
		z = np.dot(x, self.W) + self.b
		return self.activation_func(z)
	
	def get_weights()->np.array:
		return self.W
	
	def get_bias()->np.array:
		return self.b
	
	def get_size()->int:
		return self.size
	
	def set_activation_function(activation_func):
		self.activation_func = activation_func
