import numpy as np 
from dense_layer import DenseLayer
from activation_functions import sigmoid
class SimpleNN(object):
	"""
		Args:
			num_inputs (int) : The input vector size
			num_outputs (int) : classification classes no. (output vector size)
			hidden_layers_sizes (list) : list of integers representing the hidden layers's sizes
		Attributes:
			layers (list): the list of layers forming this NN
	"""
	def __init__(self, num_inputs, num_outputs, hidden_layers_sizes=(64,32)):
		super().__init__()
		#building the NN layers
		sizes = [num_inputs, *hidden_layers_sizes, num_outputs]
		self.layers = [DenseLayer(sizes[i], sizes[i+1], sigmiod) for i in range(len(sizes)-1)]
	
	def forward(self,x):
		for layer in self.layers:
			x = layer.forward(x)
		return x
	def predict(self, x):
		classes_probabilities = self.forward(x)
		return np.argmax(estimations)

