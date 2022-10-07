import numpy as np 
import dense_layer
import activation_functions

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
		self.layers = [dense_layer.DenseLayer(sizes[i], sizes[i+1], activation_functions.sigmoid) for i in range(len(sizes)-1)]
	
	def forward(self,x):
		for layer in self.layers:
			x = layer.forward(x)
		return x
	def predict(self, x)->int:
		classes_probabilities = self.forward(x)
		return np.argmax(classes_probabilities)
	
	def evaluate_accuracy(self, X_val, y_val):
		num_correct = 0
		for i in range(len(X_val)):
			if self.predict(X_val[i]) == y_val[i]:
				num_correct += 1
		return num_correct/len(y_val)

