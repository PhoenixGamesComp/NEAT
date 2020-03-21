import sys
import numpy as np


class NeuralNetwork:
	def __init__(self, dna, features, labels):
		super().__init__()
		self.dna = dna
		self.features = features
		self.labels = labels
		self.fitness = 0
		self.prediction_error = sys.maxsize
		for x in range(self.dna.input_neurons):
			self.dna.genetic_data[x].input_value = features[:, x]

	def FeedForward(self):
		for data in self.dna.genetic_data:
			for connection in data.connections:
				data.output_value = data.activation_function.Function(
					data.input_value
				)
				connection.neuron.input_value = (
					connection.neuron.input_value
					+ data.output_value
					* connection.weight
				)

		for x in range(1, self.dna.output_neurons + 1):
			curr_neuron = self.dna.genetic_data[-x]
			curr_neuron.output_value = curr_neuron.activation_function.Function(
				curr_neuron.input_value
			)
			print(curr_neuron.output_value)

	def Error(self):
		neuron_error = []
		for x in range(1, self.dna.output_neurons + 1):
			neuron_error.append(self.dna.genetic_data[-x].output_value - self.labels[-x])

		self.prediction_error = 0.5 * np.sum(neuron_error)
		return neuron_error
