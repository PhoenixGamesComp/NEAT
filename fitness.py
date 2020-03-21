from neural_networks import NeuralNetwork
from numpy import sqrt


class CalculateFitness():
	def __init__(cls, network: NeuralNetwork):
		connection_penaly = 0.03
		connection_count = network.dna.synapses
		network.Error()
		regression_error = network.prediction_error
		network.fitness = -regression_error * sqrt(
			1 + connection_penaly * connection_count
		)
