from neurons import Neuron
from mutations import mutate
from synapses import Synapse
from activation_functions import Functions
from genetic_materials import GeneticMaterial
from neural_networks import NeuralNetwork
from fitness import CalculateFitness
import numpy as np
import random
import time
random.seed(time.time())

if __name__ == "__main__":
	n1 = Neuron(1, Functions.Input(), "input", True)
	n2 = Neuron(2, Functions.Input(), "input", True)
	n3 = Neuron(3, Functions.Input(), "input", True)
	n4 = Neuron(4, Functions.Sigmoid(), "output", True)
	s1 = Synapse(n4, random.random(), True)
	s2 = Synapse(n4, random.random(), True)
	s3 = Synapse(n4, random.random(), True)
	n1.AddSynapse(s1)
	n2.AddSynapse(s2)
	n3.AddSynapse(s3)
	dna = GeneticMaterial(n1, n2, n3, n4)
	dna.PrintData()
	print("\n#\n")
	for x in range(20):
		mutate(dna)
	dna.PrintData()
	nn = NeuralNetwork(
		dna,
		np.array([[0, 1, 0], [0, 0, 1]]),
		np.array([[0], [1]])
	)
	nn.FeedForward()
	CalculateFitness(nn)
	print("Network error ({}) Network fitness ({})".format(nn.prediction_error, nn.fitness))
