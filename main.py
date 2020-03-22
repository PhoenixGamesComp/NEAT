from neurons import Neuron
from mutations import mutate
from synapses import Synapse
from activation_functions import Functions
from genetic_materials import GeneticMaterial
from neural_networks import NeuralNetwork
from fitness import CalculateFitness
from crossover import Crossover
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
	dna1 = GeneticMaterial(n1, n2, n3, n4)
	dna2 = dna1.DuplicateMaterial()
	dna1.PrintData()
	print("\n#\n")
	dna2.PrintData()
	print("\n#\n")
	for x in range(8):
		mutate(dna1)
		mutate(dna2)

	dna1.PrintActiveData()
	print("\n#\n")
	dna2.PrintActiveData()
	print("\n#\n")
	nn1 = NeuralNetwork(
		dna1,
		np.array([[0, 1, 0], [0, 0, 1]]),
		np.array([[0], [1]])
	)
	nn2 = NeuralNetwork(
		dna2,
		np.array([[0, 1, 0], [0, 0, 1]]),
		np.array([[0], [1]])
	)
	nn1.FeedForward()
	nn2.FeedForward()
	CalculateFitness(nn1)
	CalculateFitness(nn2)
	crv = Crossover(nn1, nn2)
	new_nn = crv.Reproduct(2)
	new_nn[0].dna.PrintActiveData()
	print("\n#\n")
	new_nn[1].dna.PrintActiveData()
	new_nn[0].FeedForward()
	new_nn[1].FeedForward()
	print("Network 1 error ({}) Network 1 fitness ({})".format(
		nn1.prediction_error, nn1.fitness
	))
	print("Network 2 error ({}) Network 2 fitness ({})".format(
		nn2.prediction_error, nn2.fitness
	))
	print("New Network 1 error ({}) New Network 1 fitness ({})".format(
		new_nn[0].prediction_error, new_nn[0].fitness
	))
	print("New Network 2 error ({}) New Network 2 fitness ({})".format(
		new_nn[1].prediction_error, new_nn[1].fitness
	))
