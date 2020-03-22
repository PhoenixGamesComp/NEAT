from neural_networks import NeuralNetwork
from genetic_materials import GeneticMaterial


class Crossover:
	def __init__(self, *networks: NeuralNetwork, mutation_chance: float = 0.001):
		super().__init__()
		self.networks = []
		for network in networks:
			self.networks.append(network)

	def __partition(self, low: int, high: int, arr: NeuralNetwork = []):
		i = low - 1
		pivot = arr[high]
		for j in range(low, high):
			if arr[j].fitness < pivot.fitness:
				i = i + 1
				arr[i], arr[j] = arr[j], arr[i]

		arr[i + 1], arr[high] = arr[high], arr[i + 1]
		return i + 1

	def __qsort(self, low: int, high: int, arr: NeuralNetwork = []):
		if low < high:
			pi = self.__partition(low, high, arr)

			self.__qsort(low, pi - 1, arr)
			self.__qsort(pi + 1, high, arr)

	def Reproduct(self, number_of_species):
		self.__qsort(0, len(self.networks) - 1, self.networks)
		if isinstance(len(self.networks) / 2, int):
			size = len(self.networks) / 2
		else:
			size = len(self.networks) / 2 + 0.5
		if len(self.networks) == 2:
			size = 2
		old_networks = self.networks[:size]
		new_networks = []
		for x in range(0, number_of_species, 2):
			first_pivot = int(len(old_networks[x].dna.genetic_data) / 2) + 1
			second_pivot = int(len(old_networks[x + 1].dna.genetic_data) / 2)
			first_dna = old_networks[x].dna.genetic_data[:first_pivot]
			second_dna = old_networks[x + 1].dna.genetic_data[second_pivot:]
			new_neurons = first_dna + second_dna
			new_dna = GeneticMaterial(*new_neurons)
			new_dna.DisableDuplicateNeurons()
			new_network = NeuralNetwork(new_dna, old_networks[x].features, old_networks[x].labels)
			new_networks.append(new_network)
			if len(new_networks) == number_of_species:
				break

			first_dna = old_networks[x + 1].dna.genetic_data[:second_pivot]
			second_dna = old_networks[x].dna.genetic_data[first_pivot:]
			new_neurons = first_dna + second_dna
			new_dna = GeneticMaterial(*new_neurons)
			new_dna.DisableDuplicateNeurons()
			new_network = NeuralNetwork(new_dna, old_networks[x].features, old_networks[x].labels)
			new_networks.append(new_network)

		return new_networks
