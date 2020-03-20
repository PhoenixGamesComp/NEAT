from neurons import Neuron


class GeneticMaterial:
	def __init__(self, *neurons: Neuron):
		super().__init__()
		self.genetic_data = []
		self.input_neurons = 0
		self.output_neurons = 0
		self.synapses = 0
		for neuron in neurons:
			self.genetic_data.append(neuron)

		self.__getInputNeurons()
		self.__getOutputNeurons()
		self.__getSynapses()

	def __getSynapses(self):
		for data in self.genetic_data:
			for connection in data.connections:
				if connection.status:
					self.synapses = self.synapses + 1

	def __getInputNeurons(self):
		for data in self.genetic_data:
			if data.attribute == "input":
				self.input_neurons = self.input_neurons + 1

	def __getOutputNeurons(self):
		for data in self.genetic_data:
			if data.attribute == "output":
				self.output_neurons = self.output_neurons + 1

	def AddData(self, data: Neuron):
		self.genetic_data.insert(len(self.genetic_data) - self.output_neurons, data)

	def FindSynapse(self, starting_point: int, ending_point: int):
		for connection in self.genetic_data[starting_point].connections:
			if connection.neuron.id == self.genetic_data[ending_point].id:
				return connection

		return None

	def __functionSTR(self, id):
		return {
			0: "Sigmoid",
			1: "ReLU",
			2: "TanH",
			3: "LeakyReLU",
			4: "Swish",
			5: "Input",
		}[id]

	def PrintData(self):
		for data in self.genetic_data:
			for connection in data.connections:
				print("{} *{}* -> {} *{}* ({})".format(
										data.id,
										self.__functionSTR(data.activation_function.id),
										connection.neuron.id,
										self.__functionSTR(connection.neuron.activation_function.id),
										connection.status
										))

	def PrintActiveData(self):
		for data in self.genetic_data:
			for connection in data.connections:
				if connection.status:
					print("{} *{}* -> {} *{}*".format(
											data.id,
											self.__functionSTR(data.activation_function.id),
											connection.neuron.id,
											self.__functionSTR(connection.neuron.activation_function.id),
											))
