from neurons import Neuron


class Synapse:
	def __init__(self, neuron: Neuron, weight: float, status: bool):
		super().__init__()
		self.neuron = neuron
		self.weight = weight
		self.prev_weight = weight
		self.status = status

	def SetActive(self, status: bool):
		self.status = status
