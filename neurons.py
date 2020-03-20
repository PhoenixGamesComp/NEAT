from typing import Callable


class Neuron:
	def __init__(self, id: int, activation_function: Callable,
														attribute: str, status: bool):
		super().__init__()
		self.id = id
		self.activation_function = activation_function
		self.attribute = attribute
		self.status = status
		self.connections = []
		self.input_value = 0
		self.output_value = 0

	def AddSynapse(self, synapse):
		self.connections.append(synapse)

	def RemoveSynapse(self, synapse):
		self.connections.remove(synapse)
