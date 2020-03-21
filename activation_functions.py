import numpy as np


class Functions:
	def __init__():
		pass

	class Sigmoid:
		def __init__(self):
			super().__init__()
			self.id = 0

		def Function(self, x):
			return 1 / (1 + np.exp(-x))

		def Deriv(self, x):
			return x * (1 - x)

	class ReLU:
		def __init__(self):
			super().__init__()
			self.id = 1

		def Function(self, x):
			x = np.where(x > 0, x, 0)
			return x

		def Deriv(self, x):
			x = np.where(x > 0, 1, 0)
			return x

	class TanH:
		def __init__(self):
			super().__init__()
			self.id = 2

		def Function(self, x):
			return np.tanh(x)

		def Deriv(self, x):
			return 1 - (np.tanh(x) * np.tanh(x))

	class LeakyReLU:
		def __init__(self):
			super().__init__()
			self.id = 3

		def Function(self, x):
			x = np.where(x > 0, x, x * 0.01)
			return x

		def Deriv(self, x):
			x = np.where(x > 0, 1, 0.01)
			return x

	class Swish:
		def __init__(self):
			super().__init__()
			self.id = 4

		def Function(self, x):
			return x * (1 + np.exp(-x))

		def Deriv(self, x):
			numerator = np.exp(-x) * (x + 1) + 1
			denominator = (1 + np.exp(-x)) * (1 + np.exp(-x))
			return numerator / denominator

	class Input:
		def __init__(self):
			super().__init__()
			self.id = 5

		def Function(self, x):
			return x

		def Deriv(self, x):
			return x
