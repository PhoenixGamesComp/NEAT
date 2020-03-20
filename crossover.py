from neural_networks import NeuralNetwork


class Crossover:
    def __init__(self, *networks: NeuralNetwork):
        super().__init__()
        self.networks = []
        for network in networks:
            self.networks.append(network)

    def Reproduct(self, number_of_species):
        for x in range(number_of_species):
            pass
