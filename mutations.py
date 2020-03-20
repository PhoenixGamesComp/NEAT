from genetic_materials import GeneticMaterial
from neurons import Neuron
from activation_functions import Functions
from synapses import Synapse
import random
import time
random.seed(time.time())


def my_custom_random(exclude: list):
    randInt = random.randint(0, 3)
    while randInt in exclude:
        randInt = random.randint(0, 3)
    return randInt


def random_function(x):
    return {
        0: Functions.Sigmoid(),
        1: Functions.ReLU(),
        2: Functions.TanH(),
        3: Functions.LeakyReLU(),
        4: Functions.Swish(),
    }[x]


def _neuron_mutation(dna: GeneticMaterial):
    new_neuron = Neuron(
        len(dna.genetic_data) + 1,
        random_function(random.randint(0, 4)),
        "hidden", True
    )

    starting_point = random.randint(
        0, len(dna.genetic_data) - dna.output_neurons - 1
    )

    if len(dna.genetic_data[starting_point].connections) == 1:
        ending_point = 0
    else:
        ending_point = random.randint(
            0, len(dna.genetic_data[starting_point].connections) - 1
        )

    dna.genetic_data[starting_point].connections[ending_point].SetActive(False)

    starting_synapse = Synapse(new_neuron, random.random(), True)
    dna.genetic_data[starting_point].AddSynapse(starting_synapse)

    ending_synapse = Synapse(
        dna.genetic_data[starting_point].connections[ending_point].neuron,
        random.random(), True
    )

    new_neuron.AddSynapse(ending_synapse)
    dna.synapses = dna.synapses + 2
    dna.AddData(new_neuron)


def _weight_mutation(dna: GeneticMaterial):
        random_neuron = random.randint(
            0,
            len(dna.genetic_data) - dna.output_neurons - 1
        )

        try:
            random_connection = random.randint(
                0,
                len(dna.genetic_data[random_neuron].connections) - 1
            )
            dna.genetic_data[random_neuron].connections[random_connection].weight = \
                random.random()
        except ValueError:
            _neuron_mutation(dna)


def _function_mutation(dna: GeneticMaterial):
    try:
        random_neuron = dna.genetic_data[random.randint(
            dna.input_neurons,
            len(dna.genetic_data) - dna.output_neurons - 1
        )]
        random_fun = random_function(random.randint(0, 4))
        random_neuron.activation_function = random_fun
    except ValueError:
        _neuron_mutation(dna)


def _synapse_mutation(dna: GeneticMaterial):
        starting_point = random.randint(
            0, len(dna.genetic_data) - dna.output_neurons - 1
        )

        try:
            ending_point = random.randint(
                dna.input_neurons + 1 + starting_point,
                len(dna.genetic_data) - 1
            )
            new_synapse = Synapse(
                dna.genetic_data[ending_point],
                random.random(), True
            )
            synapse_exist = dna.FindSynapse(starting_point, ending_point)
            if synapse_exist is not None:
                synapse_exist.SetActive(False)

            dna.genetic_data[starting_point].AddSynapse(new_synapse)
            dna.synapses = dna.synapses + 1
        except ValueError:
            _neuron_mutation(dna)


def mutate(dna: GeneticMaterial,
           neuron_mutation=True,
           synapse_mutation=True,
           function_mutation=True,
           weight_mutation=True
           ):

    exclude_values = []
    if neuron_mutation is False:
        exclude_values.append(0)
    if synapse_mutation is False:
        exclude_values.append(1)
    if function_mutation is False:
        exclude_values.append(2)
    if weight_mutation is False:
        exclude_values.append(3)

    mutation = my_custom_random(exclude_values)

    if mutation == 0:
        _neuron_mutation(dna)
    elif mutation == 1:
        _synapse_mutation(dna)
    elif mutation == 2:
        _function_mutation(dna)
    elif mutation == 3:
        _weight_mutation(dna)
