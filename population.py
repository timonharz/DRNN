from neuron import Neuron
import numpy as np

class Population:
    def __init__(self, population_size, input_size, output_size, initial_neurons, layer_structure):
        self.population_size = population_size
        self.neurons = []
        self.input_size = input_size
        self.output_size = output_size
        self.initial_neurons = initial_neurons
        self.layer_structure = layer_structure

        self.initialize_population()

    def initialize_population(self):
        # Initialize the population with random neurons
        for _ in range(self.population_size):
            neuron = self.generate_random_neurons(input_size=self.input_size)
            self.neurons.append(neuron)
    def generate_random_neurons(self, input_size):
        #Generate a totally random neuron with random parameters
        weights = np.random.randn(input_size)
        bias = np.random.randn()
        importance = np.random.randn()
        activation = Neuron.sigmoid
        neuron = Neuron(weights, bias, activation, importance)
        return neuron

    def compute_outputs(self, inputs):
        # Compute outputs of all neurons in the population
        layer_outputs = [inputs]

        for layer in self.layer_structure:
            layer_inputs = layer_outputs[-1]
            layer_neurons = self.get_neurons_for_layer(layer)
            layer_outputs.append(self.compute_layer_outputs(layer_neurons, layer_inputs))

        return layer_outputs[-1]

    def compute_layer_outputs(self, layer_neurons, inputs):
        outputs = []
        for neuron in layer_neurons:
            output = neuron.compute_output(inputs)
            outputs.append(output)
        return outputs

    def get_neurons_for_layer(self, layer):
        neurons = []
        if layer == 0:
            neurons = self.neurons
        elif layer > 0:
            neurons = self.neurons[layer - 1 : self.layer_structure[layer  -1] + layer - 1]
        return neurons

    def update_weights(self, inputs, outputs, targets, learning_rate):
        # Update weights of neurons based on inputs, outputs, and targets using gradient descent
        layer_errors = [targets - outputs[-1]]

        for layer in reversed(range(len(self.layer_structure))):
            layer_neurons = self.get_neurons_for_layer(layer)
            layer_inputs = outputs[layer]

            layer_errors.append(self.compute_layer_errors(layer_neurons, layer_errors[-1]))

            for neuron, error in zip(layer_neurons, layer_errors[-1]):
                gradient = error * neuron.derivative_output()
                delta_weights = learning_rate * gradient * layer_inputs
                new_weights = neuron.weights + delta_weights
                neuron.update_weights(new_weights)
    def get_layer_errors(self, layer_neurons, next_layer_errors):
        # Compute error of neurons in a specific layer
        errors = []
        for neuron, error in zip(layer_neurons, next_layer_errors):
            weights = np.array([n.weights for n in neuron.next_neurons])
            layer_errors = np.dot(error, weights)
            errors.append(layer_errors)
        return errors

    def dynamic_modification(self, importance_threshold, add_neuron_prob, remove_neuron_prob):
        # Perform dynamic modification of the population
        self.reproduce(importance_threshold, add_neuron_prob, remove_neuron_prob)

        # Update neuron importance based on performance evaluation
        self.evaluate_importance()

    def reproduce(self, importance_threshold, add_neuron_prob, remove_neuron_prob):
        # Reproduce new neurons based on importance or other criteria
        offspring_neurons = []

        # Add new neurons based on importance or other criteria
        for neuron in self.neurons:
            if neuron.importance > importance_threshold:
                offspring_neurons.append(neuron.clone())

        # Remove neurons based on probability
        remaining_neurons = []
        for neuron in self.neurons:
            if np.random.random() > remove_neuron_prob:
                remaining_neurons.append(neuron)

        self.neurons = remaining_neurons + offspring_neurons

        # Add additional neurons with a certain probability
        while len(self.neurons) < self.population_size:
            if np.random.random() < add_neuron_prob:
                # Determine the layer for the new neuron
                new_neuron_layer = np.random.randint(len(self.layer_structure))

                # Determine the input size for the new neuron
                if new_neuron_layer == 0:
                    input_size = self.input_size
                else:
                    input_size = self.layer_structure[new_neuron_layer - 1]

                neuron = self.generate_random_neuron(input_size=input_size)
                self.neurons.append(neuron)
            else:
                break

    def evaluate_importance(self):
        # Evaluate the importance of neurons based on loss gradients
        for neuron in self.neurons:
            neuron.update_importance(neuron.evaluate_importance(0.8))
    def reproduce(self):
        pass
    def evaluate_fitness(self, fitness_function, data):
        # Evaluate the fitness of each neuron in the population using a fitness function
        for network in self.networks:
            fitness = fitness_function(network, data)
            network.update_fitness(fitness)


