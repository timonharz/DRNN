import numpy as np

class Neuron:
    def __init__(self, weights, bias, activation, importance):
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.importance = importance
    def compute_output(self, inputs):
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        return self.activation(weighted_sum)

    def update_weights(self, newWeights):
        self.weights = newWeights

    def update_bias(self, newBias):
        self.bias = newBias

    def update_importance(self, newImportance):
        self.importance = newImportance

    def to_dict(self):
        return {
            'weights' : self.weights.tolist(),
            'bias' : self.bias,
            'importance' : self.importance
        }
    def evaluate_importance(self, loss_gradient):
        self.importance += np.sum(np.abs(loss_gradient))
    @staticmethod
    def from_dict(neuron_dict):
        weigths = np.array(neuron_dict['weights'])
        bias = neuron_dict['bias']
        importance = neuron_dict['importance']
        return Neuron(neuron_dict.weights, bias, importance)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    def clone(self):
        return Neuron(self.weights.copy(), self.bias, self.activation, self.importance)