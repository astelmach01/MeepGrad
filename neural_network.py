from engine import Value
import random 
from typing import List

class Neuron:

    def __init__(self, n_weights) -> None:
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_weights)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        total = 0

        for weight, value in zip(self.weights, x):
            total += weight * value

        total += self.bias
        return total.tanh()

class Layer:

    def __init__(self, n_in, n_out) -> None:
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]


class MLP:

    def __init__(self, n_in, n_outs: List) -> None:
        size = [n_in] + n_outs
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x[0]