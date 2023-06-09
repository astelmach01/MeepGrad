import random
from engine import Value

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

    def parameters(self):
        return self.weights + [self.bias]

class Layer:

    def __init__(self, n_in, n_out) -> None:
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        result = []

        for neuron in self.neurons:
            result.extend(neuron.parameters())

        return result


class MLP:

    def __init__(self, n_in, n_outs: List) -> None:
        size = [n_in] + n_outs
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x[0]

    def parameters(self):
        result = []

        for layer in self.layers:
            result.extend(layer.parameters())

        return result

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0.0

    def optimize(self, X, y_true, epochs: int = 100, alpha: float = .001, debug: bool = True):

        for epoch in range(epochs):
            # forward pass
            y_preds = [self(x) for x in X]

            
            # calculate loss
            loss = sum([(y_pred - y_true) ** 2 for y_true, y_pred in zip(y_true, y_preds)])

            if debug:
                print(f"Epoch {epoch}. Loss: {loss.data}")

            # calculate gradients
            loss.backward_pass()

            # take a step in the gradient
            for parameter in self.parameters():
                parameter.data = parameter.data - alpha * parameter.grad

            # zero grad
            self.zero_grad()