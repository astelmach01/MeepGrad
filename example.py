from neural_network import MLP
import numpy as np

x = [2.0, 3.0, -1.5]
network = MLP(3, [4,4,1])
network(x) # forward pass

shape = (4,3)

xs = np.random.rand(*shape).tolist()
ys = np.random.rand(shape[0]).tolist()

network.optimize(xs, ys, epochs=1000, alpha=.1)

print([network(x) for x in xs])