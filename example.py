from neural_network import MLP
import numpy as np

x = [2.0, 3.0, -1.5]
n = MLP(3, [4,4,1])
n(x) # forward pass

shape = (4,3)

xs = np.random.rand(*shape).tolist()
ys = np.random.rand(shape[0]).tolist()

y_preds = [n(x) for x in xs]


loss = sum([(y_pred - y_true) ** 2 for y_true, y_pred in zip(ys, y_preds)])

loss.backward()