import math

class Value:

    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self.backward = lambda: None
        self.prev = set(children)
        self.op = op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __str__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def gradient():
            self.grad += out.grad
            other.grad += out.grad

        out.backward = gradient
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __neg__(self): # -self
        return self * -1

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1


    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), "*")

        def gradient():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad


        out.backward = gradient
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __pow__(self, other):
        out = Value(self.data ** other, (self, ), 'pow')

        def gradient():
            self.grad += out.grad * (other * self.data ** (other - 1))

        out.backward = gradient
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self, ), 'exp')

        def gradient():
            self.grad += out.data * out.grad

        out.backward = gradient
        return out

    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1) / (math.exp(2*n) + 1)
        out = Value(t, (self, ), 'tanh')

        def gradient():
            self.grad += (1- t**2) * out.grad

        out.backward = gradient
        return out


    def backward_pass(self):

        # build the topological graph
        visited = set()
        topo = []

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)

                topo.append(v)
        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node.backward()