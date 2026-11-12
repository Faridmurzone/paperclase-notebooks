class Value:
    def __init__(self, data, hijos=(), op=""):
        self.data, self.grad = data, 0.0
        self._prev, self._op = tuple(hijos), op
        self._backward = lambda: None
    def __add__(self, otro):
        out = Value(self.data + otro.data, (self, otro), "+")
        def _backward():
            self.grad += out.grad
            otro.grad += out.grad
        out._backward = _backward
        return out
    def __mul__(self, otro):
        out = Value(self.data * otro.data, (self, otro), "*")
        def _backward():
            self.grad += otro.data * out.grad
            otro.grad += self.data * out.grad
        out._backward = _backward
        return out

def backward(raiz):
    orden, vistos = [], set()
    def visitar(v):
        if v not in vistos:
            vistos.add(v)
            for hijo in v._prev:
                visitar(hijo)
            orden.append(v)
    visitar(raiz)
    raiz.grad = 1.0
    for v in reversed(orden):
        v._backward()

# >>>
def relu(v):
    out = Value(max(0.0, v.data), (v,), "relu")   # forward
    def _backward():                               # derivada local:
        v.grad += (out.data > 0) * out.grad        # 1 si pasó, 0 si no
    out._backward = _backward
    return out

for z0 in (2.5, -1.0):
    z = Value(z0)
    y = relu(z)
    backward(y)
    print(f"z = {z0:4}  relu = {y.data}  dy/dz = {z.grad}")
# <<<
