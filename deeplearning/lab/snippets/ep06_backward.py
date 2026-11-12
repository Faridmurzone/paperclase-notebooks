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

# >>>
def backward(raiz):
    orden, vistos = [], set()
    def visitar(v):                 # orden topológico:
        if v not in vistos:         # cada nodo va después
            vistos.add(v)           # de todos sus hijos
            for hijo in v._prev:
                visitar(hijo)
            orden.append(v)
    visitar(raiz)
    raiz.grad = 1.0                 # dL/dL = 1
    for v in reversed(orden):       # de la salida a las hojas
        v._backward()
    return orden

a, b, c, f = Value(2.0), Value(3.0), Value(-4.0), Value(-2.0)
L = (a * b + c) * f
orden = backward(L)
print("orden:", " ".join(str(v._op or v.data) for v in orden))
# <<<
