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

def L_de(a_, b_=3.0, c_=-4.0, f_=-2.0):
    return (a_ * b_ + c_) * f_
# >>>
a = Value(2.0)
L = (a * Value(3.0) + Value(-4.0)) * Value(-2.0)
backward(L)

h = 1e-4                                   # un paso chiquito
pendiente = (L_de(2.0 + h) - L_de(2.0)) / h  # cuánto cambió L
print("autograd: dL/da =", a.grad)
print("numérico: dL/da =", round(pendiente, 6))

x = Value(3.0)
y = x * x + x                  # x se usa tres veces
backward(y)
print("x*x + x en x=3: dy/dx =", x.grad, "(2x+1 = 7)")
# <<<
