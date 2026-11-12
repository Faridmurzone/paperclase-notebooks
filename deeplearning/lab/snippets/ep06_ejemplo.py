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
a, b = Value(2.0), Value(3.0)
c, f = Value(-4.0), Value(-2.0)
e = a * b          # 6
d = e + c          # 2
L = d * f          # -4
backward(L)
for nombre, v in [("L", L), ("d", d), ("f", f), ("e", e),
                  ("c", c), ("a", a), ("b", b)]:
    print(f"{nombre}: valor={v.data:5}  grad={v.grad:5}")
# <<<
