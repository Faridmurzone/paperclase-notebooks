import torch
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

def relu(v):
    out = Value(max(0.0, v.data), (v,), "relu")
    def _backward():
        v.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out
# >>>
# nuestra versión: una neurona con ReLU
x1, x2 = Value(1.0), Value(-2.0)
w1, w2, b = Value(0.5), Value(-1.5), Value(0.25)
y = relu(w1 * x1 + w2 * x2 + b)
backward(y)
print("nuestro:", [v.grad for v in (w1, w2, b, x1, x2)])

# PyTorch: los mismos números, requires_grad=True
T = torch.tensor([1.0, -2.0, 0.5, -1.5, 0.25], requires_grad=True)
tx1, tx2, tw1, tw2, tb = T
yt = torch.relu(tw1 * tx1 + tw2 * tx2 + tb)
yt.backward()
print("torch  :", T.grad[[2, 3, 4, 0, 1]].tolist())
print("y:", y.data, "y torch:", yt.item())
# <<<
