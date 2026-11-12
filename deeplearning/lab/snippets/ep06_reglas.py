class Value:
    def __init__(self, data, hijos=(), op=""):
        self.data, self.grad = data, 0.0
        self._prev, self._op = tuple(hijos), op
        self._backward = lambda: None
# >>>
    def __mul__(self, otro):
        out = Value(self.data * otro.data, (self, otro), "*")
        def _backward():                      # regla local de *
            self.grad += otro.data * out.grad     # d(ab)/da = b
            otro.grad += self.data * out.grad     # d(ab)/db = a
        out._backward = _backward             # se guarda, no se corre
        return out
    def __add__(self, otro):
        out = Value(self.data + otro.data, (self, otro), "+")
        def _backward():                      # regla local de +
            self.grad += out.grad                 # d(a+b)/da = 1
            otro.grad += out.grad                 # d(a+b)/db = 1
        out._backward = _backward
        return out
a, b = Value(2.0), Value(3.0)
e = a * b
e.grad = 1.0          # dL/de, supongamos que vale 1
e._backward()         # reparte hacia a y b
print("a.grad =", a.grad, "  b.grad =", b.grad)
# <<<
