# >>>
class Value:
    def __init__(self, data, hijos=(), op=""):
        self.data = data          # el número
        self.grad = 0.0           # dL/d(este número), lo llena backward
        self._prev = tuple(hijos) # de qué Values salió
        self._op = op             # con qué operación
    def __add__(self, otro):      # a + b
        return Value(self.data + otro.data, (self, otro), "+")
    def __mul__(self, otro):      # a * b
        return Value(self.data * otro.data, (self, otro), "*")

a, b, c = Value(2.0), Value(3.0), Value(-4.0)
e = a * b                         # Python llama a.__mul__(b)
d = e + c
print("e =", e.data, " op:", e._op, " hijos:", [h.data for h in e._prev])
print("d =", d.data, " op:", d._op, " hijos:", [h.data for h in d._prev])
# <<<
