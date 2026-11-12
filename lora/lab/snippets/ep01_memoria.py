import json, os
c = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep01_conteo.json")))
# >>>
N = c["total"]                              # 2.614.341.888 parámetros
b = 4                                       # float32: 4 bytes
pesos = N * b
grads = N * b                               # un gradiente por peso
m = N * b                                   # Adam: promedio del gradiente
v = N * b                                   # Adam: promedio del cuadrado
GB = 1e9
for nombre, x in [("pesos", pesos), ("gradientes", grads),
                  ("Adam m", m), ("Adam v", v)]:
    print(f"{nombre:11} {x / GB:5.1f} GB")
print(f"{'total':11} {(pesos + grads + m + v) / GB:5.1f} GB")
print("bytes por parámetro:", 4 * b)
# <<<
