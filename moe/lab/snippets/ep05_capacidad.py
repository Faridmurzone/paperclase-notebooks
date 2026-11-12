import math
# >>>
T, k, E = 2048, 2, 8                     # tokens del batch, top-k, expertos
for factor in (1.0, 1.25, 2.0):
    C = math.ceil(factor * T * k / E)    # capacidad de cada experto
    lugares = E * C                      # filas que se reservan
    print(f"factor {factor:<4}: C = {C:>4} | lugares {lugares:>5}"
          f" para {T * k} asignaciones")
# <<<
