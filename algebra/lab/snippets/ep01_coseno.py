import numpy as np
# >>>
def coseno(u, v):
    return (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))

a = np.array([3.0, 1.0])
otros = {"b": [1.0, 2.0], "c": [-1.0, 3.0], "d": [-6.0, -2.0]}

for nombre, v in otros.items():
    v = np.array(v)
    cos = coseno(a, v)
    grados = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    print(f"a·{nombre} = {a @ v:+5.1f}   cos = {cos:+.3f}   ángulo = {grados:5.1f}°")
# <<<
