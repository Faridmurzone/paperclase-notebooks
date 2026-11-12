import numpy as np
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0)
# rasgos de una respuesta: [correcta, se cubre ("depende…"), largo]
def respuesta(n):
    return np.stack([rng.random(n) < 0.7, rng.random(n) < 0.5, rng.random(n)], 1).astype(float)
# >>>
n = 4000
dudosa = rng.random(n) < 0.9               # 90 %: preguntas sin respuesta clara
A, B = respuesta(n), respuesta(n)
def gusto(r, dudosa):                      # lo que premia quien etiqueta
    cubrirse = np.where(dudosa, +1.2, -1.0)    # humildad, salvo si es obvio
    return 2.0 * r[:, 0] + cubrirse * r[:, 1] + 0.3 * r[:, 2]
gana_A = rng.random(n) < 1 / (1 + np.exp(-(gusto(A, dudosa) - gusto(B, dudosa))))

rm = LogisticRegression().fit(A - B, gana_A)    # Bradley-Terry sobre rasgos
w = rm.coef_[0]
print("pesos del RM [correcta, se cubre, largo]:", w.round(2))
directa, cubierta = np.array([1, 0, 0.2]), np.array([1, 1, 0.8])
p = 1 / (1 + np.exp(-w @ (cubierta - directa)))
print(f"pregunta obvia: P(cubierta ≻ directa) = {p:.2f}")
# <<<
