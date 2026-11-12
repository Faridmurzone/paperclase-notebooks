import numpy as np
# >>>
q1 = np.array([0.6, 0.8])              # r̂
q2 = np.array([-0.8, 0.6])             # r̂ girado 90°
print("q1·q2 =", q1 @ q2, "  ‖q1‖ =", np.linalg.norm(q1))

x = np.array([2.0, 1.0])
c1, c2 = x @ q1, x @ q2                # coordenadas = productos punto
print("coordenadas:", round(c1, 2), round(c2, 2))
print("c1 q1 + c2 q2 =", (c1 * q1 + c2 * q2).round(2))

Q = np.column_stack([q1, q2])          # la base, como columnas
print("Qᵀ Q =\n", (Q.T @ Q).round(2))
print("‖Qᵀ x‖ =", round(np.linalg.norm(Q.T @ x), 3),
      "  ‖x‖ =", round(np.linalg.norm(x), 3))
# <<<
