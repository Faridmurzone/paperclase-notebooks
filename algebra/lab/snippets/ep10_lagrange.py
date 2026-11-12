import numpy as np
# >>>
# maximizar f(x, y) = x + y  con la restricción x² + y² = 1
ang = np.linspace(0, 2 * np.pi, 3601)
x, y = np.cos(ang), np.sin(ang)          # todos los puntos del círculo
i = np.argmax(x + y)
x0, y0 = x[i], y[i]
print(f"mejor punto: ({x0:.3f}, {y0:.3f})   f = {x0 + y0:.3f}")

grad_f = np.array([1.0, 1.0])            # ∇(x + y)
grad_g = np.array([2 * x0, 2 * y0])      # ∇(x² + y²)
lam = grad_f / grad_g                    # ∇f = λ ∇g, coordenada a coordenada
print("∇f =", grad_f, "  ∇g =", grad_g.round(3))
print("λ =", lam.round(3), "  1/√2 =", round(1 / np.sqrt(2), 3))
# <<<
