import numpy as np
# >>>
x = np.array([4.0, 3.0])
x_hat = x / np.linalg.norm(x)              # mismo sentido, largo 1
print("x̂ =", x_hat, "  ‖x̂‖ =", np.linalg.norm(x_hat))

y = np.array([1.0, 2.0])
y_hat = y / np.linalg.norm(y)
cos = (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))
print("coseno   =", round(cos, 3))
print("x̂ · ŷ    =", round(x_hat @ y_hat, 3))  # el mismo número

print("(10x)·y  =", (10 * x) @ y, "  vs x·y =", x @ y)
z = 10 * x
print("cos(10x, y) =", round((z / np.linalg.norm(z)) @ y_hat, 3))
# <<<
