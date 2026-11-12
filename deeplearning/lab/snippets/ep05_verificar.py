import numpy as np
import torch
x_np = np.array([1.0, 2.0]); y = 2.0
W1_np = np.array([[0.5, -1.0], [1.0, 0.5]]); w2_np = np.array([1.0, 0.5])
def perdida(W1, b1, w2, b2):
    h = np.maximum(0, W1 @ x_np + b1)
    return (w2 @ h + b2 - y) ** 2
# >>>
# 1) autograd de PyTorch, con los mismos números
W1 = torch.tensor(W1_np, requires_grad=True)
w2 = torch.tensor(w2_np, requires_grad=True)
x = torch.tensor(x_np)
L = (w2 @ torch.relu(W1 @ x) - y) ** 2
L.backward()
print("dW1:", (W1.grad + 0.0).tolist())
print("dw2:", (w2.grad + 0.0).tolist())

# 2) diferencias finitas: mover un peso un poquito y medir
h = 1e-5
Wm, Wp = W1_np.copy(), W1_np.copy()
Wm[1, 1] -= h;  Wp[1, 1] += h
num = (perdida(Wp, 0, w2_np, 0) - perdida(Wm, 0, w2_np, 0)) / (2 * h)
print("dW1[1,1] mano -2.0 numérico", round(num, 4))
# <<<
