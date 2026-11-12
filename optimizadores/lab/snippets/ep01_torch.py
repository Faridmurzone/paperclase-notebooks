import numpy as np
import torch

# >>>
w = torch.tensor([-4.5, 1.5], requires_grad=True)   # PyTorch
opt = torch.optim.SGD([w], lr=0.3)
mano = np.array([-4.5, 1.5])                        # a mano, NumPy

for t in range(20):
    opt.zero_grad()
    perdida = 0.1 * w[0] ** 2 + 3 * w[1] ** 2
    perdida.backward()                              # w.grad = (0,2x, 6y)
    opt.step()                                      # w -= lr * w.grad
    mano = mano - 0.3 * np.array([0.2, 6.0]) * mano

print("torch:", w.detach().numpy().round(4))
print("mano: ", mano.round(4))
dif = np.abs(w.detach().numpy() - mano).max()
print(f"diferencia máxima: {dif:.1e}")
# <<<
