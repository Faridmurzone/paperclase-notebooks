import torch
import torch.nn.functional as F
# >>>
y_hat = torch.tensor([2.5, 0.0, 2.0, 7.0])    # predicciones
y     = torch.tensor([3.0, -0.5, 2.0, 3.0])   # valores reales

e = y_hat - y                                 # error de cada ejemplo
print("errores:   ", e.tolist())
print("al cuadrado:", (e ** 2).tolist())
print("MSE a mano:", round((e ** 2).mean().item(), 4))
print("F.mse_loss:", round(F.mse_loss(y_hat, y).item(), 4))
print("sin el 4°: ", round((e[:3] ** 2).mean().item(), 4))
# <<<
