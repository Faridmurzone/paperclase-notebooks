import torch
torch.set_printoptions(precision=2, sci_mode=False)
# >>>
w = torch.tensor([0.12, -0.31, 0.05, 0.44, -0.08, 0.27, -0.50, 0.19])
absmax = w.abs().max()                       # la constante del bloque
codigo = torch.round(w / absmax * 7)         # enteros de −7 a 7: 4 bits
w_hat = codigo / 7 * absmax                  # de vuelta a números reales
print("absmax:", absmax.item())
print("código:", codigo.int().tolist())
print("vuelta:", w_hat)
print(f"error máx: {(w - w_hat).abs().max():.3f}")

w[3] = 4.0                                   # un valor atípico en el bloque
absmax = w.abs().max()
codigo = torch.round(w / absmax * 7)
print("con atípico:", codigo.int().tolist())
# <<<
