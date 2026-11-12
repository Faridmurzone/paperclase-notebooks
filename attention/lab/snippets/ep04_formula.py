import math
import torch
torch.set_printoptions(precision=3, sci_mode=False)
# >>>
def positional_encoding(n_pos, d):
    pos = torch.arange(n_pos).unsqueeze(1)          # [n_pos, 1]
    i2 = torch.arange(0, d, 2)                      # 0, 2, 4, …, d−2
    omega = 1 / 10000 ** (i2 / d)                   # una frecuencia por par
    PE = torch.zeros(n_pos, d)
    PE[:, 0::2] = torch.sin(pos * omega)            # columnas pares
    PE[:, 1::2] = torch.cos(pos * omega)            # columnas impares
    return PE, omega

PE, omega = positional_encoding(100, 512)
print("PE", tuple(PE.shape))
print("pos 0, dims 0–3:", PE[0, :4])
print("pos 1, dims 0–3:", PE[1, :4])
print(f"longitud de onda: {2 * math.pi / omega[0]:.2f}"
      f" … {2 * math.pi / omega[-1]:,.0f}")
# <<<
