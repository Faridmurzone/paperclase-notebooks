import torch
torch.manual_seed(0)
m, n = 2048, 2304                           # la forma de q_proj de Gemma
# >>>
r = 8
B = torch.randn(m, r) * 0.01                # m × r: alta y flaca
A = torch.randn(r, n) * 0.01                # r × n: baja y ancha
delta = B @ A                               # el parche: m × n

S = torch.linalg.svdvals(delta)
print("10 primeros sigma (x1000):", (S[:10] * 1e3).round().int().tolist())
print("sigma no nulos:", int((S > 1e-4 * S[0]).sum()))
print(f"params del parche: {r * (m + n):,}")
print(f"params de W:       {m * n:,}")
print(f"proporción: {r * (m + n) / (m * n):.2%}")
# <<<
