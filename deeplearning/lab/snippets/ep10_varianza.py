import torch
torch.manual_seed(0)
# >>>
n = 256
x = torch.randn(1000, n)               # 1000 entradas, varianza 1

w = torch.randn(n, n)                  # pesos con varianza 1
print("Var(w) = 1   → std de y:", round((x @ w.T).std().item(), 2))

w = torch.randn(n, n) / n ** 0.5       # pesos con varianza 1/n
print("Var(w) = 1/n → std de y:", round((x @ w.T).std().item(), 2))
print("raíz de n =", n ** 0.5)
# <<<
