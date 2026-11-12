import math
import torch
torch.manual_seed(0)
T, d = 5, 16
X = torch.randn(T, d)
W_Q, W_K, W_V = (torch.randn(d, d) / 4 for _ in range(3))
# >>>
Q, K, V = X @ W_Q, X @ W_K, X @ W_V               # [T, d] cada uno
scores = Q @ K.T / math.sqrt(d)                    # [T, T]: quién mira a quién
mask = torch.triu(torch.ones(T, T), 1).bool()      # no mirar el futuro
scores = scores.masked_fill(mask, float("-inf"))
A = scores.softmax(dim=-1)                         # cada fila suma 1
out = A @ V                                        # mezcla de valores
print(A.round(decimals=2))
print(A.sum(dim=-1))
# <<<
