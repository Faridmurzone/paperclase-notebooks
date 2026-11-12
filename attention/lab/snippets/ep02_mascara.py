import math
import torch
torch.manual_seed(0)
torch.set_printoptions(precision=2)


def attention(Q, K, V, mask=None):
    S = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    if mask is not None:
        S = S.masked_fill(mask, float("-inf"))
    A = torch.softmax(S, dim=-1)
    return A @ V, A


T, d = 4, 8
X = torch.randn(T, d)
# >>>
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
out, A = attention(X, X, X, mask)          # self-attention: Q = K = V = X
print(A)

X2 = X.clone()
X2[3] = torch.randn(d)                     # cambiamos el ÚLTIMO token
out2, _ = attention(X2, X2, X2, mask)
print("tokens 0–2 iguales:", torch.allclose(out[:3], out2[:3]))
print("token 3 igual:     ", torch.allclose(out[3], out2[3]))
# <<<
