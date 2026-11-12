import math
import torch
torch.manual_seed(0)
d = 8


def attention(Q, K, V, mask=None):
    S = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    if mask is not None:
        S = S.masked_fill(mask, float("-inf"))
    A = torch.softmax(S, dim=-1)
    return A @ V, A


W_q, W_k, W_v = (torch.randn(d, d) / d ** 0.5 for _ in range(3))
# >>>
memoria = torch.randn(5, d)        # salida del encoder: 5 tokens de entrada
dec = torch.randn(3, d)            # estado del decoder: 3 tokens escritos

Q = dec @ W_q                      # queries: del decoder
K, V = memoria @ W_k, memoria @ W_v   # keys y values: del encoder
out, A = attention(Q, K, V)        # sin máscara: la entrada es toda conocida
print("Q", tuple(Q.shape), " K", tuple(K.shape), " V", tuple(V.shape))
print("A", tuple(A.shape), " out", tuple(out.shape))
print("fila 0 de A:", A[0].numpy().round(2))
# <<<
