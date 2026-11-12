import math
import torch
torch.manual_seed(0)
T, d_k, d_v = 4, 8, 8
Q, K, V = torch.randn(T, d_k), torch.randn(T, d_k), torch.randn(T, d_v)
# >>>
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    S = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # [.., T_q, T_k]
    if mask is not None:                           # True = prohibido
        S = S.masked_fill(mask, float("-inf"))
    A = torch.softmax(S, dim=-1)                   # por filas
    return A @ V, A                                # [.., T_q, d_v]

out, A = attention(Q, K, V)
print("Q", tuple(Q.shape), "K", tuple(K.shape), "V", tuple(V.shape))
print("A", tuple(A.shape), " out", tuple(out.shape))
print("suma de cada fila de A:", A.sum(-1).numpy().round(3))
# <<<
