import math
import torch
import torch.nn as nn
torch.manual_seed(0)


def attention(Q, K, V, mask=None):
    S = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    if mask is not None:
        S = S.masked_fill(mask, float("-inf"))
    A = torch.softmax(S, dim=-1)
    return A @ V, A


B, T, d_model, h = 2, 5, 512, 8
X = torch.randn(B, T, d_model)
# >>>
mha = nn.MultiheadAttention(d_model, h, bias=False, batch_first=True)
W_q, W_k, W_v = mha.in_proj_weight.chunk(3)       # [512, 512] cada una
W_o = mha.out_proj.weight

def multi_head(X):
    d_k = d_model // h
    partir = lambda M: M.view(B, T, h, d_k).transpose(1, 2)
    Q, K, V = partir(X @ W_q.T), partir(X @ W_k.T), partir(X @ W_v.T)
    out, _ = attention(Q, K, V)                     # 8 heads en paralelo
    return out.transpose(1, 2).reshape(B, T, d_model) @ W_o.T

ref, _ = mha(X, X, X, need_weights=False)
print("forma:", tuple(multi_head(X).shape))
print("diferencia máxima:", f"{(multi_head(X) - ref).abs().max():.1e}")
# <<<
