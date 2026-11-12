import math
import torch
import torch.nn.functional as F
torch.manual_seed(0)


def attention(Q, K, V, mask=None):
    S = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    if mask is not None:
        S = S.masked_fill(mask, float("-inf"))
    A = torch.softmax(S, dim=-1)
    return A @ V, A
# >>>
B, T, d_k = 2, 6, 64                       # batch de 2 frases
Q, K, V = (torch.randn(B, T, d_k) for _ in range(3))
prohibido = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)

nuestro, _ = attention(Q, K, V, prohibido)
torch_a = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
torch_b = F.scaled_dot_product_attention(Q, K, V, attn_mask=~prohibido)

print("vs is_causal=True:", torch.allclose(nuestro, torch_a, atol=1e-6))
print("vs attn_mask:     ", torch.allclose(nuestro, torch_b, atol=1e-6))
print("diferencia máxima:", f"{(nuestro - torch_a).abs().max():.1e}")
# <<<
