import torch
import torch.nn as nn
torch.manual_seed(0)
B, T, d_model, h = 1, 5, 512, 8
X = torch.randn(B, T, d_model)
W_q = nn.Linear(d_model, d_model, bias=False)
# >>>
d_k = d_model // h                          # 512 / 8 = 64
Q = W_q(X)                                  # [B, T, d_model]
print("Q              ", tuple(Q.shape))
Q = Q.view(B, T, h, d_k)                    # partir las 512 columnas
print("view           ", tuple(Q.shape))
Q = Q.transpose(1, 2)                       # heads adelante
print("transpose(1, 2)", tuple(Q.shape))
S = Q @ Q.transpose(-2, -1) / d_k ** 0.5    # una T×T por head
print("scores         ", tuple(S.shape))
out = S.softmax(-1) @ Q                     # [B, h, T, d_k]
out = out.transpose(1, 2).reshape(B, T, d_model)   # concat
print("concat         ", tuple(out.shape))
# <<<
