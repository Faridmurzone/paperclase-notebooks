import torch
torch.manual_seed(0)
# >>>
logits = torch.tensor([[4.0, 1.0, 0.5, -1.0],    # 2 tokens, 4 expertos
                       [2.0, 2.0, 0.0, 0.0]], requires_grad=True)
lse = torch.logsumexp(logits, dim=-1)            # log Σ_j e^{x_j}
z = (lse ** 2).mean()                            # router z-loss
z.backward()
print("LSE:", [round(v, 3) for v in lse.tolist()], "· z =", f"{z:.3f}")

B = logits.shape[0]
analitico = 2 / B * lse[:, None] * logits.softmax(-1)   # (2/B)·LSE·p
print("máx |grad − fórmula| =", f"{(logits.grad - analitico).abs().max():.1e}")
print("grad fila 0:", [round(v, 3) for v in logits.grad[0].tolist()])

corrido = logits.detach() - 3                     # restar 3 a todos
print("misma softmax:", torch.allclose(corrido.softmax(-1), logits.softmax(-1)))
print("z corrido =", f"{(torch.logsumexp(corrido, -1) ** 2).mean():.3f}")
# <<<
