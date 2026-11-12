import torch
torch.manual_seed(0)
score = torch.tensor([1.2])                           # lo que dijo el RM de la respuesta entera
kl_tok = torch.tensor([[0.05, -0.02, 0.30, 0.10]])    # log π − log π_ref, token a token
# >>>
def recompensa(score, kl_tok, beta=0.1):
    r = -beta * kl_tok                  # penalización en cada token
    r[:, -1] += score                   # el RM puntúa al final
    return r

r = recompensa(score, kl_tok)
print("r por token:", r.numpy().round(3))
print(f"total: {r.sum():.3f} = {score.item():.1f} − 0.1 · {kl_tok.sum():.2f}")
# <<<
