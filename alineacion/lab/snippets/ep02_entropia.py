import torch
import torch.nn.functional as F
torch.manual_seed(0)
V, T = 6, 4                                   # vocabulario y posiciones de juguete
logits = torch.randn(T, V, requires_grad=True)
objetivo = torch.tensor([2, 0, 5, 1])         # el token que de verdad vino después
# >>>
logp = logits - logits.logsumexp(-1, keepdim=True)   # log softmax
a_mano = -logp[torch.arange(T), objetivo].mean()     # −log p del correcto
torch_ce = F.cross_entropy(logits, objetivo)
print(f"a mano:   {a_mano.item():.4f}")
print(f"F.cross_entropy: {torch_ce.item():.4f}")

torch_ce.backward()                                  # backprop
p = logits.softmax(-1).detach()
esperado = (p - F.one_hot(objetivo, V)) / T          # (p − y) / T
print("grad fila 0:", logits.grad[0].numpy().round(3))
print("iguales:", torch.allclose(logits.grad, esperado))
# <<<
