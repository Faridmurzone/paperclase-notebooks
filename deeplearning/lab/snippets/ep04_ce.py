import torch
import torch.nn.functional as F
# >>>
z = torch.tensor([[2.0, 1.0, 0.1],     # ejemplo 0: logits de 3 clases
                  [0.5, 2.5, 0.3]])    # ejemplo 1
y = torch.tensor([0, 2])               # clase correcta de cada uno

p = torch.exp(z) / torch.exp(z).sum(dim=1, keepdim=True)   # softmax
print("p ej. 0:", p[0].numpy().round(2))
print("p ej. 1:", p[1].numpy().round(2))
L = -torch.log(p[torch.arange(2), y])  # -log p de la clase correcta
print("pérdidas:", L.numpy().round(3))
print("a mano:         ", round(L.mean().item(), 4))
print("F.cross_entropy:", round(F.cross_entropy(z, y).item(), 4))
# <<<
