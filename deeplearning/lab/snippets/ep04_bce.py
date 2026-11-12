import torch
import torch.nn.functional as F
# >>>
z = torch.tensor([2.0, -1.0, 0.5, -3.0])   # logits: lo que sale de la red
y = torch.tensor([1.0, 0.0, 0.0, 1.0])     # respuesta correcta: 0 o 1

p = torch.sigmoid(z)                       # probabilidad de "1"
L = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))
print("p:       ", p.numpy().round(2))
print("pérdidas:", L.numpy().round(2))
print("BCE a mano:", round(L.mean().item(), 4))
print("PyTorch:   ", round(F.binary_cross_entropy_with_logits(z, y).item(), 4))
# <<<
