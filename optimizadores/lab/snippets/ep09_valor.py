import torch
F = torch.nn.functional

# >>>
g = torch.tensor([4.0, 2.0])                         # un gradiente grande
p1 = torch.zeros(2, requires_grad=True); p1.grad = g.clone()
p2 = torch.zeros(2, requires_grad=True); p2.grad = g.clone()

torch.nn.utils.clip_grad_value_([p1], clip_value=2.0)  # cada número a [−2, 2]
torch.nn.utils.clip_grad_norm_([p2], max_norm=2.0)     # todo el vector, escalado

for nombre, p in [("por valor", p1), ("por norma", p2)]:
    cos = F.cosine_similarity(p.grad, g, dim=0).item()
    print(f"{nombre}: {[round(x, 3) for x in p.grad.tolist()]}"
          f"  coseno con g = {cos:.3f}")
# <<<
