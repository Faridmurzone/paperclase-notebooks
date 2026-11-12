import torch

# >>>
a = torch.zeros(2, requires_grad=True); a.grad = torch.tensor([3.0, 4.0])
b = torch.zeros(1, requires_grad=True); b.grad = torch.tensor([12.0])

# a mano: una sola norma para todos los parámetros juntos
total = torch.cat([a.grad, b.grad]).norm()           # √(3² + 4² + 12²)
factor = min(1.0, 1.0 / (total.item() + 1e-6))
print("norma total:", total.item(), " factor:", round(factor, 4))
print("a mano:", [round(x * factor, 4) for x in [3.0, 4.0, 12.0]])

devuelto = torch.nn.utils.clip_grad_norm_([a, b], max_norm=1.0)
print("torch :", [round(x, 4) for x in torch.cat([a.grad, b.grad]).tolist()])
print("clip_grad_norm_ devolvió:", devuelto.item())
# <<<
