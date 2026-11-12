import torch

# >>>
theta = torch.tensor([2.0, -1.0, 0.5], requires_grad=True)
lam = 0.1                                       # λ: fuerza del L2

penal = lam / 2 * (theta ** 2).sum()            # (λ/2)·‖θ‖²
penal.backward()

print("θ        :", theta.tolist())
print("λ/2·‖θ‖² :", round(penal.item(), 4))
print("gradiente:", [round(x, 4) for x in theta.grad.tolist()])
print("λ·θ      :", [round(lam * x, 4) for x in theta.tolist()])
# <<<
