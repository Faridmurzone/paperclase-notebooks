import torch
torch.manual_seed(0)
# >>>
pi_ref = torch.tensor([0.40, 0.30, 0.15, 0.10, 0.05])  # 5 respuestas posibles
r = torch.tensor([0.0, 1.0, 2.0, 0.5, 3.0])             # recompensa de cada una
beta = 1.0

def objetivo(pi):                        # E[r] − β·KL(π ‖ π_ref)
    return (pi * r).sum() - beta * (pi * (pi / pi_ref).log()).sum()

pi_opt = pi_ref * torch.exp(r / beta)    # π_ref · e^(r/β)
Z = pi_opt.sum()
pi_opt = pi_opt / Z                      # normalizar
print("π* =", [round(x, 3) for x in pi_opt.tolist()])
print(f"objetivo(π*) = {objetivo(pi_opt):.4f}")
print(f"β·log Z      = {beta * Z.log():.4f}")
azar = (torch.softmax(3 * torch.randn(5), 0) for _ in range(10000))
print(f"mejor de 10000 al azar = {max(objetivo(p) for p in azar):.4f}")
# <<<
