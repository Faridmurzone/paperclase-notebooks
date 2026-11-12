import torch
# >>>
def H(p):                                  # entropía en nats
    return -(p * torch.log(p)).sum().item()

p = torch.tensor([0.50, 0.25, 0.15, 0.10])
print("sorpresa -log p:", [round(x, 2) for x in (-torch.log(p)).tolist()])

uniforme = torch.full((4,), 0.25)
picuda = torch.tensor([0.97, 0.01, 0.01, 0.01])
for nombre, q in [("uniforme", uniforme), ("p", p), ("picuda", picuda)]:
    print(f"H({nombre:8}) = {H(q):.3f} nats")
print("log 4 =", round(torch.log(torch.tensor(4.0)).item(), 3))
print("en bits, H(uniforme) =", H(uniforme) / torch.log(torch.tensor(2.0)).item())
# <<<
