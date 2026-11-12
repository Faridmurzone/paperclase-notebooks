import torch
torch.manual_seed(0)
S = torch.randn(3, 3) * 2
# >>>
bien = torch.softmax(S, dim=-1)   # normaliza cada fila
mal = torch.softmax(S, dim=0)     # normaliza cada columna

print("dim=-1, suma por fila:   ", bien.sum(-1).numpy().round(3))
print("dim=0,  suma por fila:   ", mal.sum(-1).numpy().round(3))
print("dim=0,  suma por columna:", mal.sum(0).numpy().round(3))
# <<<
