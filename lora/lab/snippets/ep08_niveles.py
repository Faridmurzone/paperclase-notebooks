import torch
# >>>
N = torch.distributions.Normal(0, 1)
d = 0.03                                     # recorte de juguete en las colas
neg = N.icdf(torch.linspace(d, 0.5, 8))      # 8 cuantiles: −… hasta 0
pos = N.icdf(torch.linspace(0.5, 1 - d, 9))  # 9 cuantiles: 0 hasta +…
nf4 = torch.cat([neg, pos[1:]])              # 16 niveles, un solo cero
nf4 = nf4 / nf4.abs().max()                  # normalizados a [−1, 1]

int4 = torch.arange(-7, 8) / 7               # los 15 niveles uniformes
print("NF4 de juguete:", len(nf4), "niveles")
for i in range(0, 16, 4):
    print("  ", " ".join(f"{v:+.2f}" for v in nf4[i:i + 4]))
print("int4:", len(int4), "niveles, paso", round(1 / 7, 3))
# <<<
