import torch
torch.manual_seed(0)
capa = torch.nn.Linear(3, 1)
x_rl, x_pre = torch.randn(4, 3), torch.randn(4, 3)
# >>>
gamma = 27.8                                        # el γ del paper (apéndice C.4)
def perdidas():                                     # dos pérdidas cualquiera
    return capa(x_rl).pow(2).mean(), capa(x_pre).abs().mean()

l_ppo, l_pre = perdidas()                           # como el paper: en pasos seguidos
l_ppo.backward()                                    # 1) gradiente de PPO
(gamma * l_pre).backward()                          # 2) se ACUMULA en .grad
acumulado = capa.weight.grad.clone()

capa.zero_grad()
l_ppo, l_pre = perdidas()
(l_ppo + gamma * l_pre).backward()                  # una sola pérdida sumada
print("acumulado:", acumulado.numpy().round(3))
print("sumado:   ", capa.weight.grad.numpy().round(3))
# <<<
