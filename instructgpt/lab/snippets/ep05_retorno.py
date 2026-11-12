import torch
# Una respuesta de 5 tokens. Log-probs sintéticos de la política y del SFT.
lp_rl = torch.tensor([-1.2, -0.4, -2.0, -0.9, -0.3])
lp_sft = torch.tensor([-1.5, -0.6, -1.8, -1.4, -0.5])
beta, r_rm = 0.02, 1.7
# >>>
kl_tok = lp_rl - lp_sft                    # log π_RL − log π_SFT, por token
recompensa = -beta * kl_tok                # el freno, en cada token
recompensa[-1] += r_rm                     # el RM puntúa al final: un bandit
retorno = recompensa.flip(0).cumsum(0).flip(0)   # G_t = Σ_{k≥t} r_k, sin descuento

print("kl:", *[f"{v:+.2f}" for v in kl_tok])
print("r: ", *[f"{v:+.3f}" for v in recompensa])
print("G: ", *[f"{v:+.3f}" for v in retorno])
print(f"G_0 = {retorno[0]:.3f}")
print(f"r − β·Σkl = {r_rm - beta * kl_tok.sum():.3f}")
# <<<
