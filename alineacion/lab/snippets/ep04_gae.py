import torch
r = torch.tensor([[-0.005, 0.002, -0.030, 1.190]])    # recompensas por token (escena anterior)
v = torch.tensor([[0.40, 0.50, 0.90, 1.00]])          # lo que el crítico esperaba en cada paso
# >>>
def gae(r, v, gamma=1.0, lam=0.95):
    A, ultimo, T = torch.zeros_like(r), 0.0, r.shape[1]
    for t in reversed(range(T)):                     # de atrás para adelante
        v_sig = v[:, t + 1] if t + 1 < T else 0.0
        delta = r[:, t] + gamma * v_sig - v[:, t]    # δ_t: sorpresa de un paso
        ultimo = delta + gamma * lam * ultimo        # A_t = δ_t + γλ A_{t+1}
        A[:, t] = ultimo
    return A, A + v                                  # ventajas y retornos

A, R = gae(r, v)
print("ventajas A:", A.numpy().round(3))
print("retornos R:", R.numpy().round(3))
# <<<
