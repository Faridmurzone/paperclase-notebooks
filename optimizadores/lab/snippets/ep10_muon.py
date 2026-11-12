import torch
torch.manual_seed(0)

# >>>
def newton_schulz(G, pasos=5):                       # del blog de Keller Jordan
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / G.norm()                                  # valores singulares ≤ 1
    for _ in range(pasos):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X           # polinomio de grado 5
    return X

G = torch.randn(4, 6) * torch.tensor([[10.0], [3.0], [1.0], [0.1]])
sv = lambda M: [round(x, 2) for x in torch.linalg.svdvals(M).tolist()]
print("valores singulares de G:  ", sv(G))
print("después de Newton–Schulz: ", sv(newton_schulz(G)))
U, S, Vt = torch.linalg.svd(G, full_matrices=False)
cos = torch.nn.functional.cosine_similarity(
    newton_schulz(G).flatten(), (U @ Vt).flatten(), dim=0)
print(f"coseno con U·Vᵀ: {cos:.3f}")
# <<<
