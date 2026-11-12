import torch
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16, dtype=torch.float32); y = torch.tensor(y)
muertos = X.abs().sum(0) == 0                  # píxeles que valen 0 en todas las imágenes

# >>>
def entrenar(Clase, pasos=1000):
    torch.manual_seed(0)
    red = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 10))
    W = red[0].weight                          # 32 × 64: una columna por píxel
    w0 = W.detach().clone()
    opt = Clase(red.parameters(), lr=1e-3, weight_decay=0.1)
    for t in range(pasos):
        b = torch.randint(0, len(X), (64,))
        opt.zero_grad()
        torch.nn.functional.cross_entropy(red(X[b]), y[b]).backward()
        opt.step()
    r = W.detach().norm(dim=0) / w0.norm(dim=0)    # cuánto quedó de cada columna
    return r[muertos].mean(), r[~muertos].mean()
print("píxeles siempre en 0:", int(muertos.sum()), "de 64")
for Clase in [torch.optim.Adam, torch.optim.AdamW]:
    m, v = entrenar(Clase)
    print(f"{Clase.__name__:5}  queda: muertos {m:.2f}  vivos {v:.2f}")
# <<<
